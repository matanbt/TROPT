from typing import Annotated, List, Literal, Optional

import numpy as np
import tiktoken
import torch
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import BatchEncoding

from tropt.common import (
    OPTIMIZED_TRIGGER_PLACEHOLDER,
    ModelOutput,
    Targets,
    TextTemplates,
)
from tropt.model import (
    BaseTokenizer,
    DefaultTokenInputManager,
    EncoderBaseModel,
    LossTextAccessMixin,
    TokenAccessMixin,
)


# --------------------------------------------------------------------------
## OpenAI Tokenizer
#  Matches HuggingFace interface to OpenAI's tokenizer
#  [From https://github.com/chawins/pal/blob/main/src/models/openai.py]
# --------------------------------------------------------------------------
class OpenAITokenizer(BaseTokenizer):
    """
    A wrapper around OpenAI's tokenizer that mimics the HuggingFace interface.
    """
    def __init__(self, model_name: str) -> None:
        # Get the tokeniser corresponding to a specific model in the OpenAI API
        try:
            self._encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback for newer models or fine-tunes not in tiktoken yet
            self._encoding = tiktoken.get_encoding("cl100k_base")

        # Set interface to match HuggingFace
        self.bos_token_id = self._encoding.eot_token
        self.eos_token_id = self._encoding.eot_token
        self.pad_token_id = self._encoding.eot_token
        self.unk_token_id = self._encoding.eot_token
        self.eot_token = self._encoding.decode([self._encoding.eot_token])

    @property
    def vocab_size(self) -> int:
        return self._encoding.max_token_value + 1

    @property
    def name_or_path(self) -> str:
        return f"openai-{self._encoding.name}"

    @property
    def all_special_ids(self) -> List[int]:
        return [
            self(tok, return_tensors="list").input_ids[0]
            for tok in list(self._encoding.special_tokens_set)
        ]

    def __call__(
        self,
        text: List[str],
        return_tensors: Literal["list", "pt", "np"] = "list",
        **kwargs,
    ) -> BatchEncoding:
        _ = kwargs  # unused
        assert isinstance(text, list), "BaseTokenizer.__call__ requires a list of strings."

        # Encode all special tokens as normal text
        _ids = self._encoding.encode_batch(text, disallowed_special=())
        max_len = max(len(i) for i in _ids)
        input_ids = np.zeros((len(_ids), max_len), dtype=np.int64)
        input_ids += self.pad_token_id
        for i, _id in enumerate(_ids):
            input_ids[i, : len(_id)] = _id

        if return_tensors == "pt":
            input_ids = torch.from_numpy(input_ids)
        elif return_tensors == "list":
            input_ids = input_ids.tolist()
        return BatchEncoding({"input_ids": input_ids})

    def encode(self, text: str, **kwargs) -> List[int]:
        _ = kwargs  # unused
        return self._encoding.encode(text, disallowed_special=())

    def _parse_ids(self, ids):
        return ids

    def decode(self, ids, **kwargs) -> str:
        _ = kwargs  # unused
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        assert isinstance(ids, list) and isinstance(
            ids[0], int
        ), f"ids must be list or int, got {type(ids)} {ids}"
        # Single-token fast path: decode_single_token_bytes is ~10x faster than
        # decode() when iterating the full vocabulary (e.g. in TokenConstraints).
        if len(ids) == 1:
            try:
                return self._encoding.decode_single_token_bytes(ids[0]).decode(
                    "utf-8", errors="replace"
                ).replace(self.eot_token, "")
            except KeyError:
                raise KeyError(f"Invalid token for decoding: {ids[0]}")
        decoded = self._encoding.decode(ids)
        return decoded.replace(self.eot_token, "")

    def batch_decode(self, ids: list | int | torch.Tensor, **kwargs) -> list[str]:
        _ = kwargs  # unused
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [[ids]]
        assert isinstance(ids, list)
        if len(ids) > 0 and isinstance(ids[0], int):
            ids = [ids]
        batch: list[list[int]] = ids
        decoded_list = self._encoding.decode_batch(batch)
        decoded_list = [s.replace(self.eot_token, "") for s in decoded_list]
        return decoded_list


# --------------------------------------------------------------------------
# OpenAI Encoder Model
# --------------------------------------------------------------------------

class EncoderOpenAIModel(
    EncoderBaseModel,
    LossTextAccessMixin,
    TokenAccessMixin,  # tokenizer access, but not loss access on it
):
    """
    OpenAI Encoder model wrapper for embedding generation via the OpenAI API.
    https://platform.openai.com/docs/guides/embeddings
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        d_model: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes the OpenAI Encoder Model wrapper.

        Args:
            model_name: The name of the OpenAI embedding model to use.
            d_model: The dimensionality of the embeddings. If None, it is deduced via a dummy API call.
            api_key: The OpenAI API key. If None, it will be read from the OPENAI_API_KEY environment variable.
            base_url: Optional base URL for the OpenAI client. If None, the default OpenAI API URL is used.
        """
        # Import openai only when instantiating (optional dependency)
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

        if d_model is None:
            # Deduce d_model via a dummy request
            try:
                # We use a single token to check the dimensionality
                response = self._client.embeddings.create(
                    input="test",
                    model=self.model_name,
                )
                d_model = len(response.data[0].embedding)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to deduce d_model for {model_name}. "
                ) from e

        self._d_model = d_model

        # Get the tokenizer
        self._tokenizer = OpenAITokenizer(self.model_name)

    @property
    def d_model(self):
        return self._d_model

    @property
    def tokenizer(self) -> OpenAITokenizer:
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def invoke_from_texts(
        self,
        input_texts: Annotated[List[str], "n_texts"],
        **kwargs
    ) -> ModelOutput:
        """
        Compute embeddings for the given texts using the OpenAI API.

        Args:
            input_texts: A list of strings to embed.

        Returns:
            ModelOutput with output_embeddings populated.
        """
        # Note: OpenAI's API handles batches of texts
        response = self._client.embeddings.create(
            input=input_texts,
            model=self.model_name,
        )

        embeddings = [data.embedding for data in response.data]
        result = torch.tensor(embeddings, dtype=torch.float32)

        # Update token usage
        self._update_invoke_stats(
            n_tokens=response.usage.total_tokens,
            n_samples=len(input_texts),
        )

        return ModelOutput(output_embeddings=result)

    def set_inputs_from_tokens(
        self,
        templates: TextTemplates,
        targets: Optional[Targets] = None,
    ) -> None:
        """
        Prepares and stores the inputs manager from raw texts.
        """
        assert isinstance(templates, list), "templates must be a list of strings."

        # 1. Tokenize the template texts
        tok_results = self._tokenizer(templates, return_tensors="list")
        tok_ids = tok_results["input_ids"]

        # 2. Build the Manager and store it
        self._token_input_manager = DefaultTokenInputManager(
            tokenizer=self._tokenizer,
            templates_ids=tok_ids,
            targets=targets,
        )



