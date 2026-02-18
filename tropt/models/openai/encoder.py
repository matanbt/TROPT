import numpy as np
import torch
import tiktoken
from typing import Annotated, Any, List, Literal, Optional
from openai import OpenAI
from jaxtyping import Float, Int
from tenacity import retry, stop_after_attempt, wait_exponential
from torch import Tensor
from transformers import BatchEncoding

from tropt.common import (
    OPTIMIZED_TRIGGER_PLACEHOLDER,
    ModelInput,
    ModelOutput,
    Targets,
)
from tropt.models import (
    BaseTokenizer,
    EncoderBaseModel,
    LossTextAccessMixin,
    TokenAccessMixin,
    TokenInputManager,
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
        text: str,
        return_tensors: Literal["list", "pt", "np"] = "list",
        **kwargs,
    ) -> BatchEncoding:
        _ = kwargs  # unused
        if text is None:
            return BatchEncoding()

        if isinstance(text, list):
            # Encode all special tokens as normal text
            _ids = self._encoding.encode_batch(text, disallowed_special=())
            max_len = max(len(i) for i in _ids)
            input_ids = np.zeros((len(_ids), max_len), dtype=np.int64)
            input_ids += self.pad_token_id
            for i, _id in enumerate(_ids):
                input_ids[i, : len(_id)] = _id
        else:
            input_ids = self._encoding.encode(text, disallowed_special=())
            input_ids = np.array(input_ids, dtype=np.int64)

        if return_tensors == "pt":
            input_ids = torch.from_numpy(input_ids)
        elif return_tensors == "list":
            input_ids = input_ids.tolist()
        return BatchEncoding({"input_ids": input_ids})
    
    def encode(self, text, **kwargs):
        return self(text, **kwargs).input_ids

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
        decoded = self._encoding.decode(ids)
        return decoded.replace(self.eot_token, "")

    def batch_decode(self, ids, **kwargs) -> list[str]:
        _ = kwargs  # unused
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [[ids]]
        if isinstance(ids, list) and isinstance(ids[0], int):
            ids = [ids]
        assert (
            isinstance(ids, list)
            and isinstance(ids[0], list)
            and isinstance(ids[0][0], int)
        ), f"ids must be list of list of int, got {type(ids)} {ids}"
        decoded_list = self._encoding.decode_batch(ids)
        decoded_list = [s.replace(self.eot_token, "") for s in decoded_list]
        return decoded_list


# --------------------------------------------------------------------------
## OpenAI Token Inputs Manager:
# --------------------------------------------------------------------------
class OpenAITokenInputManager(TokenInputManager):
    """
    Inputs manager for OpenAI models (or other black-box API models).
    Instead of managing embeddings/tensors, this manages text reconstruction 
    from token-level triggers to feed into the API.
    """

    def __init__(
        self,
        tokenizer: Any, # The OpenAITokenizer wrapper
        tok_ids: List[List[int]],
        optimized_trigger_placeholder: str = OPTIMIZED_TRIGGER_PLACEHOLDER,
        targets: Targets = None,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        
        # 1. Prepare Text Templates
        # We decode the input tokens back to text to split them by the placeholder.
        # This allows us to insert the decoded trigger string later.
        raw_texts = tokenizer.batch_decode(tok_ids)
        self.before_texts = []
        self.after_texts = []
        
        for text in raw_texts:
            assert text.count(optimized_trigger_placeholder) == 1, f"Text must contain exactly one placeholder '{optimized_trigger_placeholder}'"

            # Split only on the first occurrence
            bef, aft = text.split(optimized_trigger_placeholder, 1)
            self.before_texts.append(bef)
            self.after_texts.append(aft)

        self.n_templates = len(raw_texts)

        # 2. Prepare Targets
        self.targets = targets

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def get_triggered_inputs(
        self,
        trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"],
        chosen_template_idx: Optional[int],
        **kwargs
    ) -> ModelInput:
        """
        Constructs the full text inputs for the API by decoding the candidate trigger tokens
        and inserting them into the templates.
        """
        # 1. Decode the candidates (Token IDs -> Strings)
        # trigger_ids shape: (n_candidates, trigger_len)
        trigger_strs = self.tokenizer.batch_decode(trigger_ids, skip_special_tokens=True)
        n_candidates = len(trigger_strs)
        
        # 2. Construct the full texts
        bef = self.before_texts[chosen_template_idx]
        aft = self.after_texts[chosen_template_idx]
        
        # Create list of strings for this message across all candidates
        curr_message_candidates = [
            f"{bef}{trig}{aft}" for trig in trigger_strs
        ]

        # 3. Handle Targets (select chosen message)
        targets = self.targets.select_message(chosen_template_idx)
        
        # 4. Build ModelInput
        return ModelInput(
            trigger_ids=trigger_ids,
            trigger_strs=trigger_strs,
            input_texts=curr_message_candidates,
            targets=targets
        )


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
        api_key: str = None,
        base_url: str = None,
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
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

        if d_model is None:
            # Deduce d_model via a dummy request
            try:
                # We use a single token to check the dimensionality
                response = self.client.embeddings.create(
                    input="test",
                    model=self.model_name,
                )
                d_model = len(response.data[0].embedding)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to deduce d_model for {model_name}. "
                    f"Please specify d_model explicitly or check your API connection. Error: {e}"
                ) from e

        self.d_model = d_model

        # Get the tokenizer
        self._tokenizer = OpenAITokenizer(self.model_name)

    @property
    def tokenizer(self) -> OpenAITokenizer:
        return self._tokenizer

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def __call__(
        self,
        texts: Annotated[List[str], "n_texts"],
        return_full_output: bool = False,
        **kwargs
    ) -> Float[Tensor, "n_texts d_model"] | ModelOutput:
        """
        Generates embeddings for the given texts using the OpenAI API.

        Args:
            texts: A list of strings to embed.

        Returns:
            A tensor containing the generated embeddings.
        """
        # Note: OpenAI's API handles batches of texts
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
            **kwargs
        )

        embeddings = [data.embedding for data in response.data]
        result = torch.tensor(embeddings, dtype=torch.float32)

        # Update token usage
        self._update_usage_stats(
            tokens=response.usage.total_tokens,
            forward_calls=1,
            forward_samples=len(texts)
        )

        if return_full_output:
            return ModelOutput(
                output_embeddings=result,
            )

        return result

    def set_token_inputs(
        self,
        templates: TextTemplates,
        targets: Targets = None,
    ) -> None:
        """
        Prepares and stores the inputs manager from raw texts.
        """
        assert isinstance(templates, list), "templates must be a list of strings."

        # 1. Tokenize the template texts
        tok_results = self.tokenizer(templates, return_tensors="list")
        tok_ids = tok_results["input_ids"]

        # 2. Build the Manager and store it
        self.token_input_manager = OpenAITokenInputManager(
            tokenizer=self.tokenizer,
            tok_ids=tok_ids,
            optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER,
            targets=targets,
        )



