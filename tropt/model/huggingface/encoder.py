
import logging
from functools import cached_property
from typing import Annotated, List, Optional

import torch
from jaxtyping import Float
from sentence_transformers import SentenceTransformer
from torch import Tensor
from transformers import PreTrainedModel

from tropt.common import (
    ModelOutput,
    Targets,
    TextTemplates,
)
from tropt.model import (
    EncoderBaseModel,
    GradientTokenAccessMixin,
    LossTextAccessMixin,
    LossTokenAccessMixin,
)
from tropt.model.huggingface.base import (
    HuggingFaceBackendModel,
    HuggingFaceTokenInputManager,
)
from tropt.model.model_mixins import GradientEmbedAccessMixin

logger = logging.getLogger(__name__)


# ======================= Model logic =======================


class EncoderHFModel(
    # HF backend first so its `device`/`dtype` win MRO over `BaseModel`'s defaults:
    HuggingFaceBackendModel,
    EncoderBaseModel,
    # token-level access mixins:
    LossTokenAccessMixin,
    GradientTokenAccessMixin,
    GradientEmbedAccessMixin,
    # text-level access mixins:
    LossTextAccessMixin,
):
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str| torch.dtype] = None,
        forward_pass_batch_size: int = 512,
        backward_pass_batch_size: int = 28,
        loaded_model: Optional[SentenceTransformer] = None,
        set_model_to_train: bool = False,
        **kwargs,
    ):
        """
        Wrapper for HuggingFace Sentence Transformer Encoder Model.

        Args:
            model_name (str): Name of the HuggingFace model. (irrelevant if `loaded_model` is provided)
            device (str): Device to load the model onto. If None, defaults to 'cuda' if available else 'cpu'.
            dtype (str or torch.dtype): Data type for the model. If None, uses the model's default dtype.
            forward_pass_batch_size (int): Batch size for forward passes.
            backward_pass_batch_size (int): Batch size for backward passes.
            loaded_model (SentenceTransformer, optional): Pre-loaded SentenceTransformer model.
            set_model_to_train (bool): Keep the model trainable (train mode + unfrozen weights). Default False (eval + frozen).
            **kwargs: Additional arguments for SentenceTransformer.
        """
        if loaded_model is not None:
            assert isinstance(loaded_model, SentenceTransformer), "loaded_model must be a SentenceTransformer instance."
            self._model = loaded_model
        else:
            try:
                self._model = SentenceTransformer(
                    model_name,
                    device=device,
                    model_kwargs=dict(dtype=dtype or "auto"),
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Error loading model `{model_name}`. Please make sure you load the model properly per the HuggingFace model card (e.g., you might need to pass `trust_remote_code=True` to `{self.__class__.__name__}`): {e}")
                raise e

        # Add tokenizer and embedding layer:
        self._tokenizer = self._model.tokenizer
        self._embedding_layer = self._get_input_embeddings()

        logger.warning("[General Warning:] Common embedding models often require an instruction prefix (e.g., `query: `). For optimal performance, please make sure a suitable one is applied in the textual input templates.")

    @cached_property
    def _hf_model(self) -> PreTrainedModel:
        """Inner HuggingFace ``PreTrainedModel`` extracted from the
        ``SentenceTransformer`` wrapper, used for HF-specific introspection
        (``.config``, ``.dtype``, FLOP counting, the ``inputs_embeds`` probe).

        Note that while the _model may have additional modules (e.g., dense pooling) we assume these are negilible and exclude them here.
        """
        try:
            inner = self._model._first_module().auto_model
        except AttributeError as e:
            raise ValueError(
                f"Could not extract the inner HuggingFace model from the "
                f"SentenceTransformer wrapper for `{self._model_name}`. The first "
                f"module is expected to be a `sentence_transformers.models.Transformer` "
            ) from e
        assert isinstance(inner, PreTrainedModel), (
            f"Expected the inner ST module to be a transformers.PreTrainedModel, "
            f"got {type(inner).__name__}."
        )
        return inner

    @property
    def d_model(self):
        return self._model.get_sentence_embedding_dimension()

    def _get_input_embeddings(self) -> torch.nn.Module:
        # this is a bit hacky way to extract the embedding layer from sentence transformers,
        # but as models may differ in implementation, we try multiple methods.

        # Each function below either extracts the embedding layer, or raises an exception.
        def _get_input_emb_v1():
            # Should work for most HF encoder models.
            return self._hf_model.get_input_embeddings()

        def _get_input_emb_v2():
            # Special case of NomicBertModel which lacks get_input_embeddings
            return self._hf_model.embeddings.word_embeddings

        for _get_input_emb in [_get_input_emb_v1, _get_input_emb_v2]:
            try:
                return _get_input_emb()
            except Exception:
                continue

        raise ValueError(
            f"Could not extract embedding layer from Sentence Transformer model `{self._model_name}`. This model might need special care. Please report this issue."
        )

    # ----------------------- set_inputs_from_tokens -----------------------

    def set_inputs_from_tokens(
        self,
        templates: TextTemplates,  # n_templates templates
        targets: Optional[Targets] = None,
    ) -> None:
        """Prepare and store the given templates in the inputs manager."""
        assert isinstance(templates, list), "templates must be a string or a list of strings."
        if targets is None:
            targets = Targets()

        # Build the input manager, that will allow combining with different triggers
        tok_ids = self._tokenizer(templates, add_special_tokens=True)["input_ids"]
        self._token_input_manager = HuggingFaceTokenInputManager(
            templates_ids=tok_ids,
            device=self.device,
            tokenizer=self._tokenizer,
            embed_func=self._embedding_layer,
            use_prefix_cache=False,  # prefix caching is not meant for encoder-only architectures
            targets=targets,
        )

    # ----------------------- invoke_from_tokens -----------------------

    def invoke_from_tokens(
        self,
        input_embeds: Float[Tensor, "bsz seq_len d_model"],
        input_attention_mask: Optional[Float[Tensor, "bsz seq_len"]] = None,
        count_backward: bool = False,
        **kwargs
    ) -> ModelOutput:
        """Perform a white-box forward pass through the model using input embeddings.

        Args:
            input_embeds: Input embeddings tensor (bsz, seq_len, d_model).
            input_attention_mask: Attention mask tensor (bsz, seq_len).
            count_backward: Whether this forward pass will be back-propagated through.

        Returns:
            ModelOutput: The output from the model.
        """

        assert input_embeds is not None, "input_embeds must be provided in invoke_from_tokens."
        if input_attention_mask is None:
            input_attention_mask = torch.ones(
                input_embeds.shape[:-1], device=input_embeds.device, dtype=torch.int64
            )

        outputs = self._model(
            dict(
                inputs_embeds=input_embeds,  # (bsz, seq_len, embd_dim)
                attention_mask=input_attention_mask,  # (bsz, seq_len)
            )
        )

        self._update_invoke_stats(
            n_tokens=int(input_attention_mask.sum().item()),
            n_samples=input_embeds.shape[0],
            count_backward=count_backward,
        )
        output_emb = outputs["sentence_embedding"]  # (bsz, d_model)

        return ModelOutput(
            output_embeddings=output_emb,
        )

    # ----------------------- invoke_from_texts -----------------------

    def invoke_from_texts(
        self,
        input_texts: Annotated[List[str], "n_texts"],
        **kwargs,
    ) -> ModelOutput:
        """
        Get the embeddings for the given texts (n_texts elements).
        Note: we mostly assume any prompting/instruction will be applied before the call to this function.
        """
        assert isinstance(input_texts, list)

        emb = self._model.encode(input_texts, convert_to_tensor=True, show_progress_bar=False)
        self._update_invoke_stats(
            n_tokens=sum(len(ids) for ids in self._tokenizer(input_texts)["input_ids"]),
            n_samples=len(input_texts),
        )

        return ModelOutput(
            output_embeddings=emb,
        )
