
import logging
from typing import Annotated, List, Optional

import sentence_transformers
import sentence_transformers.models
import torch
from jaxtyping import Float, Int
from sentence_transformers import SentenceTransformer
from torch import Tensor

from tropt.common import (
    OPTIMIZED_TRIGGER_PLACEHOLDER,
    ModelInput,
    ModelOutput,
    Targets,
    TextTemplates,
)
from tropt.loss import BaseLoss
from tropt.model import (
    EncoderBaseModel,
    GradientTokenAccessMixin,
    LossTextAccessMixin,
    LossTokenAccessMixin,
)
from tropt.model.huggingface.base import _HFTokenInputManager, _HuggingFaceModelMixins
from tropt.model.model_mixins import GradientEmbedAccessMixin

logger = logging.getLogger(__name__)
# ======================= Input/Output Handlers logic =======================


class EncoderHFTokenInputManager(_HFTokenInputManager):
    targets: Targets
    # includes `target_vectors` (n_templates, d_model) if target outputs are provided;
    # to optimize towards an vector per message


# ======================= Model logic =======================


class EncoderHFModel(
    EncoderBaseModel,
    # adds implementation of common HF model methods:
    _HuggingFaceModelMixins,
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
        set_model_to_eval: bool = True,
        run_additional_checks: bool = True,
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
            set_model_to_eval (bool): Whether to set the model to evaluation mode.
            run_additional_checks (bool): Whether to run additional checks on the model to ensure compliance with this module computations. Can be disabed for faster initialization, but is good for identfying incompatability issues (mostly with models overriding HF default code).
            **kwargs: Additional arguments for SentenceTransformer.
        """
        self._model_name = model_name
        self._forward_pass_batch_size = forward_pass_batch_size
        self._backward_pass_batch_size = backward_pass_batch_size

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
        self._tokenizer = self._model.tokenizer
        self._embedding_layer = self._get_input_embeddings()

        # Set model to eval mode
        if set_model_to_eval:
            self._model.eval()
            for param in self._model.parameters():
                param.requires_grad = False

        # To make sure the placeholder will be tokenizer as is
        if OPTIMIZED_TRIGGER_PLACEHOLDER not in self._tokenizer.get_vocab():
            self._tokenizer.add_special_tokens(
                {"additional_special_tokens": [OPTIMIZED_TRIGGER_PLACEHOLDER]}
            )

        ## warning and checks:
        if self._model.dtype in (torch.float32, torch.float64):
            logger.warning(
                f"Model is in {self._model.dtype}. Use a lower precision data type, if possible, for much faster optimization."
            )
        if run_additional_checks:
            if self._requires_input_ids_with_embeds():
                logger.warning(
                    f"Model `{self._model_name}` requires `input_ids` even when `inputs_embeds` are provided. "
                    "Gradient-based optimization (GradientTokenAccessMixin / invoke_from_tokens) will not work with this model. "
                    "Only text-level access (invoke_from_texts) is supported."
                )

        logger.warning("[General Warning:] Common embedding models often require an instruction prefix (e.g., `query: `). For optimal performance, please make sure a suitable one is applied in the textual input templates.")

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self):
        return self._model.device

    @property
    def d_model(self):
        return self._model.get_sentence_embedding_dimension()

    @property
    def embedding_layer(self) -> torch.nn.Module:
        return self._embedding_layer

    def _get_input_embeddings(self):
        # this is a bit hacky way to extract the embedding layer from sentence transformers,
        # but as models may differ in implementation, we try multiple methods.

        # Each function below either extracts the embedding layer, or raises an exception.
        def _get_input_emb_v1():
            # Should work for most ST models
            transformer_module = self._model._first_module()
            assert isinstance(
                transformer_module, sentence_transformers.models.Transformer
            )
            input_embeddings = transformer_module.auto_model.get_input_embeddings()
            return input_embeddings

        def _get_input_emb_v2():
            # Special case of NomicBertModel which lacks get_input_embeddings
            transformer_module = self._model._first_module()
            input_embeddings = transformer_module.auto_model.embeddings.word_embeddings
            return input_embeddings

        # Try each method until one works
        for _get_input_emb in [_get_input_emb_v1, _get_input_emb_v2]:
            try:
                input_embeddings = _get_input_emb()
                return input_embeddings
            except Exception:
                continue

        # If none of the methods worked, raise an error
        raise ValueError(
            f"Could not extract embedding layer from Sentence Transformer model `{self._model_name}`. This model might need special care. Please report this issue."
        )

    @torch.no_grad()
    def _requires_input_ids_with_embeds(self) -> bool:
        """Returns True if the model requires input_ids even when inputs_embeds are provided."""
        dummy_len = 4
        dummy_embeds = torch.zeros(
            1, dummy_len, self.d_model,
            device=self.device, dtype=self._model.dtype,
        )
        dummy_mask = torch.ones(1, dummy_len, device=self.device, dtype=torch.int64)
        try:
            self._model(dict(inputs_embeds=dummy_embeds, attention_mask=dummy_mask))
            return False
        except TypeError:
            return True

    # ----------------------- set_inputs_from_tokens -----------------------

    def set_inputs_from_tokens(
        self,
        templates: TextTemplates,  # n_templates templates
        targets: Optional[Targets] = None,
    ) -> None:
        """Prepare and store the given templates in the inputs manager."""
        assert isinstance(templates, list), "templates must be a string or a list of strings."

        # Build the input manager, that will allow combining with different triggers
        tok_ids = self._tokenizer(templates, add_special_tokens=True)["input_ids"]
        self._token_input_manager = EncoderHFTokenInputManager(
            tok_ids=tok_ids,
            model=self._model,
            tokenizer=self._tokenizer,
            embed_func=self._embedding_layer,
            optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER,
            use_prefix_cache=False,  # prefix caching is not meant for encoder-only architectures
            targets=targets,
        )

    # ----------------------- invoke_from_tokens -----------------------

    def invoke_from_tokens(
        self,
        input_embeds: Float[Tensor, "bsz seq_len d_model"],
        input_attention_mask: Optional[Float[Tensor, "bsz seq_len"]] = None,
        **kwargs
    ) -> ModelOutput:
        """
        Perform a white-box forward pass through the model using input embeddings.

        Args:
            input_embeds: Input embeddings tensor (bsz, seq_len, d_model).
            input_attention_mask: Attention mask tensor (bsz, seq_len).
            reference_loss_func: Optional reference loss function.

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
        self._update_usage_stats(
            forward_calls=1,
            forward_samples=len(input_embeds),
            tokens=int(input_attention_mask.sum().item()),
        )
        output_emb = outputs["sentence_embedding"]  # (bsz, d_model)

        return ModelOutput(
            output_embeddings=output_emb,
        )

    # ----------------------- invoke_from_texts -----------------------

    @torch.no_grad()
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
        self._update_usage_stats(
            forward_calls=1,
            forward_samples=len(input_texts),
            tokens=sum(len(ids) for ids in self._tokenizer(input_texts)["input_ids"]),
        )

        return ModelOutput(
            output_embeddings=emb,
        )
