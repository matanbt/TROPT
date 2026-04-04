from transformers.models.siglip2.modular_siglip2 import Siglip2TextModel

import logging
from typing import Annotated, List, Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoTokenizer, CLIPTextModel, SiglipTextModel
from transformers.models.clip.modeling_clip import CLIPTextTransformer
from transformers.models.siglip.modeling_siglip import SiglipTextTransformer
from transformers.models.siglip2.modeling_siglip2 import Siglip2TextTransformer
from transformers.masking_utils import create_bidirectional_mask, create_causal_mask

from tropt.common import (
    OPTIMIZED_TRIGGER_PLACEHOLDER,
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

## TODO finish testing this module

class CLIPTextEncoderHFModel(
    EncoderBaseModel,
    HuggingFaceBackendModel,
    # token-level access mixins:
    LossTokenAccessMixin,
    GradientTokenAccessMixin,
    GradientEmbedAccessMixin,
    # text-level access mixins:
    LossTextAccessMixin,
):
    """Wrapper for Text encoders of CLIP and SigLIP models from HuggingFace.

    Implementation note: CLIP and SigLIP's text encoders do not take input embeddings, so we reimplement their forward pass logic to support it. This is a bit hacky, as it re-implement logic from Transformers's `modeling` files, but necessary to support grad-based optimization. 
    """

    def __init__(
        self,
        model_name: str,
        forward_pass_batch_size: int = 512,
        backward_pass_batch_size: int = 28,
        
        device: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        set_model_to_eval: bool = True,
        **kwargs,
    ):
        """
        Args:
            model_name: HuggingFace model name (e.g., "openai/clip-vit-large-patch14"). Currently only support CLIP and SigLIP models.
            device: Device to load the model onto.
            dtype: Data type for the model.
            forward_pass_batch_size: Batch size for forward passes.
            backward_pass_batch_size: Batch size for backward passes.
            set_model_to_eval: Whether to set the model to evaluation mode.
        """
        model_kwargs = {}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        
        if model_name.startswith("google/siglip-"):
            self._model = SiglipTextModel.from_pretrained(
                model_name, 
                device_map=device or "auto",
                **model_kwargs, 
            )
        elif model_name.startswith("google/siglip2-"):
            self._model = Siglip2TextModel.from_pretrained(
                model_name,
                device_map=device or "auto",
                **model_kwargs,
            )
        elif model_name.startswith("openai/clip-"):
            self._model = CLIPTextModel.from_pretrained(
                model_name,
                device_map=device or "auto",
                **model_kwargs,
            )
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Extract the text encoder's embedding layer
        self._embedding_layer = self._model.get_input_embeddings()

        # Optional projection: CLIP has text_projection; SigLIP outputs directly in shared space
        self._text_projection = getattr(self._model, "text_projection", None)

    @property
    def model_family(self) -> str:
        """Family of the loaded text encoder: ``"siglip"`` or ``"clip"``."""
        if isinstance(self._model, (SiglipTextModel, Siglip2TextModel)):
            return "siglip"
        if isinstance(self._model, CLIPTextModel):
            return "clip"
        raise ValueError(f"Unrecognised model type: {type(self._model)}")

    @property
    def d_model(self) -> int:
        if self.model_family == "clip":
            return self._model.out_features  # TODO fix me
        if self.model_family == "siglip":
            return self._model.config.hidden_size
        raise ValueError(f"Unrecognised model_family: {self.model_family}")

    # ----------------------- set_inputs_from_tokens -----------------------

    def set_inputs_from_tokens(
        self,
        templates: TextTemplates,
        targets: Optional[Targets] = None,
    ) -> None:
        """Prepare and store the given templates in the inputs manager."""
        assert isinstance(templates, list)

        tok_ids = self._tokenizer(templates, add_special_tokens=True)["input_ids"]
        self._token_input_manager = HuggingFaceTokenInputManager(
            templates_ids=tok_ids,
            device=self.device,
            tokenizer=self._tokenizer,
            embed_func=self._embedding_layer,
            use_prefix_cache=False,
            targets=targets,
        )

    # ----------------------- invoke_from_tokens -----------------------
    # Note: CLIP and SigLIP do not recieve input embedding, so we reimplement their forward pass here to support it

    def _encode_text_from_embeds_clip(
        self,
        input_embeds: Float[Tensor, "bsz seq_len d_text"],
        input_attention_mask: Float[Tensor, "bsz seq_len"],
    ) -> Float[Tensor, "bsz d_model"]:
        """
        CLIP text encoder forward from embeddings: causal attention, EOS pooling.

        Repeats the forward pass logic from SigLIP's text encoder with input_embeds.
        https://github.com/huggingface/transformers/blob/e1b80de84d3c5da35669b2834ef017eeaf620f93/src/transformers/models/clip/modeling_clip.py#L531-L589
        """
        text_model = self._model.text_model  # Todo rellay .text?
        assert isinstance(text_model, CLIPTextTransformer), f"Expected CLIPTextTransformer, got {type(text_model)}"
        hidden_states = text_model.embeddings(inputs_embeds=input_embeds)
        mask = create_causal_mask(
            config=text_model.config,
            inputs_embeds=hidden_states,
            attention_mask=input_attention_mask,
            past_key_values=None,
        )
        last_hidden_state = text_model.final_layer_norm(
            text_model.encoder(inputs_embeds=hidden_states, attention_mask=mask, is_causal=True).last_hidden_state
        )
        # EOS token is always the last attended position in CLIP sequences
        eos_pos = input_attention_mask.sum(dim=-1) - 1
        bsz = last_hidden_state.shape[0]
        return last_hidden_state[torch.arange(bsz, device=last_hidden_state.device), eos_pos]

    def _encode_text_from_embeds_siglip(
        self,
        input_embeds: Float[Tensor, "bsz seq_len d_text"],
        input_attention_mask: Float[Tensor, "bsz seq_len"],
    ) -> Float[Tensor, "bsz d_model"]:
        """
        SigLIP text encoder forward from input embeddings.

        Repeats the forward pass logic from SigLIP's text encoder with input_embeds.
        https://github.com/huggingface/transformers/blob/e1b80de84d3c5da35669b2834ef017eeaf620f93/src/transformers/models/siglip/modeling_siglip.py#L4890-L527
        https://github.com/huggingface/transformers/blob/e1b80de84d3c5da35669b2834ef017eeaf620f93/src/transformers/models/siglip2/modeling_siglip2.py#L571-L612
        """
        text_model = self._model.text_model  # TODO really .text_model ??
        assert isinstance(text_model, (SiglipTextTransformer, Siglip2TextTransformer)), f"Expected SiglipTextTransformer, got {type(text_model)}"
        hidden_states = text_model.embeddings(inputs_embeds=input_embeds)
        mask = create_bidirectional_mask(
            config=text_model.config,
            inputs_embeds=hidden_states,
            attention_mask=input_attention_mask,
        )
        last_hidden_state = text_model.final_layer_norm(
            text_model.encoder(inputs_embeds=hidden_states, attention_mask=mask).last_hidden_state
        )
        # TODO assert all close to SigLIP in tests
        return text_model.head(last_hidden_state[:, -1, :])

    def invoke_from_tokens(
        self,
        input_embeds: Float[Tensor, "bsz seq_len d_text"],
        input_attention_mask: Optional[Float[Tensor, "bsz seq_len"]] = None,
        count_backward: bool = False,
        **kwargs,
    ) -> ModelOutput:
        """White-box forward pass through the text encoder using input embeddings.

        Returns projected text embeddings in the shared CLIP/SigLIP space (d_model).
        """
        if input_attention_mask is None:
            input_attention_mask = torch.ones(
                input_embeds.shape[:-1], device=input_embeds.device, dtype=torch.int64
            )

        if self.model_family == "siglip":
            pooled_output = self._encode_text_from_embeds_siglip(input_embeds, input_attention_mask)
        else:  # "clip"
            pooled_output = self._encode_text_from_embeds_clip(input_embeds, input_attention_mask)

        text_embeds = self._text_projection(pooled_output) if self._text_projection is not None else pooled_output

        self._update_invoke_stats(
            n_tokens=int(input_attention_mask.sum().item()),
            n_samples=input_embeds.shape[0],
            count_backward=count_backward,
        )

        return ModelOutput(output_embeddings=text_embeds)

    # ----------------------- invoke_from_texts -----------------------

    @torch.no_grad()
    def invoke_from_texts(
        self,
        input_texts: Annotated[List[str], "n_texts"],
        **kwargs,
    ) -> ModelOutput:
        """Encode texts into the shared CLIP embedding space."""
        assert isinstance(input_texts, list)

        inputs = self._tokenizer(
            input_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        # Use text_model + text_projection directly (mirrors invoke_from_tokens),
        # since get_text_features() return type varies across model variants.
        text_outputs = self._model(**inputs)
        pooled_output = text_outputs.pooler_output
        text_embeds = self._text_projection(pooled_output) if self._text_projection is not None else pooled_output

        attn_mask = inputs.get("attention_mask")
        self._update_invoke_stats(
            n_tokens=int(attn_mask.sum().item()) if attn_mask is not None else inputs["input_ids"].numel(),
            n_samples=len(input_texts),
        )

        return ModelOutput(output_embeddings=text_embeds)
