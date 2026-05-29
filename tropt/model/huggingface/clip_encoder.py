import logging
from typing import Annotated, List, Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from transformers.masking_utils import create_causal_mask
from transformers.models.clip.modeling_clip import CLIPTextModel

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

class CLIPTextEncoderHFModel(
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
    """Wrapper for the text encoder of OpenAI CLIP models from HuggingFace.

    Implementation note:
        CLIP's text encoder does not accept input embeddings, so we reimplement its forward
        pass to support them. This is hacky, as we repeat logic from Transformers's CLIP's Modeling file, but necessary for supporting grad-based / soft-token-based optimization.
        Other discrete-optimizer implementations have turned to similar solutions, e.g., PEZ (Wen et al. 2023), that forked OpenCLIP (https://github.com/YuxinWenRick/hard-prompts-made-easy/blob/f22a1bec01991d94697304443cacbd66e0167e6b/open_clip/model.py#L230).
    """

    def __init__(
        self,
        model_name: str,
        forward_pass_batch_size: int = 512,
        backward_pass_batch_size: int = 28,
        without_final_projection: bool = False,
        device: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        set_model_to_eval: bool = True,
        **kwargs,
    ):
        """
        Args:
            model_name: HuggingFace CLIP model name (e.g., "openai/clip-vit-large-patch14").
            without_final_projection: If True, skip the final ``text_projection`` layer and the default pooling.
            This is relevant when CLIP is targeted as a backbone encoder for a downstream model; for instance, FLUX[dev] uses the non-projected CLIP text encoding -- so in this case, it is recommended to set `without_final_projection=True` to match the downstream architecture.
            We default to include project (=False), following CLIP's default in HF.
            device: Device to load the model onto.
            dtype: Data type for the model.
            forward_pass_batch_size: Batch size for forward passes.
            backward_pass_batch_size: Batch size for backward passes.
            set_model_to_eval: Whether to set the model to evaluation mode.
        """
        model_kwargs = {}
        if dtype is not None:
            model_kwargs["dtype"] = dtype

        # CLIPTextModelWithProjection includes text_projection, needed to match get_text_features()
        self._model = CLIPTextModelWithProjection.from_pretrained(
            model_name,
            device_map=device or "auto",
            **model_kwargs,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._embedding_layer = self._model.get_input_embeddings()
        self._without_final_projection = without_final_projection
        # We reimplement the forward pass to accept inputs_embeds (see invoke_from_tokens),
        # so the base-class post-init check for inputs_embeds support can be skipped.
        self._handles_input_embeds_manually = True

    @property
    def d_model(self) -> int:
        if self._without_final_projection:
            return int(self._model.config.hidden_size)
        return int(self._model.config.projection_dim)

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
    # CLIP's text encoder does not accept inputs_embeds, so we reimplement its forward pass here.

    def _encode_text_from_embeds(
        self,
        input_embeds: Float[Tensor, "bsz seq_len d_text"],
        input_attention_mask: Float[Tensor, "bsz seq_len"],
    ) -> Float[Tensor, "bsz d_model"]:
        """CLIP text encoder forward from input embeddings: causal attention, EOS pooling.

        Reimplements the forward pass from:
        https://github.com/huggingface/transformers/blob/e1b80de84d3c5da35669b2834ef017eeaf620f93/src/transformers/models/clip/modeling_clip.py#L531-L589
        """
        text_model = self._model.text_model
        assert isinstance(text_model, CLIPTextModel), f"Expected CLIPTextModel, got {type(text_model)}"

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
        pooled = last_hidden_state[torch.arange(bsz, device=last_hidden_state.device), eos_pos]
        if self._without_final_projection:
            return pooled
        return self._model.text_projection(pooled)

    def invoke_from_tokens(
        self,
        input_embeds: Float[Tensor, "bsz seq_len d_text"],
        input_attention_mask: Optional[Float[Tensor, "bsz seq_len"]] = None,
        count_backward: bool = False,
        **kwargs,
    ) -> ModelOutput:
        """White-box forward pass through the text encoder using input embeddings."""
        if input_attention_mask is None:
            input_attention_mask = torch.ones(
                input_embeds.shape[:-1], device=input_embeds.device, dtype=torch.int64
            )

        text_embeds = self._encode_text_from_embeds(input_embeds, input_attention_mask)

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
        """Encode texts into the CLIP embedding space."""
        assert isinstance(input_texts, list)

        inputs = self._tokenizer(
            input_texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        text_outputs = self._model(**inputs)
        attn_mask = inputs.get("attention_mask")

        if self._without_final_projection:
            # To avoid projection, repeat pooling logic (as HF doesn't provide it)
            eos_pos = attn_mask.sum(dim=-1) - 1
            bsz = text_outputs.last_hidden_state.shape[0]
            text_embeds = text_outputs.last_hidden_state[
                torch.arange(bsz, device=text_outputs.last_hidden_state.device), eos_pos
            ]
        else:
            text_embeds = text_outputs.text_embeds

        self._update_invoke_stats(
            n_tokens=int(attn_mask.sum().item()) if attn_mask is not None else inputs["input_ids"].numel(),
            n_samples=len(input_texts),
        )

        return ModelOutput(output_embeddings=text_embeds)
