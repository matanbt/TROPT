
import logging
from typing import Annotated, List, Optional, Union

import torch
from jaxtyping import Float
from torch import Tensor
from transformers import AutoModel, AutoProcessor, AutoTokenizer

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

class CLIPEncoderHFModel(
    EncoderBaseModel,
    HuggingFaceBackendModel,
    # token-level access mixins:
    LossTokenAccessMixin,
    GradientTokenAccessMixin,
    GradientEmbedAccessMixin,
    # text-level access mixins:
    LossTextAccessMixin,
):
    """Wrapper for CLIP-like vision-language models (CLIP, SigLIP, etc.).

    Exposes the text encoder with full white-box access for trigger optimization,
    and provides `encode_images()` for encoding images into the shared embedding space.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        forward_pass_batch_size: int = 512,
        backward_pass_batch_size: int = 28,
        
        device: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        loaded_model=None,
        
        set_model_to_eval: bool = True,
        **kwargs,
    ):
        """
        Args:
            model_name: HuggingFace model name (e.g., "openai/clip-vit-large-patch14").
            device: Device to load the model onto.
            dtype: Data type for the model.
            forward_pass_batch_size: Batch size for forward passes.
            backward_pass_batch_size: Batch size for backward passes.
            loaded_model: Pre-loaded model instance.
            set_model_to_eval: Whether to set the model to evaluation mode.
        """
        if loaded_model is not None:
            self._model = loaded_model
        else:
            model_kwargs = {}
            if dtype is not None:
                model_kwargs["torch_dtype"] = dtype
            self._model = AutoModel.from_pretrained(
                model_name, 
                device_map=device or "auto",
                **model_kwargs, 
            )  # TODO we need the text model here, is this it

        # Load processor (handles both text tokenization and image preprocessing)
        self._processor = AutoProcessor.from_pretrained(
            model_name or self._model.config._name_or_path
        )
        # The tokenizer is the text-processing part of the processor
        self._tokenizer = self._processor.tokenizer
        #  if hasattr(self._processor, "tokenizer") else AutoTokenizer.from_pretrained(model_name or self._model.config._name_or_path)  # [<<-- TODO remve me]

        # Extract the text encoder's embedding layer
        self._embedding_layer = self._get_input_embeddings()

    @property
    def d_model(self) -> int:
        # CLIP-like models project to a shared embedding dimension
        return self._model.config.projection_dim  # TODO generalizes to SigLip?

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

    def invoke_from_tokens(
        self,
        input_embeds: Float[Tensor, "bsz seq_len d_text"],
        input_attention_mask: Optional[Float[Tensor, "bsz seq_len"]] = None,
        count_backward: bool = False,
        **kwargs,
    ) -> ModelOutput:
        """White-box forward pass through the text encoder using input embeddings.

        Returns projected text embeddings in the shared CLIP space (d_model).
        """
        if input_attention_mask is None:
            input_attention_mask = torch.ones(
                input_embeds.shape[:-1], device=input_embeds.device, dtype=torch.int64
            )

        # Forward pass through text model
        text_outputs = self._model.text_model(
            inputs_embeds=input_embeds,
            attention_mask=input_attention_mask,
        )

        # Pool and project to shared space (mirrors CLIPModel.get_text_features)
        # CLIP uses the [EOS] / last token's hidden state as the pooled output
        pooled_output = text_outputs[1]  # pooler_output
        text_embeds = self._model.text_projection(pooled_output)

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

        text_embeds = self._model.get_text_features(**inputs)

        self._update_invoke_stats(
            n_tokens=int(inputs["attention_mask"].sum().item()),
            n_samples=len(input_texts),
        )

        return ModelOutput(output_embeddings=text_embeds)

    # ----------------------- Image encoding -----------------------

    @torch.no_grad()
    def encode_images(
        self,
        images: Union[list, "PIL.Image.Image"],
    ) -> Float[Tensor, "n_images d_model"]:
        """Encode images into the shared CLIP embedding space.

        Args:
            images: A single PIL image or a list of PIL images.

        Returns:
            Image embeddings in the shared space, shape (n_images, d_model).
        """
        if not isinstance(images, list):
            images = [images]

        pixel_values = self._processor(
            images=images, return_tensors="pt"
        )["pixel_values"].to(self.device, self._model.dtype)

        image_embeds = self._model.get_image_features(pixel_values=pixel_values)
        return image_embeds
