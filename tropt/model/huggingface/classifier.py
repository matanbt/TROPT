import logging
from typing import Dict, List, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tropt.common import (
    OPTIMIZED_TRIGGER_PLACEHOLDER,
    ModelOutput,
    Targets,
    TextTemplates,
)
from tropt.model import (
    ClassifierBaseModel,
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


class ClassifierHFModel(
    # HF backend first so its `device`/`dtype` win MRO over `BaseModel`'s defaults:
    HuggingFaceBackendModel,
    ClassifierBaseModel,
    # token-level access mixins:
    LossTokenAccessMixin,
    GradientTokenAccessMixin,
    GradientEmbedAccessMixin,
    # text-level access mixins:
    LossTextAccessMixin,
):
    """HuggingFace sequence classification model wrapper (for models loadable with `AutoModelForSequenceClassification`)."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str | torch.dtype] = None,
        forward_pass_batch_size: int = 512,
        backward_pass_batch_size: int = 28,
        loaded_model=None,
        set_model_to_eval: bool = True,
        **kwargs,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if loaded_model is not None:
            self._model = loaded_model
        else:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                dtype=dtype or "auto",
                **kwargs,
            ).to(device)

        # Set tokenizer and embedding layer:
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._embedding_layer = self._model.get_input_embeddings()

    @property
    def n_classes(self) -> int:
        return self._model.config.num_labels

    @property
    def id2label(self) -> Dict[int, str]:
        assert self._model.config.id2label is not None
        for k, v in self._model.config.id2label.items():
            assert isinstance(k, int) and isinstance(v, str)
        return self._model.config.id2label  # type: ignore[return-value] (verified by assertions)

    # ----------------------- set_inputs_from_tokens -----------------------

    def set_inputs_from_tokens(
        self,
        templates: TextTemplates,
        targets: Optional[Targets] = None,
    ) -> None:
        assert isinstance(templates, list)
        if targets is None:
            targets = Targets()

        tok_ids = self._tokenizer(templates, add_special_tokens=True)["input_ids"]
        self._token_input_manager = HuggingFaceTokenInputManager(
            tokenizer=self._tokenizer,
            device=self.device,
            templates_ids=tok_ids,
            embed_func=self._embedding_layer,
            use_prefix_cache=False,
            targets=targets,
        )

    # ----------------------- invoke_from_tokens -----------------------

    def invoke_from_tokens(
        self,
        input_embeds: Float[Tensor, "bsz seq_len d_model"],
        input_attention_mask: Optional[Int[Tensor, "bsz seq_len"]] = None,
        count_backward: bool = False,
        **kwargs,
    ) -> ModelOutput:
        assert input_embeds is not None
        if input_attention_mask is None:
            input_attention_mask = torch.ones(
                input_embeds.shape[:2], device=input_embeds.device, dtype=torch.int64
            )

        outputs = self._model(
            inputs_embeds=input_embeds,
            attention_mask=input_attention_mask,
        )

        self._update_invoke_stats(
            n_tokens=int(input_attention_mask.sum().item()),
            n_samples=input_embeds.shape[0],
            count_backward=count_backward,
        )

        return ModelOutput(
            output_class_logits=outputs.logits,  # (bsz, n_classes)
        )

    # ----------------------- invoke_from_texts -----------------------

    @torch.no_grad()
    def invoke_from_texts(
        self,
        input_texts: List[str],
        **kwargs,
    ) -> ModelOutput:
        assert isinstance(input_texts, list) and all(isinstance(t, str) for t in input_texts)

        inputs = self._tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        outputs = self._model(**inputs)
        self._update_invoke_stats(
            n_tokens=int(inputs["attention_mask"].sum().item()),
            n_samples=len(input_texts),
        )

        return ModelOutput(
            output_class_logits=outputs.logits,  # (bsz, n_classes)
        )
