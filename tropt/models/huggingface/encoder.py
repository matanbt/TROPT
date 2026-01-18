import logging
from typing import List, Optional, Tuple

import sentence_transformers
import torch
from jaxtyping import Float, Int
from sentence_transformers import SentenceTransformer
from torch import Tensor

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER, DEFAULT_INIT_TRIGGER
from tropt.loss.base import BaseLoss, CombinedLoss, EmbeddingBasedLoss
from tropt.models import (
    EncoderBaseModel,
    GradientTokenAccessMixin,
    LossTextAccessMixin,
    LossTokenAccessMixin,
    MessageBatchedTargetsDict,
    TargetsDict,
    TargetsDictPlus,
)
from tropt.models.huggingface.base import HFTokenInputsManager, HuggingFaceModelMixins

logger = logging.getLogger(__name__)
# ======================= Input/Output Handlers logic =======================


class EncoderHFTokenInputsManager(HFTokenInputsManager):
    targets: TargetsDict | TargetsDictPlus
    # includes `target_vectors` (n_messages, d_model) if target outputs are provided; 
    # to optimize towards an vector per message


# ======================= Model logic =======================


class EncoderHFModel(
    EncoderBaseModel,
    # adds implementation of common HF model methods:
    HuggingFaceModelMixins,
    # token-level access mixins:
    LossTokenAccessMixin,
    GradientTokenAccessMixin,
    # text-level access mixins:
    LossTextAccessMixin,
):
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        dtype: str| torch.dtype = None,
        forward_pass_batch_size: int = 512,
        backward_pass_batch_size: int = 28,
        loaded_model: Optional[SentenceTransformer] = None,
        trust_remote_code: bool = False,
        set_model_to_eval: bool = True,
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
            trust_remote_code (bool): Whether to trust remote code when loading the model.
            set_model_to_eval (bool): Whether to set the model to evaluation mode.
            **kwargs: Additional arguments for SentenceTransformer.
        """
        self.model_name = model_name
        self.forward_pass_batch_size = forward_pass_batch_size
        self.backward_pass_batch_size = backward_pass_batch_size

        if loaded_model is not None:
            assert isinstance(loaded_model, SentenceTransformer), "loaded_model must be a SentenceTransformer instance."
            self.model = loaded_model
        else:
            self.model = SentenceTransformer(
                model_name,
                device=device,
                model_kwargs=dict(dtype=dtype or "auto"),
                trust_remote_code=trust_remote_code,
                **kwargs
            )
        self.d_model = self.model.get_sentence_embedding_dimension()
        self._tokenizer = self.model.tokenizer
        self.embedding_layer = self._get_input_embeddings()

        # Set model to eval mode
        if set_model_to_eval:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        # To make sure the placeholder will be tokenizer as is
        self._tokenizer.add_special_tokens(
            {"additional_special_tokens": [OPTIMIZED_TRIGGER_PLACEHOLDER]}
        )

        ## warning and checks:
        if self.model.dtype in (torch.float32, torch.float64):
            logger.warning(
                f"Model is in {self.model.dtype}. Use a lower precision data type, if possible, for much faster optimization."
            )
        logger.warning("[General Warning:] Common embedding models often require an instruction prefix (e.g., `query: `). For optimal performance, please make sure a suitable one is applied in the textual input templates.")

    @property
    def tokenizer(self):
        return self._tokenizer
    
    @property
    def device(self):
        return self.model.device

    def _get_input_embeddings(self):
        # this is a bit hacky way to extract the embedding layer from sentence transformers,
        # but as models may differ in implementation, we try multiple methods.

        # Each function below either extracts the embedding layer, or raises an exception.
        def _get_input_emb_v1():
            # Should work for most ST models
            transformer_module = self.model._first_module()
            assert isinstance(
                transformer_module, sentence_transformers.models.Transformer
            )
            input_embeddings = transformer_module.auto_model.get_input_embeddings()
            return input_embeddings

        def _get_input_emb_v2():
            # Special case of NomicBertModel which lacks get_input_embeddings
            transformer_module = self.model._first_module()
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
            f"Could not extract embedding layer from Sentence Transformer model `{self.model_name}`. This model might need special care. Please report this issue."
        )

    def prepare_token_inputs(
        self,
        texts: List[str],  # n_messages texts
        targets: TargetsDict | TargetsDictPlus,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
    ) -> Tuple[EncoderHFTokenInputsManager, Int[Tensor, "1 trigger_seq_len"]]:

        assert isinstance(texts, list), "texts must be a string or a list of strings."
        n_messages = len(texts)
        targets = TargetsDictPlus(targets, n_messages=n_messages)

        # Build the input manager, that will allow combining with different triggers
        tok_ids = self.tokenizer(texts, add_special_tokens=True)["input_ids"]
        inputs = EncoderHFTokenInputsManager(
            tok_ids=tok_ids,
            model=self.model,
            tokenizer=self.tokenizer,
            embed_func=self.embedding_layer,
            optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER,
            use_prefix_cache=False,  # prefix caching is not meant for encoder-only architectures
            targets=targets,  # n_messages, d_model
        )

        # Tokenizer trigger
        if not initial_trigger:
            # start with an empty trigger
            trigger_tok_ids = torch.zeros((1, 0), dtype=torch.long, device=self.model.device)
        else:
            trigger_tok_ids = (
                self.tokenizer(
                    initial_trigger, add_special_tokens=False, return_tensors="pt"
                )["input_ids"]
                .to(self.model.device, torch.int64)
            )

        return inputs, trigger_tok_ids

    def _loss_hook(
        self,
        inputs_embeds: Float[Tensor, "bsz seq_len embd_dim"],
        attention_mask: Float[Tensor, "bsz seq_len"],
        targets: MessageBatchedTargetsDict,
        loss_func: BaseLoss,
        **kwargs,
    ) -> Float[Tensor, "bsz"]:

        # Forward pass
        outputs = self.model(
            dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        )
        output_emb = outputs["sentence_embedding"]  # (bsz, d_model)

        # Recursive loss calculation helper
        def _calc_loss_from_outputs(_output_emb, _targets, _loss_func):
            if isinstance(_loss_func, EmbeddingBasedLoss):
                # Base case: Compute embedding loss
                return _loss_func(
                    _output_emb,
                    _targets["target_vectors"]
                    )  # shape: (bsz,)

            elif isinstance(_loss_func, CombinedLoss):
                # Recursive case: Compute all child losses
                losses = []
                for _nested_loss_func in _loss_func.loss_funcs:
                    losses.append(
                        _calc_loss_from_outputs(_output_emb, _targets, _nested_loss_func)
                    )

                # Stack results: (n_losses, bsz)
                losses = torch.stack(losses, dim=0)

                # Apply combination logic (e.g. weighted sum) -> (bsz,)
                return _loss_func(losses)

            else:
                raise NotImplementedError(
                    f"Loss function {_loss_func} not supported for HuggingFace models yet."
                )

        return _calc_loss_from_outputs(output_emb, targets, loss_func)
    @torch.no_grad()
    def __call__(self, texts: List[str]) -> Float[Tensor, "n_texts d_model"]:
        """
        Get the embeddings for the given texts (n_texts elements).
        Note: we mostly assume any prompting/instruction will be applied before the call to this function.
        """
        assert isinstance(texts, list)

        emb = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

        return emb
