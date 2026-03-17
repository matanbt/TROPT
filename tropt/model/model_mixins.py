from abc import ABC, abstractmethod
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizer

from tropt.common import (
    ModelInput,
    ModelOutput,
    Targets,
    TextTemplates,
    TokenTriggerCandidates,
)
from tropt.loss import BaseLoss
from tropt.loss.resolution import resolve_and_compute_loss

from .inputs_manager import (
    TextInputManager,
    TokenInputManager,
)
from .model_base import BaseTokenizer

# ======================================================================
# Token Access Flow
# ======================================================================

class TokenAccessMixin(ABC):
    """Mixin for models that have a tokenizer and can prepare token-level inputs.

    This is the base mixin for any model with token-level access (tokenizer,
    set/reset inputs). Note that such models may note have access to _compute_ the loss from tokens (see LossTokenAccessMixin), but they must be able to at least prepare the token inputs (e.g., OpenAI models).
    """

    _token_input_manager: Optional[TokenInputManager] = None

    @abstractmethod
    def set_inputs_from_tokens(
        self,
        templates: TextTemplates,
        targets: Targets = None,
    ) -> None:
        """Prepare and store the inputs manager as self._token_input_manager.

        Args:
            templates: List of text templates containing the trigger placeholder.
            targets: Optional targets for the loss function.
        """
        raise NotImplementedError

    def reset_inputs_from_tokens(self) -> None:
        """Clear self._token_input_manager."""
        self._token_input_manager = None

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizer | BaseTokenizer:
        """
        Force the class using this mixin to implement a tokenizer.
        This tokenizer must be either HuggingFace tokenizer or one with the same interface.
        """
        raise NotImplementedError


class InvokeTokenAccessMixin(TokenAccessMixin):
    """Mixin for models that can perform a forward pass from token-level inputs.

    Adds the abstract invoke_from_tokens method. All compute-* mixins
    (LossTokenAccessMixin, GradientTokenAccessMixin, etc.) inherit from this.
    """
    # TODO make it only input-ids here -- the rest is for implementer interpretation!
    @abstractmethod
    def invoke_from_tokens(
        self,
        input_embeds: Float[Tensor, "bsz seq_len d_model"] = None,
        input_attention_mask: Int[Tensor, "bsz seq_len"] = None,
        input_prefix_cache_kwargs: Optional[Dict[str, Any]] = None,
        input_slices: Optional[Dict] = None,
    ) -> ModelOutput:
        """Perform a forward pass from token-level (embedding) inputs.

        Args:
            input_embeds: Input embeddings with trigger inserted.
            input_attention_mask: Attention mask for the input.
            input_prefix_cache_kwargs: Optional KV cache kwargs (HF models).
            input_slices: Position slices for different input regions.
            reference_loss_func: Optional loss function to conditionally disable/enable
                expensive outputs (e.g., attentions, hidden states).

        Returns:
            ModelOutput with the fields this model can provide.
        """
        raise NotImplementedError


## "Grey-box" Model Mixins:
class LossTokenAccessMixin(InvokeTokenAccessMixin):
    """Mixin for models that can compute losses based on token-level inputs."""

    @abstractmethod
    def compute_loss_from_tokens(
        self, candidate_trigger_ids: TokenTriggerCandidates, **kwargs
    ) -> Float[Tensor, "n_templates n_candidates"]:
        """Compute the loss on the stored token inputs with the given trigger merged in."""
        raise NotImplementedError


class LogitsTokenAccessMixin(InvokeTokenAccessMixin):
    """Mixin for models that can compute logits based on token-level inputs."""

    @abstractmethod
    def compute_logits_from_tokens(
        self, candidate_trigger_ids: TokenTriggerCandidates, **kwargs
    ) -> Float[Tensor, "trigger_seq_len vocab_size"]:
        """Compute logits w.r.t. `trigger` tokens that are merged into stored token inputs."""
        raise NotImplementedError


## "White-box" Model Mixins:
class GradientTokenAccessMixin(InvokeTokenAccessMixin):
    """Mixin for models that can compute gradients based on token-level inputs."""

    @abstractmethod
    def compute_grad_from_tokens(
        self, candidate_trigger_ids: TokenTriggerCandidates, **kwargs
    ) -> Float[Tensor, "trigger_seq_len vocab_size"]:
        """Compute gradients w.r.t. `trigger` tokens that are merged into stored token inputs."""
        raise NotImplementedError

## "White-box" Model Mixins w/ embed access:
class GradientEmbedAccessMixin(InvokeTokenAccessMixin):
    """Mixin for models that can compute gradients based on token-level inputs."""

    @abstractmethod
    def compute_grad_from_embeds(
        self,
        loss_func: BaseLoss,
        candidate_trigger_embeds: Float[Tensor, "n_candidates trigger_seq_len embed_dim"],
    ) -> Float[torch.Tensor, "n_candidates trigger_seq_len embed_dim"]:
        """Compute gradients w.r.t. `trigger` embeddings using stored token inputs."""
        raise NotImplementedError


# ======================================================================
# Text Access Flow
# ======================================================================

class TextAccessMixin(ABC):
    _text_input_manager: Optional[TextInputManager] = None

    def set_inputs_from_texts(
        self,
        templates: TextTemplates,
        targets: Targets = None,
    ) -> None:
        """Prepare and store the text-based inputs manager."""
        self._text_input_manager = TextInputManager(
            templates=templates,
            targets=targets,
        )

    def reset_inputs_from_texts(self) -> None:
        """Clear the stored text input manager."""
        self._text_input_manager = None


class LossTextAccessMixin(TextAccessMixin):
    """Mixin for models that compute losses based on text-level inputs (black-box access)."""

    @torch.no_grad()  # in case using torch model
    def compute_loss_from_texts(
        self,
        candidate_trigger_strs: List[str],
        loss_func: BaseLoss,
        keep_message_dim: bool = False,
    ) -> Float[Tensor, "n_candidates"]:
        """
        Computes the loss on all candidate string texts using the stored text inputs manager.
        This computation is based on the invoke_from_texts() method of the model, and the information it provides.
        """
        assert self._text_input_manager is not None, "Text input manager is not initialized. Please call set_inputs_from_texts() first."

        input_manager = self._text_input_manager
        n_templates = input_manager.n_templates
        n_candidates = len(candidate_trigger_strs)

        # Main Loop: for each template, we compute the loss for all candidates
        losses = []
        for template_idx in range(n_templates):
            curr_model_input = input_manager.get_triggered_inputs(
                candidate_trigger_strs, chosen_template_idx=template_idx
            )
            curr_texts, curr_targets = (
                curr_model_input.input_texts,
                curr_model_input.targets,
            )

            # Forward pass once per template bulk
            model_output = self.invoke_from_texts(
                input_texts=curr_texts,
            )  # Returns ModelOutput with available data

            # Create ModelInput wrapper
            model_input = ModelInput(
                input_texts=curr_texts,
                targets=curr_targets,
            )

            # Use unified loss resolution
            loss = resolve_and_compute_loss(model_output, model_input, loss_func)  # shape: (n_candidates,)
            losses.append(loss)

        losses = torch.stack(losses, dim=0)  # shape: (n_templates, n_candidates)

        if not keep_message_dim:
            losses = losses.mean(dim=0)  # shape: (n_candidates,)

        return losses
