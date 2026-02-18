from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Annotated

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizer

from tropt.common import (
    ModelInput,
    Targets,
    TokenTriggerCandidates,
)
from tropt.loss.base import BaseLoss
from tropt.loss.resolution import compute_loss_from_model_data

from .inputs_manager import (
    TextInputManager,
    TokenInputManager,
)
from .model_base import BaseTokenizer

# ====================== Model Mixins =======================

## -------- Token-level access mixins ------- ##
class TokenAccessMixin(ABC):
    """Mixin for models that can access token-level inputs."""

    _token_input_manager: Optional[TokenInputManager] = None

    @abstractmethod
    def set_token_inputs(
        self,
        templates: TextTemplates,
        targets: Targets = None,  # also n_templates, depends on the objective
    ) -> None:
        """Prepare and store the inputs manager as self._token_input_manager.

        Args:
            templates: List of text templates containing the trigger placeholder.
            targets: Optional targets for the loss function.
        """
        raise NotImplementedError

    def reset_token_inputs(self) -> None:
        """Clear self._token_input_manager."""
        self._token_input_manager = None

    @property
    def token_input_manager(self) -> TokenInputManager:
        """Returns the stored token input manager, raising if not initialized."""
        if self._token_input_manager is None:
            raise RuntimeError(
                f"{type(self).__name__}.token_input_manager accessed before set_token_inputs() was called."
            )
        return self._token_input_manager

    @text_input_manager.setter
    def text_input_manager(self, value: Optional[TextInputManager]) -> None:
        """Setter for the text input manager"""
        self._text_input_manager = value

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


## "Grey-box" Model Mixins:
class LossTokenAccessMixin(TokenAccessMixin):
    """Mixin for models that can compute losses based on token-level inputs."""

    @abstractmethod
    def compute_loss_from_tokens(
        self, candidate_trigger_ids: TokenTriggerCandidates, **kwargs
    ) -> Float[Tensor, "n_templates n_candidates"]:
        """Compute the loss on the stored token inputs with the given trigger merged in."""
        raise NotImplementedError


class LogitsTokenAccessMixin(TokenAccessMixin):
    """Mixin for models that can compute logits based on token-level inputs."""

    @abstractmethod
    def compute_logits_from_tokens(
        self, candidate_trigger_ids: TokenTriggerCandidates, **kwargs
    ) -> Float[Tensor, "trigger_seq_len vocab_size"]:
        """Compute logits w.r.t. `trigger` tokens that are merged into stored token inputs."""
        raise NotImplementedError


## "White-box" Model Mixins:
class GradientTokenAccessMixin(TokenAccessMixin):
    """Mixin for models that can compute gradients based on token-level inputs."""

    @abstractmethod
    def compute_grad_from_tokens(
        self, candidate_trigger_ids: TokenTriggerCandidates, **kwargs
    ) -> Float[Tensor, "trigger_seq_len vocab_size"]:
        """Compute gradients w.r.t. `trigger` tokens that are merged into stored token inputs."""
        raise NotImplementedError

## "White-box" Model Mixins w/ embed access:
class GradientEmbedAccessMixin(TokenAccessMixin):
    """Mixin for models that can compute gradients based on token-level inputs."""

    @abstractmethod
    def compute_grad_from_embeds(
        self,
        loss_func: BaseLoss,
        candidate_trigger_embeds: Float[Tensor, "n_candidates trigger_seq_len embed_dim"],
    ) -> Float[torch.Tensor, "n_candidates trigger_seq_len embed_dim"]:
        """Compute gradients w.r.t. `trigger` embeddings using stored token inputs."""
        raise NotImplementedError

## -------- Text-level access mixins ------- ##


## "Black-box" Model Mixins:
class TextAccessMixin(ABC):
    _text_input_manager: Optional[TextInputManager] = None

    @property
    def text_input_manager(self) -> TextInputManager:
        """Returns the stored text input manager, raising if not initialized."""
        if self._text_input_manager is None:
            raise RuntimeError(
                f"{type(self).__name__}.text_input_manager accessed before set_text_inputs() was called."
            )
        return self._text_input_manager

    @text_input_manager.setter
    def text_input_manager(self, value: Optional[TextInputManager]) -> None:
        """Setter for the text input manager, allowing it to be set to None to reset."""
        self._text_input_manager = value

    def set_text_inputs(
        self,
        templates: TextTemplates,
        targets: Targets = None,
    ) -> None:
        """Prepare and store the text-based inputs manager."""
        self.text_input_manager = TextInputManager(
            templates=templates,
            targets=targets,
        )

    def reset_text_inputs(self) -> None:
        """Clear the stored text input manager."""
        self.text_input_manager = None


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
        This computation is based on the __call__() method of the model, and the information it provides.
        """
        input_manager = self.text_input_manager
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
            model_output = self(
                curr_texts,
                return_full_output=True,
            )  # Returns ModelOutput with available data

            # Create ModelInput wrapper
            model_input = ModelInput(
                input_texts=curr_texts,
                targets=curr_targets,
            )

            # Use unified loss resolution
            loss = compute_loss_from_model_data(model_output, model_input, loss_func)  # shape: (n_candidates,)
            losses.append(loss)

        losses = torch.stack(losses, dim=0)  # shape: (n_templates, n_candidates)

        if not keep_message_dim:
            losses = losses.mean(dim=0)  # shape: (n_candidates,)

        return losses
