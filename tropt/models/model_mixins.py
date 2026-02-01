from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizer

from tropt.common import DEFAULT_INIT_TRIGGER
from tropt.loss.base import BaseLoss, CombinedLoss, EmbeddingBasedLoss, LogitBasedLoss
from tropt.loss.resolution import compute_loss_from_model_data
from tropt.loss.text_loss import InputReadabilityLoss

from .inputs import (
    BatchedTargetsDict,
    MessageBatchedTargetsDict,
    ModelInput,
    TargetsDict,
    TargetsDictPlus,
    TextInputsManager,
    TokenInputsManager,
    TokenTrigger,
    TokenTriggerCandidates,
)
from .model_base import BaseTokenizer

# ====================== Model Mixins =======================

## -------- Token-level access mixins ------- ##
class TokenAccessMixin(ABC):
    """Mixin for models that can access token-level inputs."""

    @abstractmethod
    def prepare_token_inputs(
        self,
        text_templates: List[str],  # n_messages texts
        initial_trigger: str,  # initial trigger string
        targets: TargetsDict = None,  # also n_messages, depends on the objective
    ) -> tuple[TokenInputsManager, TokenTrigger | str]:
        """Prepare the model's inputs object and initial trigger from raw texts.

        Args:
            texts: Can be a single string or a list of strings.
            **kwargs: Additional arguments for specific models.
        Returns:
            A tuple of (prepared inputs, initial trigger).
        """
        raise NotImplementedError

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
        self, candidate_trigger_ids: TokenTriggerCandidates, inputs: TokenInputsManager, **kwargs
    ) -> Float[Tensor, "n_messages n_candidates"]:
        """Compute the loss on the given inputs with the given trigger merged in."""
        raise NotImplementedError


class LogitsTokenAccessMixin(TokenAccessMixin):
    """Mixin for models that can compute logits based on token-level inputs."""

    @abstractmethod
    def compute_logits_from_tokens(
        self, candidate_trigger_ids: TokenTriggerCandidates, inputs: TokenInputsManager, **kwargs
    ) -> Float[Tensor, "trigger_seq_len vocab_size"]:
        """Compute logits w.r.t. `trigger` tokens that are merge into `inputs`"""
        raise NotImplementedError


## "White-box" Model Mixins:
class GradientTokenAccessMixin(TokenAccessMixin):
    """Mixin for models that can compute gradients based on token-level inputs."""

    @abstractmethod
    def compute_grad_from_tokens(
        self, candidate_trigger_ids: TokenTriggerCandidates, inputs: TokenInputsManager, **kwargs
    ) -> Float[Tensor, "trigger_seq_len vocab_size"]:
        """Compute gradients w.r.t. `trigger` tokens that are merge into `inputs`"""
        raise NotImplementedError


## -------- Text-level access mixins ------- ##


## "Black-box" Model Mixins:
class TextAccessMixin(ABC):
    def prepare_text_inputs(
        self,
        texts: List[str],  # n_messages texts
        targets: TargetsDict = None,
        initial_trigger: str = DEFAULT_INIT_TRIGGER,
    ) -> Tuple[TextInputsManager, List[str]]:
        """
        Prepares the text-based inputs manager from raw text templates.
        """
        return TextInputsManager(
            texts=texts,
            targets=targets,
        ), [initial_trigger]


class LossTextAccessMixin(TextAccessMixin):
    """Mixin for models that compute losses based on text-level inputs (black-box access)."""

    @torch.no_grad()  # in case using torch model
    def compute_loss_from_texts(
        self,
        candidate_trigger_strs: List[str],
        inputs: TextInputsManager,
        loss_func: BaseLoss,
        keep_message_dim: bool = False,
    ) -> Float[Tensor, "n_candidates"]:
        """
        Computes the loss on all candidate string texts.
        This computation is based on the __call__() method of the model, and the information it provides.
        """

        assert isinstance(
            inputs, TextInputsManager
        ), f"inputs must be of type TextInputsManager, but got {type(inputs)}"

        n_messages = inputs.n_messages
        n_candidates = len(candidate_trigger_strs)

        # Main Loop: for each message, we compute the loss for all candidates
        losses = []
        for message_idx in range(n_messages):
            curr_model_input = inputs.get_triggered_inputs(
                candidate_trigger_strs, chosen_message_idx=message_idx
            )
            curr_texts, curr_targets = (
                curr_model_input.input_texts,
                curr_model_input.targets,
            )

            # Forward pass once per message bulk
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

        losses = torch.stack(losses, dim=0)  # shape: (n_messages, n_candidates)

        if not keep_message_dim:
            losses = losses.mean(dim=0)  # shape: (n_candidates,)

        return losses
