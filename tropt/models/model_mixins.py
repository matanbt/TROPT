from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizer

from tropt.common import DEFAULT_INIT_TRIGGER
from tropt.loss.base import BaseLoss, CombinedLoss, EmbeddingBasedLoss
from tropt.loss.text_loss import InputReadabilityLoss

from .inputs import (
    BatchedTargetsDict,
    MessageBatchedTargetsDict,
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
        """Computes the loss on all candidate string texts."""

        assert isinstance(
            inputs, TextInputsManager
        ), f"inputs must be of type TextInputsManager, but got {type(inputs)}"

        n_messages = inputs.n_messages
        n_candidates = len(candidate_trigger_strs)

        # Helper function to calculate loss from outputs
        def _calc_loss_from_outputs(_inputs, _outputs, _curr_targets, _loss_func):
            if isinstance(_loss_func, EmbeddingBasedLoss):
                return _loss_func(
                    _outputs, _curr_targets[_loss_func.TARGET_KEY]
                )  # shape: (n_candidates,)
            
            if isinstance(_loss_func, InputReadabilityLoss):  # TODO more general super class here
                return _loss_func(
                    _inputs,
                )  # shape: (n_candidates,)

            elif isinstance(_loss_func, CombinedLoss):
                child_losses = []
                for _nested_loss_func in _loss_func.loss_funcs:
                    # Recursive call
                    child_losses.append(
                        _calc_loss_from_outputs(_inputs, _outputs, _curr_targets, _nested_loss_func)
                    )

                # Stack child losses: (n_candidates, n_losses)
                child_losses = torch.stack(child_losses, dim=1)

                # Apply the combinator (e.g. weighted sum) -> (n_candidates,)
                return _loss_func(child_losses)

            else:
                raise NotImplementedError(f"Loss function {_loss_func} not supported.")

        # Main Loop: for each message, we compute the loss for all candidates
        losses = []
        for message_idx in range(n_messages):
            curr_inputs_dict = inputs.get_triggered_inputs(
                candidate_trigger_strs, chosen_message_idx=message_idx
            )
            curr_texts, curr_targets = (
                curr_inputs_dict["inputs_texts"],
                curr_inputs_dict["targets"],
            )

            # Forward pass once per message bulk
            outputs = self(
                curr_texts
            )  # could be a list of n_candidate strings, a tensor of (n_candidates, d_model), etc.

            # Calculate loss
            loss = _calc_loss_from_outputs(curr_texts, outputs, curr_targets, loss_func)  # shape: (n_candidates,)
            losses.append(loss)

        losses = torch.stack(losses, dim=0)  # shape: (n_messages, n_candidates)

        if not keep_message_dim:
            losses = losses.mean(dim=0)  # shape: (n_candidates,)

        return losses
