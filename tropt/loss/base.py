"""
Base classes for loss functions.

Important note: The losses arguments must match the fields in ModelOutput and ModelInput
for unified loss resolution to work properly.
"""


import functools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Optional

import torch
from jaxtyping import Float
from torch import Tensor

logger = logging.getLogger(__name__)


class BaseLoss(ABC):
    """Base class for all loss functions."""

    is_differentiable: ClassVar[bool] = True
    """Whether this loss is back-propable. Set to False for losses that use external
    models, text generation, or other non-differentiable operations."""

    requires_target_prefill: ClassVar[bool] = False
    """Whether this loss requires the model to prefill the target response tokens (appending
    them to the input, as a response prefix)."""

    requires_generation: ClassVar[bool] = False
    """Whether this loss requires autoregressive generation."""

    requires_hidden_states: ClassVar[bool] = False
    """Whether this loss requires the model to provid the forward pass's hidden states."""

    requires_attentions: ClassVar[bool] = False
    """Whether this loss requires the model to return attention weights."""

    _last_loss_vals: Optional[Float[Tensor, "bsz"]] = None
    """Loss values from the most recent __call__, shape (bsz,). Set automatically by __init_subclass__."""

    def __post_init__(self):
        """Initialize instance state."""
        self._last_loss_vals = None

    def __init_subclass__(cls, **kwargs):
        """
        Wraps __call__ in subclasses to automatically record _last_loss_vals after every call.
        """
        super().__init_subclass__(**kwargs)
        if "__call__" in cls.__dict__:
            original = cls.__dict__["__call__"]

            @functools.wraps(original)
            def wrapped(self, *args, **kwargs):
                result = original(self, *args, **kwargs)
                self._last_loss_vals = result.detach()
                return result

            setattr(cls, "__call__", wrapped)

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Float[Tensor, "bsz"]:
        """
        Given values formatted as ModelInput / ModelOutput / Targets fields,
        computes the loss, and returns per-batch-element loss values.
        """
        pass

    def get_loss_log_dict(self) -> dict:
        """
        Returns a loggable dict of the last computed loss value, keyed by loss class name.
        Useful for verbose loss logging in optimizers.
        """
        if self._last_loss_vals is None:
            return {}
        return {f"loss/{type(self).__name__}": self._last_loss_vals.min().item()}

    def contains_loss_type(self, loss_type: type) -> bool:
        """
        Returns True if this loss is of the given type.
        Complicated losses (e.g., CombinedLoss) may override this method with different logic.
        """
        return isinstance(self, loss_type)

############################

@dataclass
class CombinedLoss(BaseLoss):
    """Combines multiple losses with given weights."""

    _last_component_loss_vals: Optional[Float[Tensor, "n_losses bsz"]] = None
    """Per-component loss values from the most recent __call__, shape (n_losses, bsz)."""

    def __init__(self, loss_funcs: List[BaseLoss], weights: Optional[List[float]] = None) -> None:
        assert weights is None or len(loss_funcs) == len(weights), "Length mismatch between loss_funcs and weights"
        assert all(isinstance(loss, BaseLoss) for loss in loss_funcs), "All elements in losses must be instances of BaseLoss"
        assert all(not isinstance(loss, CombinedLoss) for loss in loss_funcs), "CombinedLoss cannot contain another CombinedLoss"
        self.loss_funcs: List[BaseLoss] = loss_funcs

        if weights is None:
            # if weights not provided, set equal weights
            self.weights: Float[Tensor, "n_losses"] = torch.ones(len(loss_funcs)) / len(loss_funcs)
        else:
            self.weights: Float[Tensor, "n_losses"] = torch.tensor(weights, dtype=torch.float32)

    def __call__(self, losses: Float[Tensor, "n_losses bsz"]) -> Float[Tensor, "bsz"]:
        """
        Compute the combined loss (weighted sum).
        Args:
            losses: Tensor of shape (n_losses, bsz), each row corresponds to the loss values from each loss function.
        Returns:
            Tensor of shape (bsz,), the combined loss for each element in the batch.
        """
        self._last_component_loss_vals = losses.detach()  # shape: (n_losses, bsz)

        weights = self.weights.to(losses).unsqueeze(-1)  # shape: (n_losses, 1)
        loss = losses * weights
        loss = loss.sum(dim=0)  # recude over n_losses

        return loss  # shape: (bsz,)

    def get_loss_log_dict(self) -> dict:
        """
        Returns a loggable dict of the last computed loss value (of *all* the component losses), keyed by loss class name.
        Useful for verbose loss logging in optimizers.
        """
        if self._last_loss_vals is None or self._last_component_loss_vals is None:
            return {}
        return {
            f"loss/{type(self).__name__}": self._last_loss_vals.min().item(),
            **{
                f"loss/{type(lf).__name__}": val.min().item()
                for lf, val in zip(self.loss_funcs, self._last_component_loss_vals)
            },
        }

    @property
    def is_differentiable(self) -> bool:
        return all(lf.is_differentiable for lf in self.loss_funcs)

    @property
    def requires_target_prefill(self) -> bool:
        return any(lf.requires_target_prefill for lf in self.loss_funcs)

    @property
    def requires_generation(self) -> bool:
        return any(lf.requires_generation for lf in self.loss_funcs)

    @property
    def requires_hidden_states(self) -> bool:
        return any(lf.requires_hidden_states for lf in self.loss_funcs)

    @property
    def requires_attentions(self) -> bool:
        return any(lf.requires_attentions for lf in self.loss_funcs)

    def contains_loss_type(self, loss_type: type) -> bool:
        """Check if the CombinedLoss contains a loss of the specified type."""
        return any(isinstance(loss, loss_type) for loss in self.loss_funcs)

    def __iter__(self):
        """Allows iterating over the nested loss functions."""
        return iter(self.loss_funcs)


