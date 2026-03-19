"""
Base classes for loss functions.

Important note: The losses arguments must match the fields in ModelOutput and ModelInput
for unified loss resolution to work properly.
"""


import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import torch
from jaxtyping import Float
from torch import Tensor

logger = logging.getLogger(__name__)


class BaseLoss(ABC):
    """Base class for all loss functions."""

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Float[Tensor, "bsz"]:
        pass

    def contains_loss_type(self, loss_type: type) -> bool:
        """
        Check if the loss is of the specified type.
        Complicated losses (e.g., CombinedLoss) may override this method with different logic.
        """
        return isinstance(self, loss_type)

############################

@dataclass
class CombinedLoss(BaseLoss):
    """Combines multiple losses with given weights."""

    def __init__(self, loss_funcs: List[BaseLoss], weights: List[float] = None) -> None:
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
        self.last_component_losses = losses.detach()  # shape: (n_losses, bsz)

        weights = self.weights.to(losses).unsqueeze(-1)  # shape: (n_losses, 1)
        loss = losses * weights
        loss = loss.sum(dim=0)  # recude over n_losses

        return loss  # shape: (bsz,)

    def get_component_losses_dict(self) -> dict[str, Float[Tensor, "bsz"]]:
        """Returns {loss_class_name: loss_values} from the last __call__."""
        return {
            type(lf).__name__: val
            for lf, val in zip(self.loss_funcs, self.last_component_losses)
        }

    def contains_loss_type(self, loss_type: type) -> bool:
        """Check if the CombinedLoss contains a loss of the specified type."""
        return any(isinstance(loss, loss_type) for loss in self.loss_funcs)

    def __iter__(self):
        """Allows iterating over the nested loss functions."""
        return iter(self.loss_funcs)


