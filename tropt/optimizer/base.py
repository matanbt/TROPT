from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from tropt.loss.base import BaseLoss
from tropt.models.base import BaseModel, TargetsDict, TokenTrigger
from tropt.tracker.base import BaseTracker, DummyTracker


## ------- Optimizer result ------- ##
@dataclass
class OptimizerResult:  # TODO rethink it
    best_trigger: TokenTrigger
    best_trigger_str: str
    best_loss: float
    trigger_strs: List[str]
    losses: Optional[List[float]] = None
    full_prompt: Optional[str | List[str]] = None

## ------- Base Optimizer ------- ##
class BaseOptimizer(ABC):
    # list of model mixin classes that the target `model` must implement to be compatible with this optimizer
    model_requirements = []

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss=None,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
    ):
        assert all(
            isinstance(model, m) for m in self.model_requirements
        ), f"Model {type(model)} not supported by {type(self)}"
        self.model = model

        self.loss_func = loss
        self.tracker = tracker if tracker is not None else DummyTracker()
        # TODO validate the loss is supported by the model (each model should have its supported losses listed)

        if seed is not None:
            from transformers import set_seed
            set_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)


    @abstractmethod
    def optimize_trigger(
        self,
        texts: List[str],
        initial_trigger: Optional[str] | str | TokenTrigger = None,
        targets: TargetsDict = None,
    ) -> OptimizerResult:
        """Optimize the trigger to minimize the loss on the given inputs.

        Args:
            inputs: Can be a single string or a list of (n_messages) strings.
            initial_trigger: Initial trigger to start optimization from.

        Returns:
            Optimized trigger.
        """
        raise NotImplementedError
