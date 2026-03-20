from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Annotated, Any, List, Optional

import pydantic
import torch
from jaxtyping import Float

from tropt.common import Targets, TextTemplates, TokenTrigger
from tropt.loss import BaseLoss
from tropt.model import BaseModel
from tropt.tracker import BaseTracker, DummyTracker


## ------- Optimizer result ------- ##
@dataclass
class OptimizerResult:
    best_loss: float

    # Best trigger options:
    best_trigger_ids: Optional[TokenTrigger] = None
    best_trigger_str: Optional[str] = None
    best_trigger_emb: Optional[Float[torch.Tensor, "trigger_seq_len embed_dim"]] = None

    # Optiomazation records:
    losses: Optional[List[float]] = None
    trigger_strs: Optional[List[str]] = None

    # Complete artifacts:
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
        # Model requirements validation
        assert isinstance(self.model_requirements, tuple), "model_requirements must be a tuple"
        assert all(
            isinstance(m, type) for m in self.model_requirements
        ), "model_requirements must contain only classes/mixins of models."
        assert all(
            isinstance(model, m) for m in self.model_requirements
        ), f"Model {type(model)} not supported by {type(self)}"
        self.model = model

        # Loss function validation
        assert isinstance(loss, BaseLoss), "loss must be an instance of BaseLoss"
        self.loss_func = loss

        self.set_tracker(tracker if tracker is not None else DummyTracker())

        if seed is not None:
            from transformers import set_seed
            set_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)


    @abstractmethod
    @pydantic.validate_call(config=pydantic.ConfigDict(arbitrary_types_allowed=True))
    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str | TokenTrigger] = None,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:
        """Optimize the trigger to minimize the loss on the given inputs.

        Args:
            templates: Can be a single string or a list of (n_templates) strings.
            initial_trigger: Initial trigger to start optimization from, if used by the optimizer.
            targets: Target outputs for the given inputs, if applicable.

        Returns:
            Optimized trigger.
        """
        pass

    def set_tracker(self, tracker: BaseTracker):
        """
        Set the tracker for logging optimization progress.
        Useful for resetting or changing the tracker after optimizer initialization.
        """
        assert isinstance(tracker, BaseTracker), "tracker must be an instance of BaseTracker"
        self.tracker = tracker
