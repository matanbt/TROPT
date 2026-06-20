from dataclasses import dataclass, field
from typing import List, Optional

from jaxtyping import Float, Int
from torch import Tensor


@dataclass
class RunningBest:
    """
    An auxiliary object for optimizers:
    accumulates per-step losses and tracks the best trigger found so far.

    Stores:
        loss: the best loss found so far
        trigger_ids: the token IDs of the best trigger found so far
        trigger_str: the string of the best trigger found so far
        trigger_emb: the embedding of the best trigger found so far (for gradient-based optimizers; optional)
        step: the step at which the best trigger was found

        losses: a list of all losses observed at each step
        trigger_strs: a list of all trigger strings observed at each step
    """

    loss: float = float("inf")
    trigger_ids: Optional[Int[Tensor, "trigger_seq_len"]] = None
    trigger_str: Optional[str] = None
    trigger_emb: Optional[Float[Tensor, "trigger_seq_len embed_dim"]] = None
    step: int = -1

    losses: List[float] = field(default_factory=list)
    trigger_strs: List[str] = field(default_factory=list)

    def update(
        self,
        loss: float,
        trigger_ids: Optional[Int[Tensor, "trigger_seq_len"]] = None,
        trigger_str: Optional[str] = None,
        trigger_emb: Optional[Float[Tensor, "trigger_seq_len embed_dim"]] = None,
    ) -> bool:
        """
        Record a step and update the best if improved. Returns True on new best.

        Args:
            loss: the loss observed at the current step
            trigger_ids: the token IDs of the trigger at the current step
            trigger_str: the string of the trigger at the current step
            trigger_emb: the embedding of the trigger at the current step (for gradient-based optimizers; optional)
        Returns:
            bool: True if this is a new best, False otherwise.
        """
        self.losses.append(loss)
        self.trigger_strs.append(trigger_str)

        if loss < self.loss:
            self.loss = loss
            self.trigger_ids = trigger_ids.clone() if trigger_ids is not None else None
            self.trigger_str = trigger_str
            self.trigger_emb = trigger_emb.clone() if trigger_emb is not None else None
            self.step = len(self.losses) - 1
            return True
        return False

    def to_result(self):
        """Convert to an OptimizerResult."""
        from tropt.optimizer.base import OptimizerResult
        return OptimizerResult(
            best_loss=self.loss,
            best_trigger_str=self.trigger_str,
            best_trigger_ids=self.trigger_ids,
            best_trigger_emb=self.trigger_emb,
            losses=self.losses,
            trigger_strs=self.trigger_strs,
        )
