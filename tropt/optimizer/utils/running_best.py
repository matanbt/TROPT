from dataclasses import dataclass, field
from typing import List, Optional

from jaxtyping import Int
from torch import Tensor


@dataclass
class RunningBest:
    """Accumulates per-step losses and tracks the best trigger found so far."""

    loss: float = float("inf")
    trigger_ids: Optional[Int[Tensor, "trigger_seq_len"]] = None
    trigger_str: Optional[str] = None
    step: int = -1

    losses: List[float] = field(default_factory=list)
    trigger_strs: List[str] = field(default_factory=list)

    def update(
        self,
        loss: float,
        trigger_ids: Int[Tensor, "trigger_seq_len"],
        trigger_str: str,
    ) -> bool:
        """Record a step and update the best if improved. Returns True on new best."""
        self.losses.append(loss)
        self.trigger_strs.append(trigger_str)

        if loss < self.loss:
            self.loss = loss
            self.trigger_ids = trigger_ids.clone()
            self.trigger_str = trigger_str
            self.step = len(self.losses) - 1
            return True
        return False
