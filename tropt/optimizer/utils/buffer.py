from typing import Optional

import torch


class TriggerBuffer:
    """
    Enables maintaining a buffer of the best triggers found during optimization.
    https://www.haizelabs.com/blog/making-a-sota-adversarial-attack-on-llms-38x-faster
    https://arxiv.org/pdf/2402.12329
    """
    # TODO can optimize this class using heap (Python's heapq) for better performance on
    #      large buffers (which are currently rare in the package)

    def __init__(
            self,
            triggers: Optional[list[torch.Tensor]] = None,
            losses: Optional[list[float]] = None,
        ):
        self.triggers = triggers or []  # List of trigger token ID tensors
        self.losses = losses or []  # Corresponding list of losses

    @property
    def size(self) -> int:
        return len(self.triggers)

    def add(self, trigger_ids: torch.Tensor, loss: float):
        """
        Adds a new trigger and its loss to the buffer.
        Increases the buffer size by one.
        """
        self.triggers.append(trigger_ids)
        self.losses.append(loss)

    def add_if_better(self, trigger_ids: torch.Tensor, loss: float):
        """
        Adds the trigger to the buffer if its loss is better than the worst in the buffer.
        Retains the buffer size.
        """
        max_loss = self.get_highest_loss()
        if loss < max_loss:
            max_loss_idx = self.losses.index(max_loss)
            self.triggers[max_loss_idx] = trigger_ids
            self.losses[max_loss_idx] = loss

    def get_best_trigger(self) -> torch.Tensor:
        #TODO add an arg to sample from top-k to enable exploration
        min_loss_idx = self.losses.index(min(self.losses))
        return self.triggers[min_loss_idx]

    def get_highest_loss(self) -> float:
        if not self.losses:
            return float('inf')
        return max(self.losses)

    def get_lowest_loss(self) -> float:
        if not self.losses:
            return float('inf')
        return min(self.losses)
