from typing import Optional

import torch


class TriggerBuffer:
    """
    Enables maintaining a buffer of the best triggers found during optimization.
    https://www.haizelabs.com/blog/making-a-sota-adversarial-attack-on-llms-38x-faster
    https://arxiv.org/pdf/2402.12329
    """
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
        if not self.losses:
            self.add(trigger_ids, loss)
            return
        max_loss = self.get_highest_loss()
        if loss < max_loss:
            max_loss_idx = self.losses.index(max_loss)
            self.triggers[max_loss_idx] = trigger_ids
            self.losses[max_loss_idx] = loss

    def get_best_trigger(self, top_k: int = 1) -> torch.Tensor:
        """Return the lowest-loss trigger; if ``top_k > 1``, sample uniformly from the top_k for exploration."""
        # Note: this can be optimized by maintaining a heap; but large buffers are currently rare in the package, so we keep it simple for now.
        if top_k <= 1:
            min_loss_idx = self.losses.index(min(self.losses))
            return self.triggers[min_loss_idx]
        sorted_indices = sorted(range(len(self.losses)), key=lambda i: self.losses[i])[:top_k]
        return self.triggers[sorted_indices[torch.randint(len(sorted_indices), (1,)).item()]]

    def get_highest_loss(self) -> float:
        if not self.losses:
            return float('inf')
        return max(self.losses)

    def get_lowest_loss(self) -> float:
        if not self.losses:
            return float('inf')
        return min(self.losses)
