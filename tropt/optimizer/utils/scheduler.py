import math
from abc import ABC, abstractmethod

"""
Schedulers for the n_flip parameter used in optimizers to control the number
of token positions flipped during each optimization step.
"""

class NFlipScheduler(ABC):
    @abstractmethod
    def get_n_flip(self, step: int) -> int:
        """Returns the n_flip value for the given step (0-indexed)."""
        pass

class ConstantScheduler(NFlipScheduler):
    """
    A scheduler that always returns the same n_flip value.
    """
    def __init__(self, n_flip: int):
        self.n_flip = n_flip

    def get_n_flip(self, step: int) -> int:
        return self.n_flip

class LinearScheduler(NFlipScheduler):
    """
    A scheduler that linearly decreases n_flip from an initial value to 1
    over the course of optimization steps, starting from a specified step.
    """
    def __init__(self, initial_n_flip: int, total_steps: int, decline_start: int | float):
        self.initial_n_flip = initial_n_flip
        self.total_steps = total_steps
        if isinstance(decline_start, float):
            self.decline_start_step = int(total_steps * decline_start)
        else:
            self.decline_start_step = int(decline_start)

    def get_n_flip(self, step: int) -> int:
        if step < self.decline_start_step:
            return self.initial_n_flip

        steps_remaining = self.total_steps - step
        decline_duration = self.total_steps - self.decline_start_step

        if decline_duration <= 0:
            return 1

        ratio = steps_remaining / decline_duration
        return max(1, math.ceil(self.initial_n_flip * ratio))
