from abc import ABC, abstractmethod
from typing import Optional

DEFAULT_EXPERIMENT_NAME = "tropt_experiment"

class BaseTracker(ABC):
    """Interface for trackers.

    In this project, the optimizer do most interaction with the trackers.Concretely, ``BaseOptimizer`` (from which other optimizers inherit) calls ``log`` (per-step logging), ``log_metadata`` (one-time logging at start), and ``finish`` (finalization) on the tracker.
    Also usable as a
    context manager.
    """

    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        config_dump: Optional[dict] = None,
    ):
        self.experiment_name = experiment_name
        self.config_dump = config_dump

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.finish()

    @abstractmethod
    def log(self, data: dict):
        """Logs data to the logging backend.

        Args:
            data (dict): A dictionary containing the data to log.
        """
        pass

    @abstractmethod
    def finish(self):
        """Flush/close the backend tracker. 
        
        In this project, finish is handled automatically by ``BaseOptimizer`` at the end of ``optimize_trigger``."""
        pass

    def log_metadata(self, metadata: dict):
        """Logs run metadata (hparams, model name, templates, targets) once at run start.

        Called automatically by ``BaseOptimizer`` before optimization begins.
        """
        pass

