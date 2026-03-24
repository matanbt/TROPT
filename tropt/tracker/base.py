from abc import ABC, abstractmethod
from typing import Optional

DEFAULT_EXPERIMENT_NAME = "tropt_experiment"

class BaseTracker(ABC):
    """
    Base class for trackers.
    Supports context manager usage.
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
        """Closes the tracker and performs any necessary cleanup."""
        pass

    def log_metadata(self, metadata: dict):
        """Logs run metadata (hparams, model name, templates, targets) once at run start.

        Called automatically by BaseOptimizer before optimization begins.
        Override in subclasses to persist metadata in the appropriate backend.
        """
        pass

