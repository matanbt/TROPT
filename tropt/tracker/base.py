import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME = "tropt_experiment"

class BaseTracker(ABC):
    """Interface for experiment trackers.

    Lifecycle (managed by ``BaseOptimizer``'s wrapper)::

        __init__()        # by the user: stores backend config, no run started
        init(config)      # by the optimizer wrapper: opens a run (now stateful)
        log(data)         # per optimization step
        finish(summary)   # by the optimizer wrapper: closes the run (back to stateless)

    The tracker can be reused across multiple runs: after ``finish()``,
    a subsequent ``init()`` opens a fresh run on the same backend.

    Subclasses must implement ``_init``, ``_log``, and ``_finish``.
    The base class manages their guards.
    """

    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        experiment_config: Optional[dict] = None,
    ):
        self.experiment_name = experiment_name
        self.experiment_config = experiment_config
        self._active = False

    # ── Public interface (guarded) ──────────────────────────────────────

    def init(self, config: Optional[dict] = None):
        """Open a new run. Called by the optimizer wrapper before optimization.

        Merges the user's ``experiment_config`` (from ``__init__``) with the
        optimizer-generated ``config`` and forwards to ``_init``.
        """
        if self._active:
            logger.warning(
                "init() called on already-active tracker (previous run not finished). "
                "Auto-finishing the previous run."
            )
            self.finish()
        self._active = True
        merged = {**(self.experiment_config or {}), **(config or {})} or None
        self._init(merged)

    def log(self, data: dict):
        """Log per-step data. Must be called between ``init`` and ``finish``."""
        assert self._active, "log() called on inactive tracker (missing init()?)"
        self._log(data)

    def finish(self, summary: Optional[dict] = None):
        """Close the current run, optionally logging a final summary."""
        assert self._active, "finish() called on inactive tracker (missing init()?)"
        self._finish(summary)
        self._active = False

    # ── Subclass hooks ──────────────────────────────────────────────────

    @abstractmethod
    def _init(self, config: Optional[dict] = None):
        """Open a run on the backend. ``config`` contains run metadata/hparams."""
        ...

    @abstractmethod
    def _log(self, data: dict):
        """Write one step's data to the backend."""
        ...

    @abstractmethod
    def _finish(self, summary: Optional[dict] = None):
        """Close the run, writing ``summary`` if provided."""
        ...
