import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, Optional

import livelossplot
import trackio
from livelossplot.outputs import MatplotlibPlot

import wandb

from .base import DEFAULT_EXPERIMENT_NAME, BaseTracker

logger = logging.getLogger(__name__)


class DummyTracker(BaseTracker):
    """No-op tracker. Discards all logged data."""

    def _init(self, config: Optional[dict] = None):
        pass

    def _log(self, data: dict):
        pass

    def _finish(self, summary: Optional[dict] = None):
        pass


class JSONTracker(BaseTracker):
    """Writes accumulated logs to a JSON file on ``finish()``."""

    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        experiment_config: Optional[dict] = None,
        log_file_path: str = "./logs/{experiment_name}.json",
    ):
        super().__init__(experiment_name, experiment_config)
        self.log_file_path = log_file_path.format(experiment_name=experiment_name)
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        self._log_data: dict = {}

    def _init(self, config: Optional[dict] = None):
        self._log_data = defaultdict(list)
        if config:
            self._log_data["config"] = config

    def _log(self, data: dict):
        for key, value in data.items():
            self._log_data[key].append(value)

    def _finish(self, summary: Optional[dict] = None):
        if summary:
            self._log_data["run_summary"] = summary
        with open(self.log_file_path, "w") as f:
            json.dump(self._log_data, f, indent=4)


class WandbTracker(BaseTracker):
    """Logs to Weights & Biases.

    Construction stores backend parameters (project, entity, etc.)
    without starting a run. The run is opened on ``init()`` and closed
    on ``finish()``.
    """

    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        experiment_config: Optional[dict] = None,
        project_name: str = DEFAULT_EXPERIMENT_NAME,
        **wandb_kwargs,
    ):
        """
        Args:
            experiment_name: Run name in WandB.
            project_name: WandB project name.
            experiment_config: User-provided config merged with optimizer config on every run.
            **wandb_kwargs: Extra kwargs forwarded to ``wandb.init()``.
        """
        super().__init__(experiment_name, experiment_config)
        self.project_name = project_name
        self._wandb_kwargs = wandb_kwargs

    def _init(self, config: Optional[dict] = None):
        wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            config=config,
            **self._wandb_kwargs,
        )

    def _log(self, data: Dict[str, Any]):
        import torch
        sanitized = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                try:
                    v = v.item()
                except (ValueError, RuntimeError):
                    continue
            sanitized[k] = v
        wandb.log(sanitized)

    def _finish(self, summary: Optional[dict] = None):
        if summary:
            wandb.run.summary.update(summary)
        wandb.finish()


class DictTracker(BaseTracker):
    """Accumulates logged values in plain Python dicts.

    Attributes:
        records (list[dict]): Each ``log()`` call appends one record (the raw dict).
        history (dict[str, list]): Per-key view — ``history[key]`` contains only values
            from records that included *key*. Convenient but records from different keys
            may not be index-aligned; use ``records`` when you need to join across keys.
        config (dict): Run config, if logged.
        summary (dict): Run summary, if logged.
    """

    def _init(self, config: Optional[dict] = None):
        self.records: list[dict] = []
        self.history: Dict[str, list] = defaultdict(list)
        self.config: dict = config or {}
        self.summary: dict = {}

    def _log(self, data: dict):
        self.records.append(data)
        for key, value in data.items():
            self.history[key].append(value)

    def _finish(self, summary: Optional[dict] = None):
        self.summary = summary or {}


class TrackioTracker(BaseTracker):
    """Logs to Hugging Face's Trackio.

    Construction stores backend parameters (project, space_id, etc.) without starting a run.
    The run is opened on ``init()`` and closed on ``finish()``.

    Trackio has no per-run summary object (unlike WandB); ``finish(summary=...)`` records the
    summary as a final ``trackio.log()`` entry whose keys are prefixed by ``"summary/"`` so it
    can be recovered from the run history.

    See https://huggingface.co/docs/trackio for backend details.
    """

    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        experiment_config: Optional[dict] = None,
        project_name: str = DEFAULT_EXPERIMENT_NAME,
        space_id: Optional[str] = None,
        **trackio_kwargs,
    ):
        """
        Args:
            experiment_name: Run name in Trackio.
            project_name: Trackio project name.
            experiment_config: User-provided config merged with optimizer config on every run.
            space_id: Optional HuggingFace Space identifier (``"user/space_name"``) for hosted
                dashboards. If ``None``, Trackio persists locally to ``~/.trackio``.
            **trackio_kwargs: Extra kwargs forwarded to ``trackio.init()``.
        """
        super().__init__(experiment_name, experiment_config)
        self.project_name = project_name
        self.space_id = space_id
        self._trackio_kwargs = trackio_kwargs

    def _init(self, config: Optional[dict] = None):
        trackio.init(
            project=self.project_name,
            name=self.experiment_name,
            config=config,
            space_id=self.space_id,
            **self._trackio_kwargs,
        )

    def _log(self, data: Dict[str, Any]):
        import torch
        sanitized = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                try:
                    v = v.item()
                except (ValueError, RuntimeError):
                    continue
            sanitized[k] = v
        trackio.log(sanitized)

    def _finish(self, summary: Optional[dict] = None):
        if summary:
            trackio.log({f"summary/{k}": v for k, v in summary.items()})
        trackio.finish()


class PrintTracker(BaseTracker):
    """Prints each optimisation step to stdout and accumulates history.

    Useful for Jupyter notebooks or any situation where you want live
    step-by-step loss/trigger output without a heavyweight logging backend.

    Attributes:
        history (dict): Accumulated values keyed by metric name.
    """

    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        print_keys: tuple = ("loss", "best_trigger_str"),
    ):
        super().__init__(experiment_name)
        self.print_keys = print_keys
        self.history: dict = defaultdict(list)
        self._step = 0

    def _init(self, config: Optional[dict] = None):
        self._step = 0
        if config:
            print(f"=== Run config: {config} ===", flush=True)

    def _log(self, data: dict):
        self._step += 1
        for key, val in data.items():
            self.history[key].append(val)
        parts = [f"step={self._step:>4}"]
        for key in self.print_keys:
            if key in data:
                val = data[key]
                parts.append(
                    f"{key}={val:.4f}" if isinstance(val, float) else f"{key}={val!r}"
                )
        print(" | ".join(parts), flush=True)

    def _finish(self, summary: Optional[dict] = None):
        if summary:
            parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v!r}" for k, v in summary.items()]
            print(f"=== Run summary: {' | '.join(parts)} ===", flush=True)


class LiveLossPlotTracker(BaseTracker):
    """Live-updating loss plot via ``livelossplot``."""

    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        focus_on_metrics: tuple = ("loss",),
    ):
        super().__init__(experiment_name)
        self.focus_on_metrics = focus_on_metrics
        self._plotlosses: Optional[livelossplot.PlotLosses] = None

    def _init(self, config: Optional[dict] = None):
        def _relabel_x(axes):
            for ax in axes.values():
                ax.set_xlabel("step")
        self._plotlosses = livelossplot.PlotLosses(
            outputs=[MatplotlibPlot(figsize=(7, 3), after_plots=_relabel_x)],
        )

    def _log(self, data: Dict[str, Any]):
        self._plotlosses.update({
            k: v for k, v in data.items()
            if (isinstance(v, (int, float))
                and k in self.focus_on_metrics)
        })
        self._plotlosses.send()

    def _finish(self, summary: Optional[dict] = None):
        pass
