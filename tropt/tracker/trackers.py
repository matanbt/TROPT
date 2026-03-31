import json
import os
from collections import defaultdict
from typing import Any, Dict, Optional

import livelossplot
import wandb

from .base import DEFAULT_EXPERIMENT_NAME, BaseTracker


class DummyTracker(BaseTracker):
    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        config_dump: Optional[dict] = None,
    ):
        super().__init__(experiment_name, config_dump)

    def log(self, data: dict):
        pass

    def finish(self):
        pass

# TODO decouple the tracker to separate modules


class JSONTracker(BaseTracker):
    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        config_dump: Optional[dict] = None,
        log_file_path: str = "./logs/{experiment_name}.json",
    ):
        super().__init__(experiment_name, config_dump)
        self.log_file_path = log_file_path.format(experiment_name=experiment_name)
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        self.log_data = defaultdict(list)
        if config_dump:
            self.log_data["config"] = config_dump

    def log(self, data: dict):
        for key, value in data.items():
            self.log_data[key].append(value)

    def log_metadata(self, metadata: dict):
        self.log_data["run_metadata"] = metadata

    def finish(self):
        with open(self.log_file_path, "w") as f:
            json.dump(self.log_data, f, indent=4)


class WandbTracker(BaseTracker):
    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        project_name: str = DEFAULT_EXPERIMENT_NAME,
        config_dump: Optional[dict] = None,
        **wandb_kwargs
    ):
        """
        Initializes the WandbTracker.

        Args:
            experiment_name (str): The name of the experiment (preferably unique and informative).
            project_name (str): The name of the WandB project.
            config_dump (dict, optional): Configuration dictionary used for the experiment. Defaults to None.
            **wandb_kwargs: Additional keyword arguments for wandb.init().
        """
        super().__init__(experiment_name, config_dump)
        self.project_name = project_name

        import wandb
        wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            config=self.config_dump,
            **wandb_kwargs
        )

    def log(self, data: Dict[str, Any]):
        import torch
        sanitized = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                try:
                    v = v.item()
                except ValueError:
                    continue
            sanitized[k] = v
        wandb.log(sanitized)

    def log_metadata(self, metadata: dict):
        wandb.config.update(metadata, allow_val_change=True)

    def finish(self):
        wandb.finish()

class DictTracker(BaseTracker):
    """Accumulates logged values in plain Python dicts.

    Attributes:
        records (list[dict]): Each ``log()`` call appends one record (the raw dict).
        history (dict[str, list]): Per-key view — ``history[key]`` contains only values
            from records that included *key*. Convenient but records from different keys
            may not be index-aligned; use ``records`` when you need to join across keys.
        metadata (dict): Run metadata, if logged.
    """

    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        config_dump: Optional[dict] = None,
    ):
        super().__init__(experiment_name, config_dump)
        self.records: list[dict] = []
        self.history: Dict[str, list] = defaultdict(list)
        self.metadata: dict = {}

    def log(self, data: dict):
        self.records.append(data)
        for key, value in data.items():
            self.history[key].append(value)

    def log_metadata(self, metadata: dict):
        self.metadata = metadata

    def finish(self):
        pass


# TODO Add HF's trackio integration

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
        config_dump: Optional[dict] = None,
        print_keys: tuple = ("loss", "best_trigger_str"),
    ):
        super().__init__(experiment_name, config_dump)
        self.history = defaultdict(list)
        self._step = 0
        self.print_keys = print_keys

    def log(self, data: dict):
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

    def log_metadata(self, metadata: dict):
        print(f"=== Run metadata: {metadata} ===", flush=True)

    def finish(self):
        pass


class LiveLossPlotTracker(BaseTracker):
    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        config_dump: Optional[dict] = None,
        focus_on_metrics: tuple = ("loss",),
        **llp_kwargs
    ):
        """
        Initializes the LiveLossPlotTracker.
        """
        super().__init__(experiment_name)
        self._plotlosses = livelossplot.PlotLosses()
        self.focus_on_metrics = focus_on_metrics

    def log(self, data: Dict[str, Any]):
        self._plotlosses.update({
            k: v for k, v in data.items()
            if (isinstance(v, (int, float))
                and k in self.focus_on_metrics)
        })
        self._plotlosses.send()

    def finish(self):
        pass
