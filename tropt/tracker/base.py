import json
import os
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict

import livelossplot
import wandb


class BaseTracker(ABC):
    """
    Base class for trackers.
    Supports context manager usage.
    """
    def __init__(
        self,
        experiment_name: str,
        config_dump: dict = None,
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

class DummyTracker(BaseTracker):
    def __init__(
        self,
        experiment_name: str = "dummy",
        config_dump: dict = None,
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
        experiment_name: str,
        config_dump: dict = None,
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

    def finish(self):
        with open(self.log_file_path, "w") as f:
            json.dump(self.log_data, f, indent=4)



class WandbTracker(BaseTracker):
    def __init__(
        self,
        experiment_name: str,
        project_name: str,
        config_dump: dict = None,
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
        wandb.log(data)

    def finish(self):
        wandb.finish()

# TODO Add HF's trackio

class LiveLossPlotTracker(BaseTracker):
    def __init__(
        self,
        experiment_name: str,
        config_dump: dict = None,
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
    