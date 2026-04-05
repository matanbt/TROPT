import functools
import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import pydantic
import torch
from jaxtyping import Float
from tqdm import tqdm

from tropt.common import Targets, TextTemplates, TokenTrigger
from tropt.loss import BaseLoss
from tropt.model import BaseModel
from tropt.tracker import BaseTracker, DummyTracker

logger = logging.getLogger(__name__)


## ------- Optimizer result ------- ##
@dataclass
class OptimizerResult:
    best_loss: float

    # Best trigger options:
    best_trigger_ids: Optional[TokenTrigger] = None
    best_trigger_str: Optional[str] = None
    best_trigger_emb: Optional[Float[torch.Tensor, "trigger_seq_len embed_dim"]] = None
    best_trigger_probs: Optional[Float[torch.Tensor, "trigger_seq_len vocab_size"]] = None

    # Optiomazation records:
    losses: Optional[List[float]] = None
    trigger_strs: Optional[List[str]] = None

    # Complete artifacts:
    full_prompt: Optional[str | List[str]] = None

    def to_dict(self) -> dict:
        """Lightweight summary dict for final logging (no tensors or lists !)."""
        d: dict = {"best_loss": self.best_loss}
        if self.best_trigger_str is not None:
            d["best_trigger_str"] = self.best_trigger_str
        if self.full_prompt is not None:
            d["full_prompt"] = self.full_prompt
        return d

## ------- Base Optimizer ------- ##
class BaseOptimizer(ABC):
    """
    Base class for all trigger optimizers.

    - Implements common functionality and interface for optimizers, including tracking.
    - Subclasses must implement the ``optimize_trigger`` method, which contains the core optimization loop and returns an ``OptimizerResult``; this method is automatically wrapped to handle logging, model state resets, and tracker finalization.
    """

    model_requirements = ()
    """Tuple of model mixin classes the primary model must satisfy; validated in ``__init__``.

    Convention: declare the least-restrictive configuration the optimizer supports.

    - Black-box only -> ``(LossTextAccessMixin,)``, even if gradient modes exist.
    - Always needs token-level loss → include ``LossTokenAccessMixin``.
    - Always needs gradients → include ``GradientTokenAccessMixin`` or ``GradientEmbedAccessMixin``.

    Notes:
    - Requirements that only occur in an optional flow (e.g.,
    ``candidate_selection="gradient"`` requiring ``GradientTokenAccessMixin``) must be
    validated explicitly in ``__init__`` after ``super().__init__()``, with an assert/error.
    - Requirements on auxiliary models (proxy_model, util_model, etc.) are not covered here, and should also be validated explicitly in ``__init__``.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Inject a wrapper around optimize_trigger
        # This saves some boilerplate in optimizers impl.
        if "optimize_trigger" in cls.__dict__:
            original = cls.optimize_trigger
            # Capture once at class-definition time (not per call).
            # _sig is needed to resolve subclass-specific defaults (e.g. DEFAULT_INIT_TRIGGER)
            # so log_config receives the actual values used, not None.
            _sig = inspect.signature(original)
            validated = pydantic.validate_call(
                config=pydantic.ConfigDict(arbitrary_types_allowed=True)
            )(original)

            @functools.wraps(original)
            def _wrapper(self, *args, **kwargs):
                # Reset usage stats on all models before the run
                for val in self.__dict__.values():
                    if isinstance(val, BaseModel):
                        val.reset_usage_stats()

                # Resolve full arg values (including defaults) for logging
                ba = _sig.bind(self, *args, **kwargs)
                ba.apply_defaults()
                self.log_config(ba.arguments.get("templates"), ba.arguments.get("initial_trigger"), ba.arguments.get("targets"))

                # Run the actual optimization method
                result: OptimizerResult = validated(self, *args, **kwargs)

                # Post-run teardown: log summary, reset all model input state, close tracker
                self.log(**result.to_dict())
                # Make sure we reset the model's input state after the optimization
                # (just to avoid any unexpected, accidental state leakage across runs, although optimizers
                #  should set the state (`set_inputs_from_*`) and override it anyway)
                for val in self.__dict__.values():
                    if isinstance(val, BaseModel):
                        for _reset in ("reset_inputs_from_tokens", "reset_inputs_from_texts"):
                            fn = getattr(val, _reset, None)
                            if fn is not None:
                                fn()
                self.tracker.finish()
                return result

            setattr(cls, 'optimize_trigger', _wrapper)

    def __init__(
        self,
        model: BaseModel,
        loss: Optional[BaseLoss] = None,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            model: The target model to attack.
            loss: The loss function to be optimized (optional, but most optimizers will require one).
            tracker: An optional tracker for logging optimization progress.
            seed: Random seed for reproducibility; set in initialization.
        """
        # Model requirements validation
        assert isinstance(self.model_requirements, tuple), "model_requirements must be a tuple"
        assert all(
            isinstance(m, type) for m in self.model_requirements
        ), "model_requirements must contain only classes/mixins of models."
        assert all(
            isinstance(model, m) for m in self.model_requirements
        ), f"Model {type(model)} not supported by {type(self)}"
        self.model = model

        # Loss function validation
        assert isinstance(loss, BaseLoss), "loss must be an instance of BaseLoss"
        self.loss_func = loss

        tracker = tracker if tracker is not None else DummyTracker()
        assert isinstance(tracker, BaseTracker), "tracker must be an instance of BaseTracker"
        self.tracker = tracker

        self._pbar = None

        # Single active budget (limit, metric, scope), or None. See set_budget().
        self._budget: Optional[tuple[int, str, str]] = None

        if seed is not None:
            from transformers import set_seed
            set_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

    @abstractmethod
    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = None,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:
        """Optimize the trigger to minimize the loss on the given inputs.

        Subclasses only implement the search loop and return an
        ``OptimizerResult``.

        Args:
            templates: Can be a single string or a list of (n_templates) strings.
            initial_trigger: Initial trigger to start optimization from, if used by the optimizer.
            targets: Target outputs for the given inputs, if applicable.

        Returns:
            Optimized trigger.


        Note:
            This method is wrapped by the baseclass to handle common pre-run setup and post-run teardown (done via ``__init_subclass__``). This includes hading logging the config at start, and reseting the model input state, logging the final result, and calling tracker's `finish()` at the end.

            The optimization loop should iterate via :meth:`track_steps`, which
            handles `tqdm` progress bar and enforces any budget upper-bound configured via :meth:`set_budget`.
        """
        ...

    def log_config(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = None,
        targets: Optional[Targets] = None,
    ):
        """
        Logs run metadata, along the full optimizer config, to the tracker.
        Ideally called at the start of optimization (the `optimize_trigger` method).
        """
        _skip = {"model", "loss_func", "tracker"}
        hparams = {
            f"hparam/{k}": v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
            for k, v in self.__dict__.items()
            if k not in _skip
        }

        targets_repr = None
        if targets is not None:
            targets_repr: dict[str, Any] = {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in targets.model_dump().items()
                if v is not None
            }

        metadata = {
            "optimizer": type(self).__name__,
            "model_name": self.model.get_model_name(),
            "loss": repr(self.loss_func),
            "templates": list(templates) if not isinstance(templates, list) else templates,
            "initial_trigger": str(initial_trigger) if initial_trigger is not None else None,
            "targets": targets_repr,
            **hparams,
        }
        self.tracker.log_metadata(metadata)

        # also log this metadata:
        lines = ["\n=== Optimizer Run Config ==="]
        for k, v in metadata.items():
            lines.append(f"  {k}: {v}")
        lines.append("===========================")
        logger.info("\n".join(lines))

    def log(self, loss: Optional[float] = None, trigger_str: Optional[str] = None, **extra) -> None:
        """Log metrics to the tracker.

        When ``loss`` is omitted (e.g. ``self.log(**result.to_dict())`` at the end of a run),
        the kwargs are forwarded directly to the tracker with no per-step enrichment.

        When ``loss`` is provided, automatically enriches with per-step stats:
          - ``loss/*``: loss function component stats via ``loss_func.get_loss_log_dict()``.
          - ``target_model_stats/*``: usage stats for ``self.model``.
          - ``{attr}_stats/*``: usage stats for any other ``BaseModel`` instances found on ``self``
            (that are _not_ ``self.model``).
          - ``total_models_stats/*``: element-wise sum across all model stats.

        Args:
            loss: Per-step loss value. Omit for final/summary logging.
            trigger_str: Current trigger string (omitted from log dict if None).
            **extra: Any additional key-value pairs to include in the log dict.
        """
        if loss is None:
            self.tracker.log(extra)
            return

        log_dict: dict = {"loss": loss}
        if trigger_str is not None:
            log_dict["trigger_str"] = trigger_str
        log_dict.update(extra)

        # Loss function stats
        for k, v in self.loss_func.get_loss_log_dict().items():
            log_dict[f"loss/{k}"] = v

        # Collect distinct model instances (self.model → "target_model_stats")
        seen_ids: set = set()
        model_stats: dict[str, dict] = {}
        for attr, val in self.__dict__.items():
            if not isinstance(val, BaseModel) or id(val) in seen_ids:
                continue
            prefix = "target_model_stats" if attr == "model" else f"{attr}_stats"
            model_stats[prefix] = val.get_usage_stats()
            seen_ids.add(id(val))

        for prefix, stats in model_stats.items():
            for k, v in stats.items():
                log_dict[f"{prefix}/{k}"] = v

        # Total across all distinct models (only when more than one)
        total: dict = {}
        for stats in model_stats.values():
            for k, v in stats.items():
                total[k] = total.get(k, 0) + v
        for k, v in total.items():
            log_dict[f"total_models_stats/{k}"] = v

        self.tracker.log(log_dict)

        if self._pbar is not None:
            desc = f"{loss=:.4f}"
            if trigger_str is not None:
                desc += f" {trigger_str=}"
            self._pbar.set_description(desc)

    #### -- Budget tracking and step iteration utilities -- ####
    def reset_budget(self):
        """Clears any budget set by :meth:`set_budget`."""
        self._budget = None

    def set_budget(self, limit: int, metric: str = "total_tokens", scope: str = "all") -> None:
        """Registers an upper-bound resource budget enforced by :meth:`track_steps`.

        The budget is a ceiling, not a quota: if the optimizer terminates naturally before reaching it, the budget has no effect.

        Common metrics (keys of :meth:`BaseModel.get_usage_stats`):
        - ``"total_flops"``: Estimated FLOPs consumed. Requires ``model.set_flop_counting("manual")`` on any model whose FLOPs should count. Best choice for white-box compute-equalised comparisons.
        - ``"total_tokens"``: Total tokens processed (prompt + generation). Best for black-box models where FLOPs aren't observable but token usage is.

        Args:
            limit: Integer upper bound on the ``metric``.
            metric: The metric the budget is set by. Defaults to the token usage count.
            scope: What models to take the metric against. 
                In optimizers that accomodate multiple models (e.g., proxy models), this may be critical choice.
                ``"all"``, sums the metric across all models found on ``self``. 
                ``"target"`` only considers the primary target model (``self.model``), which is useful if we only care about the target model API token usage.
        """
        assert scope in ("all", "target"), "scope must be 'all' or 'target'"
        assert metric in ("total_flops", "total_tokens"), "currently only 'total_flops' and 'total_tokens' metrics are supported"
        self._budget = (int(limit), metric, scope)

    def _budget_usage(self) -> float:
        if self._budget is None:
            return 0.0
        _, metric, scope = self._budget

        # Budget on target model only:
        if scope == "target":
            return float(self.model.get_usage_stats().get(metric, 0))

        # Budget across all available models:
        seen: set[int] = set()
        total = 0.0
        for val in self.__dict__.values():
            if isinstance(val, BaseModel) and id(val) not in seen:
                seen.add(id(val))
                total += val.get_usage_stats().get(metric, 0)
        return total

    def _budget_exhausted(self) -> bool:
        return self._budget is not None and self._budget_usage() >= self._budget[0]

    def track_steps(self, *args, **kwargs):
        """Iterator for the optimization loop that handles progress bar and budget enforcement.
        
        This supplement the optimziation loop with:
        - a ``tqdm`` progress bar (args/kwargs forwarded) that :meth:`log`
          calls auto-updates with the current loss and trigger string, and
        - early termination when the budget set via :meth:`set_budget` is hit.

        The budget is checked at the top of each step, so overshoot is bounded
        by one step's work. Without a budget set, behaves like plain ``tqdm``.

        Usage::

            for _ in self.track_steps(range(self.num_steps), desc="MyOpt"):
                ...
        """
        self._pbar = tqdm(*args, **kwargs)
        for item in self._pbar:
            if self._budget_exhausted():
                limit, metric, scope = self._budget  # type: ignore[misc]
                logger.info(
                    f"[{type(self).__name__}] budget reached "
                    f"({metric}[{scope}]={self._budget_usage():.3g} >= {limit:.3g}); stopping early."
                )
                self._pbar.close()
                return
            yield item
