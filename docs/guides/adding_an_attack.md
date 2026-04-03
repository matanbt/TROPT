# Adding an Attack

An attack script glues together a **Model**, **Loss**, and **Optimizer** into a runnable function. This works in any standalone script — no registration required. See `tropt/attack_zoo/` for examples.

## Minimal Example

```python
from typing import Optional

from tropt.common import Targets
from tropt.loss import PrefillCELoss
from tropt.model.huggingface.lm import LMHFModel
from tropt.optimizer import OptimizerResult
from tropt.optimizer.gcg_optimizer import GCGOptimizer
from tropt.tracker import BaseTracker

_INITIAL_TRIGGER = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"


def run_myattack(
    model_name: str = "google/gemma-3-270m-it",
    instruction: str = "Do something harmful. {{OPTIMIZED_TRIGGER}}",
    target_response: str = "Sure, here's how:",
    model_obj: Optional[LMHFModel] = None,
    tracker: Optional[BaseTracker] = None,
) -> OptimizerResult:
    """One-line description. https://arxiv.org/abs/XXXX.XXXXX"""
    if model_obj is None:
        model_obj = LMHFModel(model_name=model_name, use_prefix_cache=True)

    optimizer = GCGOptimizer(
        model=model_obj,
        loss=PrefillCELoss(),
        tracker=tracker,
        num_steps=500,
        n_candidates=512,
    )

    return optimizer.optimize_trigger(
        templates=[instruction],
        targets=Targets(target_response_strs=[target_response]),
        initial_trigger=_INITIAL_TRIGGER,
    )
```

## Checklist

1. **Template placeholder** -- use `{{OPTIMIZED_TRIGGER}}` where the trigger should be injected.
2. **Model** -- pick a model class whose mixins satisfy the optimizer's `model_requirements`. Accept an optional `model_obj` parameter so callers can reuse a pre-loaded model.
3. **Loss** -- choose from `tropt/loss/`. Use `CombinedLoss` for multi-objective optimization.
4. **Optimizer** -- choose from `tropt/optimizer/`. Pass `tracker` through for logging.
5. **Return type** -- always `OptimizerResult`, returned by `optimizer.optimize_trigger(...)`.

## Output

`OptimizerResult` (see `tropt/optimizer/base.py`) contains:
- `best_trigger_str` / `best_trigger_ids` -- the optimized trigger
- `best_loss` -- final loss value
- `losses`, `trigger_strs` -- per-step history
- `full_prompt` -- complete prompt(s) with the best trigger substituted in

## Adding to TROPT

If you want to contribute the attack to the package's Attack Zoo (not just use it in your own script), add your function to `tropt/attack_zoo/__init__.py`:

```python
from tropt.attack_zoo.myattack import run_myattack

ATTACK_RECIPES = {
    ...
    "myattack": run_myattack,
}
```
