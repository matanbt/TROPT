# Adding an Attack

An attack script glues together a **Model**, **Loss**, and **Optimizer** into a runnable function. This works in any standalone script — no registration required. See `tropt/recipe_hub/` for examples.

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

## Tracking compute and capping runs

For benchmark-style scripts you often want to compare optimizers at equal compute, or stop a run once it spends a given amount of resources. To this end, the package support the follwing features:

**Enable FLOP counting** on the model: The package enable the automatic track on the optimzier work. The flops will appear in `model.get_usage_stats()` as `total_flops` alongside `forward_calls`, `forward_samples`, `total_tokens`, etc. Technically, they are auto-logged on every `self.log()` call inside optimizers.
Note: FLOPs are counted at the model `invoke_from_tokens` / `invoke_from_texts` level only — the cost of optimizer-internal and loss-internal computation (e.g. candidate sampling, sorting, Gumbel draws) is knowingly excluded. For implementation details see the [FLOP counting](adding_a_model.md#flop-counting) section of the model guide and [`tropt/model/flop_counter.py`](../../tropt/model/flop_counter.py).

Usage in the recipe:

```python
model = LMHFModel(model_name="google/gemma-3-270m-it")
model.set_flop_counting("manual")  # adds "total_flops" to get_usage_stats()
```

**Cap a run by resource usage** via `optimizer.set_budget(limit, metric=..., scope=...)`: It is possible to pick a metric (e.g., `total_tokens`, or--when enabled--`total_flops`), that you want to cap. The optimizer will that stop when surpassing the defined resource budget. It's an upper bound, not a quota: if the optimizer terminates naturally under the limit, it's unaffected. `scope="all"` (default) sums across every `BaseModel` on the optimizer (target + any proxies); `scope="target"` uses only `self.model`.

This is useful when we would like to cap the token budget for API calls, or to match the compute used by compared recipes.

Usage:
```python
# Whitebox — cap compute by FLOPs (across target + any proxy LM)
optimizer = GCGOptimizer(model=model_obj, loss=PrefillCELoss(), num_steps=10_000)
optimizer.set_budget(1e17, metric="total_flops")

# Blackbox — cap by target-model tokens (FLOPs aren't observable on API models)
optimizer.set_budget(1_000_000, metric="total_tokens", scope="target")
```

Set `num_steps` generously when budgeting — the budget becomes the real stopping criterion; `num_steps` is a safety ceiling.

## Output

`OptimizerResult` (see `tropt/optimizer/base.py`) contains:
- `best_trigger_str` / `best_trigger_ids` -- the optimized trigger
- `best_loss` -- final loss value
- `losses`, `trigger_strs` -- per-step history
- `full_prompt` -- complete prompt(s) with the best trigger substituted in

## Adding to TROPT

If you want to contribute the attack to the package's Recipe Hub (not just use it in your own script), add your function to `tropt/recipe_hub/__init__.py`:

```python
from tropt.recipe_hub.myattack import run_myattack

RECIPES = {
    ...
    "myattack": run_myattack,
}
```
