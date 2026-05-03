# Runnign a Recipe

[WIP] 

<!-- [[TODO create a whole seprate "running a recipe" section; also we should NOTE there that::
 when reproducing existing methods through \tropt{}'s hosted recipes, we advise verifying the component instantiation (e.g., optimizer parameters; several papers have multiple parameter sets) to ensure accurate reproductio.
 ]] -->


# Composing a Recipe

A recipe script glues together a **Model**, **Loss**, and **Optimizer** into a runnable function. This works in any standalone script — no registration required. See `tropt/recipe_hub/` for examples.
<!-- TODO discuss model compatability: how to check it / take it into account when desining a composing recipe -->

## Minimal Example

```python
from tropt.common import Targets
from tropt.loss import PrefillCELoss
from tropt.model.huggingface.lm import LMHFModel
from tropt.optimizer import OptimizerResult
from tropt.optimizer.gcg_optimizer import GCGOptimizer
from tropt.optimizer.utils.token_initializers import get_printable_random_trigger
from tropt.tracker import LiveLossPlotTracker


def run_myattack(
    model_name: str = "google/gemma-3-270m-it",
    instruction: str = "Do something harmful. {{OPTIMIZED_TRIGGER}}",
    target_response: str = "Sure, here's how:",
) -> OptimizerResult:
    """One-line description. https://arxiv.org/abs/XXXX.XXXXX"""
    model = LMHFModel(model_name=model_name, use_prefix_cache=True)
    loss = PrefillCELoss()
    tracker = LiveLossPlotTracker()
    initial_trigger = get_printable_random_trigger(trigger_len=20, tokenizer=model.tokenizer)

    optimizer = GCGOptimizer(
        model=model,
        loss=loss,
        tracker=tracker,
        num_steps=500,
        n_candidates=512,
    )

    return optimizer.optimize_trigger(
        templates=[instruction],
        targets=Targets(target_response_strs=[target_response]),
        initial_trigger=initial_trigger,
    )
```

## Checklist

1. **Template placeholder** -- use `{{OPTIMIZED_TRIGGER}}` where the trigger should be injected.
2. **Model** -- pick a model class whose mixins satisfy the optimizer's `model_requirements`.
3. **Loss** -- choose from `tropt/loss/`. Use `CombinedLoss` for multi-objective optimization.
4. **Optimizer** -- choose from `tropt/optimizer/`.
5. **Tracker** (optional) -- pass a `BaseTracker` subclass to log per-step metrics. See the Trackers section below.
6. **Return type** -- always `OptimizerResult`, returned by `optimizer.optimize_trigger(...)`. Exposes `best_trigger_str` / `best_trigger_ids`, `best_loss`, per-step `losses` and `trigger_strs`, and `full_prompt`. See `tropt/optimizer/base.py`.

## Trackers

Pass a tracker to the optimizer to log per-step metrics. Pick based on the setting:

- `LiveLossPlotTracker` — inline loss plot, useful in notebooks.
- `WandbTracker` — Weights & Biases, useful for experiments at scale.
- `JSONTracker` — local JSON records, useful for offline analysis.
- `PrintTracker` / `DictTracker` / `DummyTracker` — stdout, in-memory, or no-op.

```python
from tropt.tracker import WandbTracker
optimizer = GCGOptimizer(model=model_obj, loss=PrefillCELoss(), tracker=WandbTracker())
```

See `tropt/tracker/` for the full set.

## Recipe Conventions and Special Cases

Patterns that recur across `tropt/recipe_hub/` and things to notice.

- **Seed everything.** Pass `seed=seed` to the optimizer — `BaseOptimizer.__init__` calls `transformers.set_seed` internally, seeding torch/numpy/random for the whole run. No manual seeding needed.
- **Token blacklist.** Build a `TokenConstraints(...)` once, pass it to the optimizer (`token_constraints=...`) and reuse its `get_blacklist_ids(tokenizer)` when sampling the initial trigger — so the starting point respects the same filters the optimizer enforces.
- **Prefer `bfloat16` over `float32`.** On CUDA `float32` is expensive and rarely worth it for adversarial search; `dtype="bfloat16"` is a safe default.
- **Special cases — losses.** Some losses need extra model wiring. E.g. `AttentionBasedLoss` subclasses need `use_eager_attention=True` on `LMHFModel` since SDPA doesn't expose attentions.
- **Special cases — models.** Thinking-style models (Qwen3, etc.) emit a `<think>...</think>` block before the reply; prepend `"<think>\n\n</think>\n\n"` to the target so prefilling lands on the actual answer.
- **Special cases — optimizers.** Some optimizers need an auxiliary model. Proxy-based optimizers (e.g. `PALOptimizer`, `QCGOptimizer`) take a `proxy_model=`; decoding-based ones (e.g. `BeamSearchOptimizer` in AdvDecoding mode) take a `util_lm=`.

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

## Maximal Example

Let's now gather all of the additional, optional components together to demonstrate an inclusive example — covering the conventions, special cases, and budgeting features from the previous sections: `PALOptimizer` (proxy-guided) under a `CombinedLoss` of `PrefillCELoss` + `AttentionEnhLoss`, with `TokenConstraints`, a seeded random initial trigger, `bfloat16`, FLOP counting, a FLOP budget, and `WandbTracker`.

```python
import math

from tropt.common import SliceKey, Targets
from tropt.loss import AttentionEnhLoss, CombinedLoss, PrefillCELoss
from tropt.model.huggingface.lm import LMHFModel
from tropt.optimizer import OptimizerResult
from tropt.optimizer.pal_optimizer import PALOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.optimizer.utils.token_initializers import get_printable_random_trigger
from tropt.tracker import WandbTracker


def run_myattack_maximal(
    model_name: str = "google/gemma-2-2b-it",
    instruction: str = "Do something harmful. {{OPTIMIZED_TRIGGER}}",
    target_response: str = "Sure, here's how:",
    seed: int = 42,
    flop_budget: float = 3e17,
) -> OptimizerResult:
    """Maximal recipe demonstrating every optional feature."""
    # Attention-based losses need eager attention; bfloat16 for cheap CUDA compute.
    model = LMHFModel(
        model_name=model_name,
        dtype="bfloat16",
        use_eager_attention=True,
        use_prefix_cache=False,
    )
    model.set_flop_counting("manual")  # enables "total_flops" in get_usage_stats()

    # Custom token blacklist — reuse for both the optimizer and the initial trigger.
    token_constraints = TokenConstraints(
        disallow_non_ascii=True,
        disallow_special_tokens=True,
    )

    # CombinedLoss: prefill CE + attention-hijacking on middle layers.
    loss = CombinedLoss(
        [
            PrefillCELoss(),
            AttentionEnhLoss(
                targeted_layers=slice(
                    math.floor(0.1 * model.n_layers),
                    math.ceil(0.9 * model.n_layers),
                ),
                src_slc_name=SliceKey.TRIGGER,
                dst_slc_name=SliceKey.INPUT_AFTER,
            ),
        ],
        weights=[1.0, 100.0],
    )

    tracker = WandbTracker("myattack_maximal", project_name="tropt-demo")

    initial_trigger = get_printable_random_trigger(
        trigger_len=20,
        tokenizer=model.tokenizer,
        token_constraints=token_constraints,
    )

    # Proxy-based optimizer: self-proxy in the whitebox case (same model for proxy).
    optimizer = PALOptimizer(
        model=model,
        proxy_model=model,
        loss=loss,
        tracker=tracker,
        seed=seed,
        num_steps=20_000,  # generous; the FLOP budget is the real stopping criterion
        token_constraints=token_constraints,
    )
    optimizer.set_budget(flop_budget, metric="total_flops")

    return optimizer.optimize_trigger(
        templates=[instruction],
        targets=Targets(target_response_strs=[target_response]),
        initial_trigger=initial_trigger,
    )
```

# Contributing a recipe to TROPT

If you want to contribute the recipe to the package's Recipe Hub (not just use it in your own script), add your function to `tropt/recipe_hub/__init__.py`. Use the [naming convention](#naming-convention) — the function name and dict key must be identical.

```python
# Bare name (not a paper reproduction):
from tropt.recipe_hub.myattack import run_myattack

RECIPES = {
    ...
    "myattack": run_myattack,
}

# Paper reproduction (algorithm + hparams faithful to first-author+year paper):
from tropt.recipe_hub.MyAttack import run_myattack__doe2024

RECIPES = {
    ...
    "myattack__doe2024": run_myattack__doe2024,
}
```




<!-- [[TODO make this more concise, and this should ONLY be part of the contributing secetion:]] -->
## Naming convention

Recipe entry-point function names and the `RECIPES` dict keys in `tropt/recipe_hub/__init__.py` use the same string. The format is:

```
{method}[_{variant}][_{task}][__{paperYYYY}]
```

- `method` — short canonical name (`gcg`, `arca`, `beast`, ...).
- `_{variant}` — algorithmic variant, when one method has several (`gcgp_whitebox`, `gcgp_blackbox`, `gcg_perplexity`).
- `_{task}` — distinct task application of the same method (`advdecoding_jailbreak` vs `advdecoding_retrieval`, `uat_classifier` vs `uat_prompt_injection`).
- `__{paperYYYY}` — the **reproduction tag** (double underscore; first author + year, lowercased, no separator). Reserved for recipes that precisely reproduce a published method.

### When to use the reproduction tag

Use `__{paperYYYY}` *only* if all three hold:

1. The recipe implements the paper's algorithm step-by-step (no different optimizer, no different per-step rule, no missing core ingredient like a buffer or a momentum term).
2. The recipe's defaults match the hyperparameters the paper reports (or directly cites from the paper's reference implementation).
3. If the recipe ports the method to a different setting than the paper's main experiment (e.g. CLIP→causal-LM), the algorithm transfers cleanly and the loss function is the canonical analogue. State the port in the docstring.

If any of these fails — different optimizer, missing scheduling, hparams not from the paper, recipe introduces non-paper knobs as defaults — **omit the tag**. Use the bare `{method}[_{variant}][_{task}]` form instead.

Examples (live in the repo):

| Recipe key | Why this form |
| --- | --- |
| `gcg__zou2023` | Algorithm 1 + paper hparams (B=512, k=256, T=500). |
| `gcg_perplexity` | GCG composed with `TriggerPerplexityLoss`; not in any paper. |
| `gcgp_whitebox__hayase2024` | GCG+ white-box variant from §4.1 of the QCG paper. |
| `prs` | Recipe documents explicit deviations from the paper (schedule, restarts, no judge). |
| `arca_toxic_reverse` | Reproduces the *task* of Jones et al. §4.2.1 but uses GCG instead of ARCA — algorithm differs. |

### Multiple recipes from the same paper

When a single paper introduces multiple variants and the recipe hub exposes them as separate functions, each one carries the same paper tag, with the variant slot disambiguating: `pal__sitawarin2024`, `ral__sitawarin2024`, `gcgp_pal__sitawarin2024`.