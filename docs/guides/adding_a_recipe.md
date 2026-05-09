# Composing a Recipe

Going one level of abstraction down from executing Recipe Hub instnaces, the next guide shows how once an build their _own_ custom recipes.

A TROPT recipe is a script that combines TROPT's four core components into a single function: a **Model**, **Loss**, **Optimizer**, and a **Input Setup**.
Importantly, as we discuss below, TROPT recipes must reflect a _valid_ combination of these componenets; for example, some optimziers may require gradient computation, so models that have only black-box API access are automatically incompatible with these.

This guide in effect explains how each recipe in TROPT's Recipe Hub is implemented; so referring to existing recipes in `tropt/recipe_hub/` can provide helpful examples.



[TODO TODONOW make sure that each recipe here runs]

## A Minimal Recipe

We'll start with a minimal recipe that implements the [GCG](https://arxiv.org/abs/2307.15043) LLM jailbreaks against an HuggingFace model.

Our recipe function will accept the target method, a harmful instruction template (w/ a placeholder for the trigger), and a target response; it will then return the optimized trigger.

```python
from tropt.model.huggingface.lm import LMHFModel
from tropt.loss import PrefillCELoss
from tropt.optimizer import GCGOptimizer, OptimizerResult
from tropt.common import Targets

def my_gcg_recipe(
    model_name: str = "google/gemma-3-270m-it",
    instruction: str = "How to pick a lock. {{OPTIMIZED_TRIGGER}}",
    target_response: str = "Sure, here's how:",
) -> :
    """Implements GCG's LLM Jailbreak."""
    # Define model & loss componenets, and wire them with the optimizer
    model = LMHFModel(model_name=model_name, use_prefix_cache=True)
    loss = PrefillCELoss()
    optimizer = GCGOptimizer(
        model=model,
        loss=loss,
        num_steps=500,
        n_candidates=512,
    )

    # Define the input setup (single instruction and its target response)
    templates = [instruction]
    targets = Targets(target_response_strs=[target_response])

    # Run optimization
    result: OptimizerResult = optimizer.optimize_trigger(
        templates=templates,
        targets=targets,
        initial_trigger="! ! ! ! ! ! ! ! ! ! !",
    )

    return result.best_trigger_str  # return the string of the best trigger found
```

A few things we should note from the implementation:
- The model and loss are initialized are registered in the optimizer before the optimization.
    - Any selection of these three componenets from TROPT's respective subpackages (`from tropt.{loss,optimizer,model} import [...]`) works; this is up to compatability limitations (below).
- The input setup enables multiple templates (and, accordingly, targets); we only use a single templte-target pair.
- The templates provided to the optimizer must include the trigger location with a placeholder. 
    - This placeholder string is `{{OPTIMIZED_TRIGGER}}` (defined as a constant under `tropt.common.OPTIMIZED_TRIGGER_PLACEHOLDER`).
- The optimization returns the result in a `OptimizerResult` object, where the best trigger string can be found and returned.


## Enhancing the recipe

The recipe can be further enhanced with TROPT-supported primitives and tools, such as third-party monitoring. We add these to our recipe in the following example:

[TODO make sure our docs support the visualization of this diff with python snippet:]

```python
from tropt.model.huggingface.lm import LMHFModel
from tropt.loss import PrefillCELoss
from tropt.optimizer import GCGOptimizer, OptimizerResult
from tropt.common import Targets

+ from tropt.optimizer.utils.token_initializers import get_printable_random_trigger
+ from tropt.optimizer.utils.token_constraints import TokenConstraints
+ from tropt.tracker import WandbTracker

def my_gcg_recipe(
    model_name: str = "google/gemma-3-270m-it",
    instruction: str = "How to pick a lock. {{OPTIMIZED_TRIGGER}}",
    target_response: str = "Sure, here's how:",
) -> :
    """Implements GCG's LLM Jailbreak."""
    # Define model & loss componenets, and wire them with the optimizer
    model = LMHFModel(model_name=model_name, use_prefix_cache=True)
    loss = PrefillCELoss()

    + # Define a tracker for the optimization
    + tracker = WandbTracker()

    + # Define token constraints on the trigger
    + token_constraints = TokenConstraints()

    optimizer = GCGOptimizer(
        model=model,
        loss=loss,
    +   seed=42,  # define a reproducible seed
    +   tracker=tracker,  # register the tracker
        num_steps=500,
        n_candidates=512,
    +   token_constraints=token_constraints,
    )

    # Define the input setup (single instruction and its target response)
    templates = [instruction]
    targets = Targets(target_response_strs=[target_response])
    + initial_trigger = get_printable_random_trigger(trigger_len=20, tokenizer=model.tokenizer, token_constraints=token_constraints)
    
    # Run optimization
    result: OptimizerResult = optimizer.optimize_trigger(
        templates=templates,
        targets=targets,
    -    initial_trigger="! ! ! ! ! ! ! ! ! ! !",
    +    initial_trigger=initial_trigger,
    )

    return result.best_trigger_str  # return the string of the best trigger found
```

Let's break down the additions:

**Tracker.** Trackers can be attached to the optimizer to track per-step metrics. In the example, we attach a Wandb tracker, that will be fed with the loss per-step, as well as other metrics (e.g., token usage). TROPT support several other trackers such as `LiveLossPlotTracker` (useful for notebook loss plotting), `JSONTracker`(records metrics to a JSON), etc.

**Token blacklist.** Most optimizer accept, like `GCGOptimizer`, a series of constraints on the tokens used to craft the trigger. By default (i.e.,  by initializing `TokenConstraints()`) it will black-list non-ascii and special tokens, to make a printable token.

**Seed.** To enable reproducible optimization run, it is possible to pass a seed to the optimizer; internally we use `transformers.set_seed`, which fixes the seed for torch/numpy/random for the whole run. No manual seeding needed.

**Trigger Inintialization.** Most optimizer also accept an initial trigger. Here, we use a utility funciton of TROPT to initialize a random trigger, sampled from our allowed token constrinats.



## Even further enhacning the recipe

[TODO from here: make it another evolution of the GCG recipe]

To cover additional recipe patterns possible in TROPT, in our next gather all of the additional, optional components together to demonstrate an inclusive example — covering the conventions, special cases, and budgeting features from the previous sections: `PALOptimizer` (proxy-guided) under a `CombinedLoss` of `PrefillCELoss` + `AttentionEnhLoss`, with `TokenConstraints`, a seeded random initial trigger, `bfloat16`, FLOP counting, a FLOP budget, and `WandbTracker`.

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



- **Prefer `bfloat16` over `float32`.** On CUDA `float32` is expensive and rarely worth it for adversarial search; `dtype="bfloat16"` is a safe default.
- **Special cases — losses.** Some losses need extra model wiring. E.g. `AttentionBasedLoss` subclasses need `use_eager_attention=True` on `LMHFModel` since SDPA doesn't expose attentions.
- **Special cases — models.** Thinking-style models (Qwen3, etc.) emit a `<think>...</think>` block before the reply; prepend `"<think>\n\n</think>\n\n"` to the target so prefilling lands on the actual answer.
- **Special cases — optimizers.** Some optimizers need an auxiliary model. Proxy-based optimizers (e.g. `PALOptimizer`, `QCGOptimizer`) take a `proxy_model=`; decoding-based ones (e.g. `BeamSearchOptimizer` in AdvDecoding mode) take a `util_lm=`.

**FLOPS**

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




## Recipe Components Compatability

To enable maximum flexability of recipes, TROPT a-priori enables the combination of any four components, _but_ will dynamically rasie an error throughtout the optimization in the case of incompatable combination of cpompoents.

To ensure your component combination is expected to be valid, you may refer to the API reference of each componenet, where their requriements are specified.
**Since _any component mismatch will be reported by TROPT_, the best way to handle compatability is to try your component selection and see if it fails.**
In practice, in the common case of targeting the permissive HuggingFace models, invalid combination are not so common.

Generally, your choice of model should comply with the optimzier requriements (e.g., if it requires gradients) and with the losses' input requriement (e.g., if it requires logits). And your choice of loss should comply with the targets you provide (e.g., a target response).

In more details, below we provide the full considerations for valid componenet selection, while discussing their implementation details:

**Model.** The model must satisfy the optimizer access requirements of the chosen optimizer. Some optimizers rely on access tokenizer
- These requirements, including tokenizer access and gradient computation, are listed in the class field `YourOptimizer.model_requirements`. TROPT reflects model requirements by the model implementation of certain _mixins_ (e.g., ).
- The optimizer will enforce these requirements w.r.t. the model, upon the optimizer initialization.
- In our example recipe, `GCGOptimizer` will enforce a gradient access of the target model, by looking for its `GradientTokenAccessMixin` mixins; this is indeed the case for HuggingFace's LMs (`LMHFModel`).
- Optimizers documentation include these requrements, so you can find them there.

**Loss.** Loss computation requires certain input/output information of the model, and sometimes also definition of a target. 
Concretely, in the case of `PrefillCELoss`, the loss expects:
- The model output to provide `prefill_response_logits`. This means the logits ontop of the prefilled target response, so the model backend must implement this interface. This is indeed the case for HuggingFace LMs that we use (`LMHFModel`). 
- Importantly, computing the logits on a prefilled target response, and then compute the loss w.r.t. it. This means that such target response must be provided. 
This is indeed the case, as we provide the `Targets` object to the optimizer, and include the field `target_response_strs`, which is used by this loss.
- Some losses, such as LLM-as-a-Judge, as not differentiable; thus if we try and run `GCGOptimzier`, for example, on these losses, we will encounter an error (as it will try to backpropagate through this non-differentaible loss).
- Losses requirements of models and target can be found in the losses documentaiton.

It is also possible to to refer to the [Compatability Matrix](TODO LINK), for a _rough_ list of all the compatible combinations.




