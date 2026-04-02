# Adding a New Optimizer

This guide walks you through adding a new optimizer to TROPT. Optimizers implement the search algorithm that finds a text trigger minimizing a given loss.

> For the *why* behind the design, see [DESIGN.md](../../DESIGN.md) (Pillar 4: Optimizers). For the full API reference, see the [optimizer API docs](../api/optimizer.html).

---

## Background

### What an optimizer does

An optimizer takes a model, a loss function, text templates (with a `{{OPTIMIZED_TRIGGER}}` placeholder), and an initial trigger. It repeatedly evaluates and updates the trigger to minimize the loss, returning an `OptimizerResult`.

### What you don't need to implement

TROPT's three supporting pillars handle the heavy lifting:

- **Input management** — handled by `set_inputs_from_tokens()` / `set_inputs_from_texts()`
- **Loss computation** — handled by `compute_loss_from_tokens()` / `compute_loss_from_texts()`
- **Gradient computation** — handled by `compute_grad_from_tokens()`

Your optimizer calls these model methods and decides what to do with the results.

### Access levels

Your optimizer declares what model access it needs via `model_requirements`, validated at init:

| Mixin | What it provides | Typical use |
|---|---|---|
| `LossTokenAccessMixin` | `compute_loss_from_tokens(candidate_ids, loss)` | Evaluate candidate triggers |
| `GradientTokenAccessMixin` | `compute_grad_from_tokens(trigger_ids, loss)` | Get gradient w.r.t. token inputs |
| `LogitsTokenAccessMixin` | `compute_logits_from_tokens(trigger_ids)` | Get raw logits |
| `LossTextAccessMixin` | `compute_loss_from_texts(texts, loss)` | Black-box loss evaluation |
| `GradientEmbedAccessMixin` | `compute_grad_from_embeds(embeds, loss)` | Gradient w.r.t. continuous embeddings |

Reading a few existing optimizers in `tropt/optimizer/` before writing your own is recommended.

---

## Step-by-step

### 1. Define your class

Inherit from `BaseOptimizer` and declare `model_requirements` — a tuple of mixin classes your optimizer needs (see [Access levels](#access-levels)). Incompatible model/optimizer combinations fail early with a clear error.

### 2. Implement `__init__`

Call `super().__init__(model, loss, tracker, seed)` and store your hyperparameters.

### 3. Implement `optimize_trigger`

Set up model inputs, run your optimization loop, clean up model state, return an `OptimizerResult`. See the [skeleton](#skeleton) below.

### 4. Register

Export from [`tropt/optimizer/__init__.py`](../../tropt/optimizer/__init__.py).

---

## Common patterns

### Evaluating candidates

```python
losses = self.model.compute_loss_from_tokens(
    candidate_trigger_ids=candidate_ids,
    loss_func=self.loss_func,
)  # (n_candidates,)
best_idx = losses.argmin()
```

Most models handle batching internally, so you don't need to split candidates into batches yourself.

### Computing gradients

```python
grad = self.model.compute_grad_from_tokens(
    trigger_ids=trigger_ids,  # (1, trigger_seq_len)
    loss_func=self.loss_func,
)  # (1, trigger_seq_len, vocab_size)
```

The gradient has shape `(1, trigger_seq_len, vocab_size)` — one value per position per vocabulary token. It approximates how much the loss would change if you substituted each token. Negating and selecting high-value entries identifies promising substitutions. Different optimizers use this signal differently — see existing implementations for examples.

### Black-box evaluation (text-level)

For optimizers working with black-box models:

```python
self.model.set_inputs_from_texts(templates, targets)
losses = self.model.compute_loss_from_texts(
    trigger_strs=["candidate 1", "candidate 2"],
    loss_func=self.loss_func,
)  # (n_candidates,)
self.model.reset_inputs_from_texts()
```

---

## Skeleton

Minimal boilerplate — copy and fill in:

```python
from typing import Optional

from tropt.common import DEFAULT_INIT_TRIGGER, Targets, TextTemplates
from tropt.loss import BaseLoss
from tropt.model import BaseModel, LossTokenAccessMixin  # import the mixins you need
from tropt.optimizer.base import BaseOptimizer, OptimizerResult
from tropt.optimizer.utils.running_best import RunningBest
from tropt.tracker import BaseTracker


class MyOptimizer(BaseOptimizer):
    # Declare the model access level your optimizer requires
    model_requirements = (LossTokenAccessMixin,)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        num_steps: int = 500,
        # ... other hyperparameters ...
    ):
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)
        self.num_steps = num_steps
        # ... store other hyperparameters ...

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:

        # 1. Setup: prepare model inputs
        #    Choose the flow matching your access level:
        #    self.model.set_inputs_from_{tokens|texts}(templates, targets)
        self.model.set_inputs_from_tokens(templates, targets)
        best = RunningBest()

        # 2. Optimization loop — register_tqdm wires self.log() into the progress bar
        for _ in self.register_tqdm(range(self.num_steps)):
            # Use compute_* methods matching your model_requirements:
            #   compute_{loss|grad|logits}_from_{tokens|texts}(...)
            losses = self.model.compute_loss_from_tokens(
                candidate_trigger_ids=candidate_ids, loss_func=self.loss_func,
            )
            # ... generate candidates, evaluate, update trigger ...

            # self.log() enriches automatically with loss stats and model usage
            self.log(loss=current_loss, trigger_str=trigger_str)
            best.update(loss=current_loss, trigger_ids=trigger_ids, trigger_str=trigger_str)

        # Saves the best trigger found
        return best.to_result()
```

---

## Design notes

### Keep optimizers self-contained

Following our [design principles](../../DESIGN.md), optimizers should be self-contained. The repo has already factored out everything that's *not* optimizer-specific. Duplicating a few lines across optimizers is better than a fragile shared abstraction.

### Shared utilities

Optimizers *should* use the utilities in `tropt/optimizer/utils/`:
- `TokenConstraints` — token blacklisting (special tokens, non-ASCII, etc.)
- `retokenize_filtering` — filters candidates that don't survive a decode/encode roundtrip
- `TriggerBuffer` — maintains a pool of the best triggers found so far
- `NFlipScheduler` — controls how many positions to flip over optimization steps
- `get_printable_random_trigger` — random printable ASCII trigger initialization


### The `set_inputs_from_tokens` / `reset_inputs_from_tokens` contract

Always call `set_inputs_from_tokens` (or `set_inputs_from_texts`) at the start of `optimize_trigger`. Cleanup (`reset_inputs_from_*`) is called automatically by `BaseOptimizer` after your method returns — you don't need to call it yourself. The same wrapper also calls `log_config`, logs the final result, and calls `tracker.finish()`.

### Multi-template aggregation

`compute_loss_from_tokens` averages loss across templates by default. Pass `keep_message_dim=True` for per-template losses of shape `(n_templates, n_candidates)`.

### Other pointers

- If the `optimize_trigger` implementation is too long, it often makes sense to delegate some logical chuncks to private method of the optimizer class (eg., `self._init_buffer()`). It is preferable, however, to not _overuse_ these, to keep the `optimize_trigger` flow readable and informative.

---

## Checklist

1. **`model_requirements`** — Declare the exact mixins your optimizer calls.
2. **Cleanup** — `reset_inputs_from_*`, final logging, and `tracker.finish()` are handled automatically by `BaseOptimizer`; no need to call them in your implementation.
3. **Register** — Export from `tropt/optimizer/__init__.py`.
4. **Test** — Requirements validation, basic optimization, optimizer-specific features. See `tests/optimizer/`.
5. **Attack zoo entry** (optional) — Add a recipe in `tropt/attack_zoo/` for published attacks.
