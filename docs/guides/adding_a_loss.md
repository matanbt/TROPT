# Adding a New Loss

This guide walks you through implementing a new loss function, whether you create it in your own separate script, or you intend to contibute to the package (for the latter, also see [Adding to TROPT](#adding-to-tropt) section).

> For the *why* behind the design, see [DESIGN.md](../../DESIGN.md) (Pillar 3: Losses). For the full API reference, see the [loss API docs](../api/loss.html).

---

## Background

### How losses work in TROPT

Every loss is a callable that returns a per-sample loss tensor of shape `(bsz,)`. Losses never call the model — the [loss resolution system](../../tropt/loss/resolution.py) automatically wires model data to the loss via **introspection** of `__call__` parameter names.

> **Your `__call__` parameter names must exactly match field names in `ModelOutput`, `ModelInput`, or `MessageTargets`.**

This is the single most important rule. Get the names right and everything connects automatically.

### Available parameter names

The resolver matches against fields in three dataclasses:

**From `ModelOutput`** (model computation results):

| Parameter name | Type | Provided by |
|---|---|---|
| `output_embeddings` | `Float[Tensor, "bsz d_model"]` | Encoder models |
| `full_logits` | `Float[Tensor, "bsz seq_len vocab_size"]` | LMs (full sequence) |
| `prefill_response_logits` | `Float[Tensor, "bsz response_seq_len vocab_size"]` | LMs (response region only, prefilled) |
| `full_hidden_states` | `Float[Tensor, "bsz n_layers seq_len d_model"]` | Models with `require_hidden_states=True` |
| `full_attentions` | `Float[Tensor, "bsz n_layers n_heads seq_len seq_len"]` | Models with `require_attentions=True` |
| `generated_response_strs` | `List[str]` | LMs after generation |

**From `ModelInput`** (input metadata):

| Parameter name | Type | Description |
|---|---|---|
| `input_trigger_ids` | `Int[Tensor, "bsz trigger_seq_len"]` | Current trigger token IDs |
| `input_slices` | `Dict[SliceKey, slice]` | Position markers for input regions |
| `input_texts` | `List[str]` | Full texts with trigger inserted |

**From `MessageTargets`** (optimization targets, per-message):

| Parameter name | Type | Description |
|---|---|---|
| `target_response_toks` | `Int[Tensor, "target_seq_len"]` | Tokenized target response |
| `target_vectors` | `Float[Tensor, "d_model"]` | Target embedding vector |
| `target_directions` | `Float[Tensor, "d_model"]` | Target direction in activation space |
| `target_response_strs` | `str` | Raw target response text |

For the full definitions, see [`ModelOutput`, `ModelInput`, and `MessageTargets` in `tropt/common.py`](../../tropt/common.py). Parameters with default values are ignored by the resolver.

### Loss hierarchy

Losses are organized by the type of model output they consume. Each category has an abstract base class (e.g., `PrefillBasedLoss`, `EmbeddingBasedLoss`, `TextBasedLoss`). Browse [`tropt/loss/`](../../tropt/loss/) for the full set.

Placing your loss under the right base class is a **convention for readability**, not a hard requirement — the resolver dispatches by `__call__` parameter names, not by base class.

### `require_*` class attributes

Some losses need the model to do extra work before returning output — run generation, return attention weights, etc. Declare this via `ClassVar` boolean attributes on your loss:

| Attribute | Effect on model invocation |
|---|---|
| `require_target_prefill` | model appends target tokens and returns `prefill_response_logits` |
| `require_generation` | model performs autoregressive generation and returns `generated_response_strs` |
| `require_hidden_states` | model returns `full_hidden_states` |
| `require_attentions` | model returns `full_attentions` |
| `require_first_token_logprobs` | model returns `response_first_token_logprobs` |

These are defined as `False` by default in `BaseLoss`. Base classes like `PrefillBasedLoss` or `GeneratedResponseBasedLoss` already set the right ones — you only need to override them when creating a new category. The **parameter names in `invoke_*` methods are identical** to these attribute names, so what the loss declares is exactly what the model receives.

---

## Step-by-step

### 1. Choose a base class

Pick the base class matching the model output your loss operates on. Browse [`tropt/loss/`](../../tropt/loss/) to see what's available. If none fit, see [Adding a new loss category](#adding-a-new-loss-category).

### 2. Implement your loss

See the [skeleton](#skeleton) below. The critical rule: **name your `__call__` parameters to match fields in `ModelOutput`, `ModelInput`, or `MessageTargets`**.

### 3. Use it

```python
loss = MyLoss()
optimizer = SomeOptimizer(model=model, loss=loss)
result = optimizer.optimize_trigger(templates=..., targets=...)
```

No registration is needed — the [loss resolution system](../../tropt/loss/resolution.py) discovers parameters via introspection, so any `BaseLoss` subclass works out of the box regardless of where it's defined.

---

## Skeleton

Minimal boilerplate — copy and fill in:

```python
from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from tropt.loss import BaseLoss


@dataclass
class MyLoss(BaseLoss):
    # Hyperparameters as dataclass fields with defaults
    # my_param: float = 1.0

    # Override require_* flags as needed (all False by default):
    #   require_target_prefill  — model appends target tokens, returns prefill_response_logits
    #   require_generation      — model runs autoregressive generation
    #   require_hidden_states   — model returns full_hidden_states
    #   require_attentions      — model returns full_attentions

    def __call__(
        self,
        # Name parameters to match fields in ModelOutput, ModelInput, or MessageTargets.
        # The resolver will automatically wire the correct data.
        # See tropt/common.py for available field names.
        ...
    ) -> Float[Tensor, "bsz"]:
        # Compute and return per-sample loss of shape (bsz,).
        # Losses are minimized — negate if you want to maximize something.
        ...
```

> **Note:** The existing abstract base classes (e.g., `PrefillBasedLoss`, `EmbeddingBasedLoss`) are provided as a **convention for readability** — they set the right `require_*` flags and define a typed `__call__` signature. Inheriting directly from `BaseLoss` and setting the flags yourself is equally valid.

As a concrete example, [`SimilarityLoss`](../../tropt/loss/losses.py) is a good reference — it takes `output_embeddings` and `target_vectors` as `__call__` parameters (matching `ModelOutput` and `MessageTargets` fields), and returns negated cosine similarity as shape `(bsz,)`.

---

## Using CombinedLoss

`CombinedLoss` wraps multiple losses with weights for multi-objective optimization:

```python
from tropt.loss import CombinedLoss

loss = CombinedLoss(
    loss_funcs=[LossA(), LossB()],
    weights=[0.8, 0.2],
)
```

Each component is resolved independently. `CombinedLoss` cannot be nested.

---

## Adding a new loss category

If no existing base class fits, create one inheriting from `BaseLoss` with an abstract `__call__`. If the model output field you need doesn't exist in `ModelOutput`, add it in [`tropt/common.py`](../../tropt/common.py) and populate it in the relevant model. Same for new target types in `MessageTargets` / `Targets`.

---

## Checklist

1. **Naming** — `__call__` parameter names match fields in `ModelOutput`, `ModelInput`, or `MessageTargets`.
2. **Return shape** — `(bsz,)`.
3. **Sign convention** — Losses are *minimized*. Negate if maximizing.
4. **Test** — Output shape, known input/output pairs, edge cases. See `tests/loss/`.

---

## Adding to TROPT

If you want to contribute the loss to the package (not just use it in your own script):

1. **Register** — Export from [`tropt/loss/__init__.py`](../../tropt/loss/__init__.py).
2. **Test** — Add tests under `tests/loss/`.
