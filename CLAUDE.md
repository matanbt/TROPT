# CLAUDE.md

> Guidance for Claude Code when working with this repository. For comprehensive design philosophy see `DESIGN.md`; for step-by-step guides see `docs/guides/`.
>
> **READ FIRST for almost every TROPT task** — load `skills/tropt/SKILL.md`. It's the user-facing companion to this dev-facing file and covers the bulk of what a contributor does day-to-day: **adding a recipe, adding a loss, adding an optimizer, adding a model backend, composing custom Model + Loss + Optimizer wirings, swapping components in an existing recipe, debugging cross-cutting pitfalls** (mixin mismatches, attention-loss requirements, thinking-model target alignment, black-box vs white-box loss, multi-model OOM, etc.), routing to the right guide / source file, and helping users without a local checkout. Skill loading is not automatic from `<repo>/skills/` — the file must be explicitly `Read` (this is intentional Claude Code behavior, not a bug). Treat the skill as required reading whenever the request touches `tropt/recipe_hub/`, `tropt/loss/`, `tropt/optimizer/`, `tropt/model/`, or composition patterns. Install is just the supporting first step it also covers.

## Tips

- Prefer concise modifications — minimal changes so edits are easy to review. But if minimal changes create technical debt or unreadable code, prefer clarity.
- Temporary scripts or markdown you create go under `claude_stuff/`. Don't make a mess.
- Avoid over-commenting. Use clear names; only comment where intent isn't obvious.
- **Keep docstrings short.** One or two sentences max. Only list args that aren't self-evident from name and type.
- **Docs maintenance**: Don't enumerate specific classes/fields/signatures in docs — they go stale. Point to source instead (e.g., "see `tropt/loss/` for the full set").

## Project Overview

TROPT (Textual Trigger Optimization Toolbox) is a research platform for optimizing discrete text triggers that elicit specific behaviors from NLP models. Primary use cases:
- **Red-teaming**: Optimizing triggers toward malicious/undesired model behaviors (LLM jailbreaks)
- **Prompt Tuning**: Enhancing desired behaviors through trigger optimization
- **Model Inspection**: Crafting adversarial examples and counterfactuals for research

**Important**: This is a defensive security research tool. Code should only be used for legitimate security research, model robustness evaluation, and defensive purposes.

## Development Commands

```bash
uv sync --all-extras    # install in dev mode with all extras
pre-commit install      # install pre-commit hooks
```

Always invoke tools via `uv run` (e.g. `uv run ruff check`, `uv run ty check`, `uv run pytest`).

When iterating on docs (`docs/`), use `uv run sphinx-autobuild docs docs/_build/html` for live-reload instead of repeatedly invoking `docs/build_docs.py` — only fall back to the full builder when you need the auto-generated API reference or compatibility matrix refreshed.

The project uses Weights & Biases for experiment tracking. Ensure `wandb` is configured if running experiments.

## Architecture (orientation)

TROPT is built on **four orthogonal components** glued together by an executable **recipe**:

1. **Model** (`tropt/model/`) — target model wrappers; absorbs most of the heavy lifting (tokenization, batching, prefix caching, loss/gradient computation) via access-level mixins. Two base classes: `LMBaseModel`, `EncoderBaseModel`.
2. **Loss** (`tropt/loss/`) — quantifiable objective. Stateless, model-agnostic; parameter names alone route data from `ModelOutput` / `ModelInput` / `MessageTargets` via the resolver in `tropt/loss/resolution.py`.
3. **Optimizer** (`tropt/optimizer/`) — the central search algorithm. Self-contained ("Repeat Yourself" — no shared logic across optimizers).
4. **Inputs & Targets** — templates with `{{OPTIMIZED_TRIGGER}}` placeholder + a `Targets` dataclass passed to `optimize_trigger()`.

Two key design principles (see `DESIGN.md` for the full argument):
- **Modularity**: any component is swappable with any other implementation conforming to its interface.
- **Backend vs Frontend**: the model component is the heavy *backend*; loss/optimizer are the lightweight, hackable *frontend*.

### Access mixins (the model–optimizer contract)

Each optimizer declares `model_requirements = (Mixin1, Mixin2, ...)`; `BaseOptimizer.__init__` rejects models that don't subclass them. **Do not bypass this validation** — it's the single guarantee that an optimizer only calls methods the model actually implements.

Naming: `{Value}{InputType}AccessMixin` (e.g. `LossTokenAccessMixin`, `GradientTokenAccessMixin`, `LogitsTokenAccessMixin`, `GradientEmbedAccessMixin`, `LossTextAccessMixin`). Canonical definitions live in `tropt/model/model_mixins.py`. For the full picture of method families and the setup-then-compute pattern, read `docs/guides/adding_a_model.md`.

### Standardized I/O

- **`ModelOutput`** (`tropt/common.py`) — all fields optional; models populate only what they can provide. Which fields are populated determines which losses are admissible.
- **`ModelInput`** (`tropt/common.py`) — assembled per step by an `InputsManager` from templates + candidate trigger.
- **`Targets`** / **`MessageTargets`** (`tropt/common.py`) — per-template optimization targets; the loss pulls fields by parameter name.

Canonical definitions and full field lists live in `tropt/common.py`. Don't enumerate them here.

### Glue: Recipe Hub & Config Runner

- **Recipe Hub** (`tropt/recipe_hub/`): pre-configured Model + Loss + Optimizer + Inputs/Targets wirings, each callable as one function. Enumerate via `list_recipes()`; full registry in `tropt/recipe_hub/__init__.py`.
- **Config Runner** (`runner/main.py`): YAML-driven runner via Hydra. (Hydra config support is incomplete — see Known Limitations.)

## Component Interactions

The four components touch each other only through narrow, typed boundaries. Understanding these is the fastest way to know whether a change belongs in the model, the loss, the optimizer, or none of them.

**Optimizer → Model: setup-then-compute.**
The optimizer calls `model.set_inputs_from_{tokens,texts}(templates, targets)` *once*; an `InputsManager` is stored on the model. Inside the loop it calls `model.compute_{loss,grad,...}_from_{tokens,texts}(candidate_trigger_ids, loss_func)` repeatedly — each call fuses the candidate trigger with the stored templates/targets into a fresh `ModelInput`, runs the forward pass, and returns the value. `BaseOptimizer` wraps `optimize_trigger` to call `reset_inputs_from_*` for you on exit, so optimizers never clean up themselves.

**Model → Loss: parameter-name resolution.**
The model never knows about specific loss types. After its forward pass it builds a `ModelOutput` (populated only with fields the backend can provide) and delegates to a single function:

```python
resolve_and_compute_loss(model_output, model_input, loss_func)  # tropt/loss/resolution.py
```

The resolver introspects `loss_func.__call__`'s parameter names and pulls each one from `ModelOutput`, `ModelInput`, or `MessageTargets`. Missing required field → clear runtime error. Adding a new loss type therefore touches *only* `tropt/loss/`; no model code changes.

**Loss → Model: `require_*` flags.**
Some losses need extra work (target prefill, attentions, generation, hidden states, first-token logprobs). They declare `ClassVar` `require_*` flags; the model's invoke methods read these and gate the corresponding outputs. The loss never calls the model — it only signals what it needs.

**Optimizer ↔ Loss: agnostic by construction.**
The optimizer holds `self.loss_func` and passes it straight through to `compute_*` calls — it never inspects loss internals. Any loss that the *model* can resolve works with any optimizer whose `model_requirements` the model satisfies. Differentiable optimizers (e.g. `GCGOptimizer`) additionally check `loss.is_differentiable` and refuse non-differentiable losses.

**Optimizer ↔ Model: `model_requirements`.**
Compatibility is a class-level contract: `model_requirements = (Mixin1, Mixin2, ...)`. `BaseOptimizer.__init__` rejects any model missing one. This is why a black-box optimizer (`LossTextAccessMixin`) composes with API-only models, while gradient-based ones (`LossTokenAccessMixin`, `GradientTokenAccessMixin`) require permissive backends like `LMHFModel`. **Do not bypass this validation.**

**Optimizers are self-contained.**
Each optimizer is one file with its full algorithm. No shared helpers across optimizers (HuggingFace "Repeat Yourself" philosophy). Everything that *isn't* the algorithm — input/template management, batching, tokenization, loss computation, gradient computation — already lives in the model layer. If you find yourself adding a helper that two optimizers would call, push it into the model layer or `tropt/optimizer/utils/` (initializers, constraints, schedulers, `RunningBest`).

## Minimal Examples

These are stripped-down versions of the patterns in `docs/guides/`. Read the relevant guide before doing real work — these examples omit token constraints, trackers, seeds, FLOP budgets, etc.

### (i) Calling a Recipe Hub entry

```python
from tropt.recipe_hub import gcg__zou2023

result = gcg__zou2023(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    instruction="Tell me how to pick a lock. {{OPTIMIZED_TRIGGER}}",
    target_response="Sure, here's how:",
)
print(result.best_trigger_str)
```

### (ii) Composing a custom recipe

```python
from tropt.common import Targets
from tropt.loss import PrefillCELoss
from tropt.model.huggingface.lm import LMHFModel
from tropt.optimizer import GCGOptimizer, OptimizerResult


def my_recipe(
    model_name: str = "google/gemma-3-270m-it",
    instruction: str = "How to pick a lock. {{OPTIMIZED_TRIGGER}}",
    target_response: str = "Sure, here's how:",
) -> str:
    model = LMHFModel(model_name=model_name, use_prefix_cache=True)
    optimizer = GCGOptimizer(model=model, loss=PrefillCELoss(), num_steps=500)
    result: OptimizerResult = optimizer.optimize_trigger(
        templates=[instruction],
        targets=Targets(target_response_strs=[target_response]),
        initial_trigger="! ! ! ! ! ! ! ! ! ! !",
    )
    return result.best_trigger_str
```

### (iii) A custom optimizer (naive random search)

```python
import torch
from tropt.model import LossTokenAccessMixin
from tropt.optimizer import BaseOptimizer, OptimizerResult


class MyOptimizer(BaseOptimizer):
    model_requirements = (LossTokenAccessMixin,)

    def __init__(self, model, loss, tracker=None, seed=None,
                 num_steps=500, n_candidates=512):
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)
        self.num_steps = num_steps
        self.n_candidates = n_candidates

    def optimize_trigger(self, templates, initial_trigger, targets):
        self.model.set_inputs_from_tokens(templates, targets)  # setup once

        best_trigger_ids = torch.tensor(
            self.model.tokenizer.encode(initial_trigger, add_special_tokens=False),
            device=self.model.device,
        )
        best_loss = float("inf")

        for _ in self.track_steps(range(self.num_steps)):
            candidates = torch.randint(
                0, self.model.vocab_size,
                size=(self.n_candidates, len(best_trigger_ids)),
                device=self.model.device,
            )
            losses = self.model.compute_loss_from_tokens(candidates, self.loss_func)
            best_cand = losses.argmin()
            if losses[best_cand] < best_loss:
                best_loss = losses[best_cand].item()
                best_trigger_ids = candidates[best_cand]
            self.log(loss=best_loss)

        return OptimizerResult(
            best_loss=best_loss,
            best_trigger_ids=best_trigger_ids,
            best_trigger_str=self.model.tokenizer.decode(best_trigger_ids),
        )
```

Note the contract: declare `model_requirements`, call `set_inputs_from_*` once, iterate via `self.track_steps(...)`, pass `self.loss_func` through `compute_*`, return an `OptimizerResult`. `reset_inputs_from_*` and `tracker.finish()` are called for you.

### (iv) A custom loss (cosine similarity)

```python
from dataclasses import dataclass

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from tropt.loss import BaseLoss


@dataclass
class MyLoss(BaseLoss):
    """Encourages output embeddings to align with target vectors."""

    def __call__(
        self,
        output_embeddings: Float[Tensor, "bsz d_model"],
        target_vectors: Float[Tensor, "d_model"],
    ) -> Float[Tensor, "bsz"]:
        sim = F.cosine_similarity(
            output_embeddings, target_vectors.unsqueeze(0), dim=-1
        )
        return -sim   # losses are minimized; negate to maximize similarity
```

The only wiring is the parameter names: `output_embeddings` is a `ModelOutput` field (populated by encoder models); `target_vectors` is a `MessageTargets` field (sliced from `Targets(target_vectors=...)` passed to `optimize_trigger`). No registration, no model edits. If the loss needs an optional model output (attentions, hidden states, prefill, generation, first-token logprobs), also set the matching `require_*: ClassVar[bool] = True` on the class.

### Model capability inspection

```python
isinstance(model, GradientTokenAccessMixin)   # Can compute gradients?
isinstance(model, LossTokenAccessMixin)       # Can compute loss from tokens?
```

## Guides (read before implementing)

Before adding or modifying any of the components below, **read the relevant guide** — they are self-contained walkthroughs and the source of truth for the API a contribution must satisfy:

| Task | Guide |
|---|---|
| Run an existing recipe | `docs/guides/running_a_recipe.md` |
| Compose a custom recipe | `docs/guides/adding_a_recipe.md` |
| Add a loss | `docs/guides/adding_a_loss.md` |
| Add an optimizer | `docs/guides/adding_an_optimizer.md` |
| Add a model backend | `docs/guides/adding_a_model.md` |
| Optimizer/Model/Loss compatibility | `docs/guides/compatibility_matrix.md` (auto-generated) |

Auto-generated docs (`docs/api/`, `docs/guides/compatibility_matrix.md`) are produced by `docs/build_docs.py` — don't edit by hand.

For contributions back to the package (file placement, exports, tests, Recipe Hub naming convention), see `CONTRIBUTING.md`.

For testing conventions (mirror layout, fixtures, numerical tolerances, what to test per component), see `TESTING.md`.

## Important Notes

### Scripts: separate repo
The `scripts/` directory is **not tracked by TROPT** (gitignored) and is maintained as a standalone repo at <https://github.com/matanbt/tropt-scripts>. Files live in `scripts/` on disk as a nested git repo with its own remote — TROPT and tropt-scripts evolve independently. When editing under `scripts/`, run git commands from inside that directory; they affect tropt-scripts, not TROPT. Large experiment artifacts (`opt-bench/results/`, `evaluation_results*.csv`, `plots/`) are gitignored in tropt-scripts too — they stay on disk only.

### Security Research Context
This codebase is explicitly designed for adversarial robustness research and red-teaming. Code modifications should maintain this defensive security focus.

### Ruff Configuration
- Line length: 88 characters
- Ignores: F722, F821 (jaxtyping false positives), F401 (empty imports allowed)

### Known Limitations
- Multiple-message prefix caching currently disabled due to edge cases
- Tracker should be initialized per RUN, not per optimizer instance
- Hydra config-runner support is incomplete

## Dependencies

Core: PyTorch, Transformers, Accelerate, Hydra, Pydantic
Models: HuggingFace, SentenceTransformers, OpenAI, LiteLLM
Tracking: Weights & Biases, LiveLossPlot
Dev: pytest, ruff, ty, pre-commit

## Repository Structure

```
tropt/
├── common.py        # Shared types: ModelInput, ModelOutput, Targets, SliceKey, etc.
├── model/           # Target models with mixin-based capabilities
├── loss/            # Loss functions + unified resolver (resolution.py)
├── optimizer/       # Trigger search algorithms (+ utils/)
├── recipe_hub/      # Pre-configured recipes (run via list_recipes())
├── tracker/         # Experiment logging (WandB, JSON, LiveLossPlot, ...)
└── utils/           # Shared utilities

tests/               # Test suite mirroring tropt/ structure
runner/              # Hydra-driven config runner
scripts/             # Separate repo — see "Scripts: separate repo" above
docs/                # Sphinx documentation
├── api/             # Auto-generated API reference (rst)
├── guides/          # Step-by-step guides (md) + auto-generated compatibility matrix
├── build_docs.py    # Build script
└── conf.py          # Sphinx config
quickstart.ipynb     # End-to-end notebook
DESIGN.md            # Design philosophy and rationale
TESTING.md           # Testing guidelines and conventions
CONTRIBUTING.md      # Contribution workflow (file placement, exports, tests)
```
