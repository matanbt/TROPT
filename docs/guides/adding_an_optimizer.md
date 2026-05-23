# Building a New Optimizer

Going one level of abstraction down from picking an existing optimizer off the shelf as part of a recipe composition, this guide shows how to build your *own* custom optimizer.

A TROPT optimizer is the search algorithm at the heart of the recipe: given a model, a loss, text templates (with a `{{OPTIMIZED_TRIGGER}}` placeholder), and an initial trigger, it repeatedly evaluates and updates the trigger to minimize the loss, returning an {py:class}`~tropt.optimizer.OptimizerResult`.

**Design in a nutshell.** Optimizers are deliberately **self-contained**: each one lives in a single file, owns its full search algorithm, and shares no logic with its siblings — a *Repeat Yourself* philosophy (inspired by [HuggingFace's `transformers` single-file modeling philosophy](https://huggingface.co/blog/transformers-design-philosophy)) that trades a few duplicated lines for readability, hackability, and easy head-to-head comparison. To keep them focused on the algorithm, TROPT pushes everything model-specific and infrastructural — input/template management, batching, tokenization, loss computation, gradient computation — into the model component, exposed via a small set of `compute_*_from_*` methods. Your optimizer just calls them.


This guide effectively explains how every optimizer in {py:mod}`tropt.optimizer` is implemented; browsing the existing optimizers there can provide helpful concrete examples.

> If you would like to *contribute* an optimizer back to TROPT, see [CONTRIBUTING.md](https://github.com/matanbt/TROPT/blob/main/CONTRIBUTING.md). This guide focuses on building optimizers for your own use.


## A Minimal Optimizer

We start with a naive **random-search** optimizer: at each step, sample a batch of fully random candidate triggers, evaluate them under the loss, keep the best one seen so far.

```python
import torch

from tropt.model import LossTokenAccessMixin
from tropt.optimizer import BaseOptimizer, OptimizerResult


class MyOptimizer(BaseOptimizer):
    """Naive random search optimizer."""

    # requires the target model to support token-level loss evaluation
    model_requirements = (LossTokenAccessMixin,)

    def __init__(
        self,
        model, loss, tracker=None, seed=None,
        # optimizer-specific parameters:
        num_steps=500, n_candidates=512,
    ):
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)
        self.num_steps = num_steps
        self.n_candidates = n_candidates

    def optimize_trigger(self, templates, initial_trigger, targets):
        # register model inputs (templates + targets) for repeated evaluation
        self.model.set_inputs_from_tokens(templates, targets)

        # initialize trigger and loss
        best_trigger_ids = self.model.tokenizer.encode(
            initial_trigger, add_special_tokens=False
        )  # (trigger_len,)
        best_loss = float("inf")

        for step in self.track_steps(range(self.num_steps)):
            # sample fully random candidate triggers
            candidates = torch.randint(
                0, self.model.vocab_size,
                size=(self.n_candidates, len(best_trigger_ids)),
                device=self.model.device,
            )

            # evaluate candidates under the loss
            losses = self.model.compute_loss_from_tokens(
                candidates, self.loss_func,
            )  # (n_candidates,)

            # update if improved
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

**Let's break the implementation down, and see what every optimizer must satisfy:**

- **Inherits from {py:class}`~tropt.optimizer.BaseOptimizer`.** Any subclass works regardless of where it lives — no registration step, no entry point.
- **Declares `model_requirements`.** A class-level tuple of mixin classes the model must implement. `BaseOptimizer.__init__` enforces this on construction, so incompatible (model, optimizer) pairs fail at the moment you wire the recipe.
- **Implements `optimize_trigger(templates, initial_trigger, targets)`.** The entry point every optimizer defines; returns an `OptimizerResult` carrying the best trigger found.
- **Setup-then-compute.** Call `self.model.set_inputs_from_tokens(templates, targets)` *once* before the loop; inside the loop, `self.model.compute_loss_from_tokens(candidates, self.loss_func)` reuses that registered state to score any batch of candidate triggers. Most models handle internal batching of large `n_candidates`, so you don't split candidates into mini-batches yourself. The matching `reset_inputs_from_tokens` (plus a final result log and `tracker.finish()`) is called automatically when `optimize_trigger` returns — you don't have to clean up.
- **Iterates via `self.track_steps(...)`.** Wraps `range(...)` to provide a tqdm progress bar, automatic per-step logging, and resource-budget early stopping. More on this in the next section.
- **Logs per step with `self.log(...)`.** Forwards metrics to the attached tracker (Wandb, JSON, live-plot, …) and updates the progress bar's description automatically — no manual `set_description` calls.

To use the optimizer, drop it into any recipe whose model satisfies `LossTokenAccessMixin` (which most HuggingFace LMs do):

```python
from tropt.common import Targets
from tropt.loss import PrefillCELoss
from tropt.model.huggingface.lm import LMHFModel

model = LMHFModel(model_name="google/gemma-3-270m-it")
optimizer = MyOptimizer(model=model, loss=PrefillCELoss(), num_steps=200)
result = optimizer.optimize_trigger(
    templates=["How to pick a lock. {{OPTIMIZED_TRIGGER}}"],
    initial_trigger="! ! ! ! ! !",
    targets=Targets(target_response_strs=["Sure, here's how:"]),
)
print(result.best_trigger_str)
```

(For more on composing optimizers with models and losses into a runnable attack, see [Composing a Recipe](adding_a_recipe.md).)


## Enhancing the Optimizer

The version below layers in the utilities every TROPT optimizer typically uses: a {py:class}`~tropt.optimizer.utils.token_constraints.TokenConstraints` filter on candidates, a `RunningBest` helper to track the best-so-far, type-hinted signatures with sensible defaults, and an opt-in resource budget. Each piece is independent — pick what your search actually needs.

```{code-block} python

from typing import Optional

import torch

from tropt.common import DEFAULT_INIT_TRIGGER, Targets, TextTemplates
from tropt.loss import BaseLoss
from tropt.model import BaseModel, LossTokenAccessMixin
from tropt.optimizer import BaseOptimizer, OptimizerResult
from tropt.optimizer.utils.running_best import RunningBest
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker import BaseTracker


class MyOptimizer(BaseOptimizer):
    """Random search with token-constrained candidates and a running best."""

    model_requirements = (LossTokenAccessMixin,)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        num_steps: int = 500,
        n_candidates: int = 512,
        token_constraints: Optional[TokenConstraints] = None,
    ):
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)
        self.num_steps = num_steps
        self.n_candidates = n_candidates
        self.token_constraints = token_constraints or TokenConstraints()

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:
        self.model.set_inputs_from_tokens(templates, targets)

        # cache the set of allowed token IDs (special/non-ASCII blocked by default)
        allowed_ids = self.token_constraints.allowed_token_ids(self.model.tokenizer)

        trigger_ids = torch.tensor(
            self.model.tokenizer.encode(initial_trigger, add_special_tokens=False),
            device=self.model.device,
        )
        best = RunningBest()

        for _ in self.track_steps(range(self.num_steps)):
            # sample candidates from the allowed-token set only
            idx = torch.randint(
                0, len(allowed_ids),
                size=(self.n_candidates, len(trigger_ids)),
                device=self.model.device,
            )
            candidates = allowed_ids[idx]

            losses = self.model.compute_loss_from_tokens(
                candidate_trigger_ids=candidates, loss_func=self.loss_func,
            )
            best_cand = losses.argmin()
            current_loss = losses[best_cand].item()
            current_trigger_ids = candidates[best_cand]
            current_trigger_str = self.model.tokenizer.decode(current_trigger_ids)

            best.update(
                loss=current_loss,
                trigger_ids=current_trigger_ids,
                trigger_str=current_trigger_str,
            )
            self.log(loss=current_loss, trigger_str=current_trigger_str)

        return best.to_result()
```
(*Highlighted lines* are new or changed compared to the minimal optimizer.)

Breaking down the additions:

**Token constraints.** Most optimizers should not be sampling from the *full* vocabulary — special tokens, partial-UTF8 fragments, non-ASCII junk make terrible trigger tokens. `TokenConstraints()` (defaults block special and non-ASCII) carries the policy; the optimizer pulls the resulting allowed-ID set once and samples from it inside the loop. Custom blocklists are easy to add — see {py:class}`~tropt.optimizer.utils.token_constraints.TokenConstraints`.

**`RunningBest`.** A small helper that maintains the best (trigger, loss) triple seen across the run and converts to an `OptimizerResult` at the end. Hand-rolling this for every optimizer is what made the original codebase brittle — use the helper.

**`self.track_steps` — what it actually does.** Iterating via `self.track_steps(range(self.num_steps))` (instead of bare `range`) is the single chokepoint `BaseOptimizer` uses to provide cross-cutting features to every optimizer without per-optimizer boilerplate:

- **Progress bar.** Creates and registers a `tqdm` bar so that `self.log()` automatically updates its description with the current loss and trigger string. All `tqdm` kwargs (e.g. `desc=...`) are forwarded.
- **Resource budget (upper bound).** If the caller has set a budget via `optimizer.set_budget(limit, metric=...)`, `track_steps` halts as soon as cumulative usage reaches `limit`. `metric` is any key from {py:meth}`~tropt.model.BaseModel.get_usage_stats` — e.g. `"total_flops"`, `"forward_calls"`, `"total_tokens"` — summed across every `BaseModel` attribute on the optimizer (so auxiliary/proxy models are included automatically). The budget is a **ceiling, not a quota**: optimizers that converge earlier stop earlier, and we allow it.

Callers opt in per run:

```python
optimizer = MyOptimizer(model=model, loss=loss)
optimizer.set_budget(1e15, metric="total_flops")  # optional; omit to run unbudgeted
optimizer.optimize_trigger(...)
```

**The `set_inputs_from_tokens` / `reset_inputs_from_tokens` contract.** Always call `set_inputs_from_tokens` (or `set_inputs_from_texts`) at the start of `optimize_trigger`. Cleanup is automatic — `BaseOptimizer` wraps your method to call `reset_inputs_from_*`, log the run config and final result summary, and invoke `tracker.finish()` for you.

**Trigger initialization.** Callers can pass an explicit `initial_trigger` (the `DEFAULT_INIT_TRIGGER` is `"! ! ! ..."`-style). For something smarter, sample from the constrained vocabulary itself via {py:func}`~tropt.optimizer.utils.token_initializers.get_printable_random_trigger`.

The remaining building blocks in [optimizer utilities](../api/optimizer_utils) worth knowing about: `retokenize_filtering` (drop candidates that don't survive a decode → encode round-trip), `TriggerBuffer` (best-K pool instead of a single best), `NFlipScheduler` (control how many positions to mutate per step). Pull them in only when your search actually needs them — they're not boilerplate.


## Going White-Box: Gradient-Guided Search

Up to here our search was purely zeroth-order — we sampled candidates and asked the model to score them. The same architecture handles **first-order** signal just as easily: if the model exposes gradients w.r.t. the trigger tokens, the optimizer can use them to *propose* better candidates instead of sampling blindly. This is the [HotFlip](https://arxiv.org/abs/1712.06751) / [GCG](https://arxiv.org/abs/2307.15043) family.

The change is small: declare a stronger access requirement ({py:class}`~tropt.model.GradientTokenAccessMixin`), and use {py:meth}`~tropt.model.GradientTokenAccessMixin.compute_grad_from_tokens` once per step to bias the candidate distribution toward the top-k token replacements at each position.

```{code-block} python

from tropt.model import GradientTokenAccessMixin, LossTokenAccessMixin


class MyOptimizer(BaseOptimizer):
    """Gradient-guided coordinate search (GCG-style)."""

    model_requirements = (LossTokenAccessMixin, GradientTokenAccessMixin)

    # __init__ unchanged, plus a top-k knob (e.g. top_k: int = 256).

    def optimize_trigger(self, templates, initial_trigger, targets):
        self.model.set_inputs_from_tokens(templates, targets)
        trigger_ids = ...                            # initialize as before
        best = RunningBest()

        for _ in self.track_steps(range(self.num_steps)):
            # 1. Gradient w.r.t. every (position, token) substitution.
            grad = self.model.compute_grad_from_tokens(
                trigger_ids=trigger_ids.unsqueeze(0),
                loss_func=self.loss_func,
            )  # (1, trigger_seq_len, vocab_size)

            # 2. For each position, take the top-k most-promising replacements.
            top_k = (-grad[0]).topk(self.top_k, dim=-1).indices  # (trigger_seq_len, top_k)

            # 3. Sample n_candidates substitutions over the top-k slots.
            candidates = sample_substitutions(trigger_ids, top_k, self.n_candidates)

            losses = self.model.compute_loss_from_tokens(candidates, self.loss_func)
            # ... pick best, update RunningBest, log — exactly as before
```

Structurally nothing else changes — same `BaseOptimizer` subclass, same `optimize_trigger` signature, same `track_steps` / `RunningBest` / `set_inputs_from_tokens` plumbing. The differences are:

- **Access requirement:** the model must also implement `GradientTokenAccessMixin`. `BaseOptimizer` rejects incompatible models (e.g. black-box LiteLLM) at init.
- **Per-step signal:** one `compute_grad_from_tokens` call returns a `(1, trigger_seq_len, vocab_size)` tensor — one value per (position, vocabulary token) telling you how much the loss would change under that substitution. Negate and `topk` to identify promising replacements; different optimizers use this signal differently (see existing implementations for examples).

The same template specializes to AutoPrompt (averaged gradient over multiple samples), MAC (momentum on top of the gradient), GASLITE (multi-coordinate flips), and so on — see {py:class}`~tropt.optimizer.GCGOptimizer`, {py:class}`~tropt.optimizer.AutoPromptOptimizer`, and the rest of {py:mod}`tropt.optimizer` for full implementations.


## Going Black-Box: Text-Level Optimization

So far our optimizers asked the model for tensor-level signals — token-level losses, gradients. But many real-world targets (OpenAI, Anthropic, hosted endpoints behind LiteLLM) only expose a *text* interface: send a string, get a string back. The same `BaseOptimizer` architecture covers this case — switch to the text-level access methods and your optimizer composes with API-only models.

```{code-block} python

from tropt.model import LossTextAccessMixin


class MyOptimizer(BaseOptimizer):
    """Random search over trigger strings, evaluated via the model's text interface."""

    model_requirements = (LossTextAccessMixin,)

    def optimize_trigger(self, templates, initial_trigger, targets):
        # text-level setup (not token-level)
        self.model.set_inputs_from_texts(templates, targets)
        best = RunningBest()
        current_trigger_str = initial_trigger

        for _ in self.track_steps(range(self.num_steps)):
            candidate_strs = mutate(current_trigger_str, n=self.n_candidates)

            # text-level evaluation
            losses = self.model.compute_loss_from_texts(
                trigger_strs=candidate_strs, loss_func=self.loss_func,
            )  # (n_candidates,)

            # ... pick best, update, log
```

Same overall flow, swapped at the access boundary:

- **Access requirement:** `LossTextAccessMixin` instead of `LossTokenAccessMixin`. Models like {py:class}`~tropt.model.LiteLLMModel` only implement the text-level mixin.
- **Setup/compute pair:** `set_inputs_from_texts` + `compute_loss_from_texts`, working in strings rather than token IDs. The optimizer no longer touches the tokenizer.
- **Candidates are strings, not tokens.** Your mutation strategy (random words, character flips, beam search via an auxiliary LM, …) operates in string space. See {py:class}`~tropt.optimizer.RandomSearchOptimizer` and {py:class}`~tropt.optimizer.BeamSearchOptimizer` for real implementations.


## A Few More Patterns

**Multi-template aggregation.** When you pass multiple templates, `compute_loss_from_tokens` averages per-candidate losses across them by default. If your search needs the per-template losses (e.g. for adaptive template sampling), pass `keep_message_dim=True` to get a `(n_templates, n_candidates)` tensor back.

**Auxiliary models.** Surrogate-based optimizers (PAL, BEAST, QCG, AdvDecoding) carry one or more *auxiliary* models on the optimizer — proxy gradients, sampling LMs, etc. Make them constructor arguments and store them as attributes; usage stats and the FLOP budget will roll them up automatically (any `BaseModel` attribute on the optimizer is summed into the budget).

**Splitting a long `optimize_trigger`.** If the method grows past a screenful, delegate logical chunks to private methods (`self._propose_candidates(...)`, `self._update_best(...)`). Don't over-do it though — the top-level `optimize_trigger` should still read as the *algorithm*, not as glue.

**Keep optimizers self-contained.** Resist the urge to share helpers across optimizers. The repo deliberately factored everything that's *not* optimizer-specific into the model and utility layers; what remains *is* the algorithm, and three similar lines in two files beats a fragile shared abstraction. See `DESIGN.md` at the repo root (Component 3: Optimizers) for the full rationale.


## Available Access Levels

An optimizer's `model_requirements` is a tuple of these mixins; the model must implement all of them. Canonical definitions live in {py:mod}`tropt.model`.

| Mixin | What it provides on the model | Typical use |
|---|---|---|
| {py:class}`~tropt.model.LossTokenAccessMixin` | `compute_loss_from_tokens(candidate_trigger_ids, loss_func)` | Zeroth-order: evaluate batches of candidate triggers |
| {py:class}`~tropt.model.GradientTokenAccessMixin` | `compute_grad_from_tokens(trigger_ids, loss_func)` | First-order: gradient w.r.t. trigger tokens (HotFlip / GCG family) |
| {py:class}`~tropt.model.LogitsTokenAccessMixin` | `compute_logits_from_tokens(trigger_ids)` | Direct access to raw logits |
| {py:class}`~tropt.model.GradientEmbedAccessMixin` | `compute_grad_from_embeds(embeds, loss_func)` | Continuous relaxation methods (GBDA, PEZ) |
| {py:class}`~tropt.model.LossTextAccessMixin` | `compute_loss_from_texts(trigger_strs, loss_func)` | Black-box: API-only models (LiteLLM, OpenAI, …) |

`BaseOptimizer.__init__` validates `model_requirements` against the passed model and raises immediately if any mixin is missing. **Do not bypass this check** — it's what guarantees the optimizer only calls methods the model actually supports.


> Want to contribute your optimizer back to the TROPT package? See [CONTRIBUTING.md](https://github.com/matanbt/TROPT/blob/main/CONTRIBUTING.md) for the registration, testing, and (optional) Recipe Hub steps.
