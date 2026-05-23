# Building a New Loss

Going one level of abstraction down from picking an existing loss off the shelf as part of a recipe composition, this guide shows how to build your *own* custom loss.

A TROPT loss defines the optimization objective: given a model's output on a triggered input (and sometimes the input itself), it returns a per-sample scalar that the optimizer minimizes.

**Design in a nutshell.** The loss is agnostic to both the model and the optimizer: it does not call the model — it operates on the model's artifacts — and is _not_ wired into the optimization loop directly. 
Instead, the model component invokes the registered loss as part of its forward pass, and TROPT's resolver ({py:func}`~tropt.loss.resolve_and_compute_loss`) automatically fills in the right model data by inspecting the loss's `__call__` signature. 
This keeps losses small, swappable, and trivial to compose into recipes (see [Composing a Recipe](adding_a_recipe.md)).

> **The one rule.** The names of your `__call__` parameters must match field names on {py:class}`~tropt.common.ModelOutput`, {py:class}`~tropt.common.ModelInput`, or {py:class}`~tropt.common.MessageTargets`. Get the names right and everything connects automatically. The most common fields are listed in the [reference table](#available-parameter-names) at the end of this guide; the canonical definitions live in [`tropt/common.py`](https://github.com/matanbt/TROPT/blob/main/tropt/common.py).

This guide effectively explains how every loss in [`tropt/loss/`](https://github.com/matanbt/TROPT/tree/main/tropt/loss) is implemented; browsing the existing losses there can provide helpful concrete examples.

> If you would like to *contribute* a loss back to TROPT, see [CONTRIBUTING.md](https://github.com/matanbt/TROPT/blob/main/CONTRIBUTING.md). This guide focuses on building losses for your own use.


## A Minimal Loss

We start with a simple loss useful against embedding models: maximizing cosine similarity between an encoder's output embedding and a target embedding vector — e.g., aligning a trigger-augmented text with a target vector.

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

**Let's break the implmenetation down, and see what every loss must satisfy:**

- **Inherits from {py:class}`~tropt.loss.BaseLoss`.** Any subclass of `BaseLoss` works regardless of where it lives — no registration step, no entry point.
- **Parameter names do all the wiring.** `output_embeddings` is a field on {py:class}`~tropt.common.ModelOutput` (populated by encoder models); `target_vectors` is a field on {py:class}`~tropt.common.MessageTargets` (sliced from the {py:class}`~tropt.common.Targets` you pass to {py:meth}`~tropt.optimizer.BaseOptimizer.optimize_trigger`). The resolver matches by name, no other registration needed.
- **Return shape `(bsz,)`.** One per-sample loss, no batch reduction. The optimizer handles aggregation. Note that there is no ned to handle batching internally within the loss; this is the _caller_ duty.
- **Sign convention.** Losses are *minimized*. Negate inside the loss if you want to maximize the underlying quantity.
- **`@dataclass` decoration.** Not strictly required, but lets you add hyperparameters cleanly later (next section). All built-in losses use it.

To use this loss, just drop it into any optimizer that supports embedding-based objectives:

```python
from tropt.common import Targets
from tropt.optimizer import GASLITEOptimizer

optimizer = GASLITEOptimizer(model=encoder_model, loss=MyLoss(), num_steps=50)
result = optimizer.optimize_trigger(
    templates=["This passage is great. {{OPTIMIZED_TRIGGER}}"],
    targets=Targets(target_vectors=target_vec.unsqueeze(0)),  # (1, d_model)
)
```

(For more on composing losses with models and optimizers into a runnable attack, see [Composing a Recipe](adding_a_recipe.md).)


## Enhancing the Loss

The version below adds (i) hyperparameters via dataclass fields, (ii) a `require_*` flag asking the model for extra output, and (iii) inheritance from an existing category base class.

For this we rewrite the loss as **cross-entropy on a prefilled target response** — the canonical [GCG](https://arxiv.org/abs/2307.15043)-style jailbreak loss, which encourages the model to produce an affirmative target string (e.g. `"Sure, here's how:"`). This is what {py:class}`~tropt.loss.PrefillCELoss` does in TROPT.

```{code-block} python

from dataclasses import dataclass
from typing import ClassVar

import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from tropt.loss import PrefillBasedLoss


@dataclass
class MyLoss(PrefillBasedLoss):
    """Maximizes the likelihood of a target response."""

    # Hyperparameters as dataclass fields
    temperature: float = 1.0

    # Ask the model to prefill the target response so we get logits over it.
    # (PrefillBasedLoss already declares this; shown here for emphasis.)
    require_target_prefill: ClassVar[bool] = True

    def __call__(
        self,
        prefill_response_logits: Float[Tensor, "bsz response_seq_len vocab_size"],
        target_response_toks: Int[Tensor, "response_seq_len"],
    ) -> Float[Tensor, "bsz"]:
        logits = prefill_response_logits / self.temperature
        # CE expects (bsz, vocab, seq); broadcast the target across the batch.
        targets = target_response_toks.unsqueeze(0).expand(logits.shape[0], -1)
        nll = F.cross_entropy(logits.transpose(-1, -2), targets, reduction="none")
        return nll.mean(dim=-1)  # (bsz,)
```
(*Highlighted lines* are new or changed compared to the minimal loss.)

Breaking down the additions:

**Hyperparameters as dataclass fields.** Anything you want callers to tweak at construction time — temperatures, margins, layer ranges, mode flags — goes here. Crucially, *per-template* data (target tokens, target vectors, target classes) does **not** belong here; it belongs on {py:class}`~tropt.common.Targets`, and the loss pulls it in by parameter name (`target_response_toks` above). This keeps the loss instance stateless w.r.t. the attack — the optimizer can resample or subsample templates without telling the loss.

**`require_*` flags.** Some losses need the model to do extra work before returning output — append the target response and return logits over it, return attention weights, run real generation, etc. Declare this via a `ClassVar` bool on the loss; the model backend inspects these before its forward pass and turns the right outputs on. The full set:

| Attribute | What the model does when it's `True` |
|---|---|
| `require_target_prefill` | Appends the target tokens and returns `prefill_response_logits` over them |
| `require_hidden_states` | Returns `full_hidden_states` |
| `require_attentions` | Returns `full_attentions` (needs `use_eager_attention=True` on HF LMs) |
| `require_generation` | Performs autoregressive generation, returns `generated_response_strs` |
| `require_first_token_logprobs` | Returns `response_first_token_logprobs` (e.g. for OpenAI-style logprobs APIs) |

**Inheriting from a category base class.** [`tropt/loss/`](https://github.com/matanbt/TROPT/tree/main/tropt/loss) defines abstract bases like {py:class}`~tropt.loss.PrefillBasedLoss`, {py:class}`~tropt.loss.EmbeddingBasedLoss`, {py:class}`~tropt.loss.HiddenStateBasedLoss`, {py:class}`~tropt.loss.AttentionBasedLoss`, {py:class}`~tropt.loss.ClassificationBasedLoss`, and {py:class}`~tropt.loss.TextBasedLoss`. Each sets the right `require_*` flag and pins a typed `__call__` signature for its category — inheriting from the right base saves boilerplate and signals intent. The categorization is a **convention for readability**, not a hard requirement: the resolver dispatches by parameter names, not by base class. If your loss doesn't fit any existing category, inheriting from `BaseLoss` directly and setting the flags yourself is equally valid.


## A Variant: Activation Steering

Up to here we asked the model for logits over a target response. The same pattern works just as well for *any* model artifact — swap the base class, swap the `require_*` flag, and swap the `__call__` parameter names, and you have a loss over a completely different signal.

To demonstrate, we rewrite the loss as **activation steering** ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)): encouraging the model's hidden activations at chosen layers/positions to align with (or away from) a target direction in activation space. This has been used both to suppress refusal in jailbreaks and to probe internal representations. It's what {py:class}`~tropt.loss.SteeringActivationLoss` does in TROPT.

```{code-block} python

from dataclasses import dataclass
from typing import ClassVar

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from tropt.loss import HiddenStateBasedLoss


@dataclass
class MyLoss(HiddenStateBasedLoss):
    """Steers hidden activations along (or away from) a target direction."""

    # Hyperparameters
    targeted_layers: slice = slice(None)
    steer_away: bool = False  # if True, push activations *away* from the direction

    # Ask the model to return all hidden states.
    # (HiddenStateBasedLoss already declares this; shown here for emphasis.)
    require_hidden_states: ClassVar[bool] = True

    def __call__(
        self,
        full_hidden_states: Float[Tensor, "bsz n_layers seq_len d_model"],
        target_directions: Float[Tensor, "d_model"],
    ) -> Float[Tensor, "bsz"]:
        # Score the last token of each sample, averaged over the selected layers.
        h = full_hidden_states[:, self.targeted_layers, -1, :]  # (bsz, n_layers, d_model)
        align = F.cosine_similarity(h, target_directions[None, None, :], dim=-1)
        align = align.mean(dim=-1)  # (bsz,)
        return align if self.steer_away else -align
```

Structurally nothing is new — same `BaseLoss` lineage, same dataclass-fields-as-hyperparameters, same `require_*` declaration, same per-sample return shape. The differences are entirely in *which* names appear:

- **Base class & flag:** {py:class}`~tropt.loss.HiddenStateBasedLoss` (sets `require_hidden_states=True`), replacing `PrefillBasedLoss` / `require_target_prefill`.
- **Model output:** `full_hidden_states` replaces `prefill_response_logits`.
- **Per-template target:** `target_directions` (provided via `Targets(target_directions=...)`) replaces `target_response_toks`.

The same one-knob swap also produces attention-based losses (inherit from `AttentionBasedLoss`, name `full_attentions`) and classifier losses (inherit from `ClassificationBasedLoss`, name `output_class_logits`) — see the existing implementations of {py:class}`~tropt.loss.AttentionEnhLoss` and {py:class}`~tropt.loss.MisclassCELoss` for those variants.


## Going Non-Differentiable: Scoring Triggered Text

So far our losses have operated on **tensors** produced by the target model — embeddings, logits, hidden states. But some losses need to score the **triggered text itself** (the trigger alone, or the full input with the trigger inserted) via an *external* system: another LM, a classifier, a heuristic. These losses are typically non-differentiable, and so only black-box optimizers will accept them — but they unlock objectives the target model can't express on its own.

The version below penalizes non-fluent triggered inputs by measuring their perplexity under an external base LM. It mirrors what {py:class}`~tropt.loss.InputFluencyLoss` and {py:class}`~tropt.loss.ExternalTriggerPerplexityLoss` do in TROPT, and has been used to discourage gibberish suffixes that would be easy to filter at inference time.

```{code-block} python

from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jaxtyping import Float

from tropt.loss import TextBasedLoss


@dataclass
class MyLoss(TextBasedLoss):
    """Penalizes non-fluent triggered inputs via an external base LM's perplexity."""

    # Loss runs an external model — flag explicitly as non-differentiable.
    is_differentiable: ClassVar[bool] = False

    judge_model_name: str = "google/gemma-2-2b"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Internal state — not constructor args
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()
        self._model = AutoModelForCausalLM.from_pretrained(
            self.judge_model_name, dtype=torch.bfloat16
        ).eval().to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.judge_model_name)

    def __call__(
        self,
        input_texts: Annotated[List[str], "bsz"],
    ) -> Float[torch.Tensor, "bsz"]:
        enc = self._tokenizer(
            input_texts, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            logits = self._model(**enc).logits
        nll = torch.nn.functional.cross_entropy(
            logits[:, :-1].transpose(1, 2),  # (bsz, vocab, seq-1)
            enc.input_ids[:, 1:],            # (bsz, seq-1)
            reduction="none",
        )
        mask = enc.attention_mask[:, 1:].float()
        return (nll * mask).sum(-1) / mask.sum(-1)  # mean NLL per sample
```
(*Highlighted lines* are new or changed compared to the previous examples.)

Breaking down the changes:

**Naming `input_texts` (or `input_trigger_strs`).** Both fields live on {py:class}`~tropt.common.ModelInput`. `input_texts` is the entire triggered input string with the optimized trigger inserted (one string per candidate in the batch); `input_trigger_strs` is just the trigger substring. Which one you name as a parameter scopes whether your loss scores the trigger in isolation or in context — scoring `input_texts` penalizes gibberish triggers that *also* disrupt the surrounding instruction's fluency, whereas scoring `input_trigger_strs` only checks the trigger's local quality.

**`is_differentiable = False`.** Once your loss touches an external model, runs generation, or otherwise breaks the autograd graph, mark it explicitly. Differentiable optimizers (e.g. {py:class}`~tropt.optimizer.GCGOptimizer`, which back-props through the target model) check this flag and will refuse to run with a non-differentiable loss. Black-box optimizers (e.g. {py:class}`~tropt.optimizer.RandomSearchOptimizer`, {py:class}`~tropt.optimizer.BeamSearchOptimizer`, {py:class}`~tropt.optimizer.PALOptimizer`) accept either. `TextBasedLoss` already sets this to `False` for you — shown above for visibility.

**Loading external resources in `__post_init__`.** {py:meth}`~tropt.loss.BaseLoss.__post_init__` exists for exactly this; subclasses override it to load auxiliary models, tokenizers, classifiers, or anything else heavy. Use `field(default=None, init=False, repr=False)` for these so they don't appear as constructor args or in the loss's `repr`. Note that even though the loss now carries an *external* model on the instance, it still carries no per-template state — everything the loss needs flows in via the resolver each call.


## Composing Multiple Losses

For multi-objective optimization, wrap your losses in {py:class}`~tropt.loss.CombinedLoss` — it resolves each component independently (via the same resolver) and returns a weighted sum:

```python
from tropt.loss import CombinedLoss

loss = CombinedLoss(
    loss_funcs=[LossA(), LossB()],
    weights=[0.8, 0.2],
)
```

Two constraints to know about:

- `CombinedLoss` cannot be nested inside another `CombinedLoss`.
- The combined `require_*` flags are the OR over components; the combined `is_differentiable` is the AND. So adding one non-differentiable component flips the whole combined loss to non-differentiable.


## Available Parameter Names

The resolver matches your `__call__` parameter names against fields on three classes (see [`tropt/common.py`](https://github.com/matanbt/TROPT/blob/main/tropt/common.py) for the canonical definitions and full list).

**From {py:class}`~tropt.common.ModelOutput`** — produced by the model's forward pass:

| Parameter name | Type | Provided when |
|---|---|---|
| `output_embeddings` | `Float[Tensor, "bsz d_model"]` | Encoder models |
| `full_logits` | `Float[Tensor, "bsz seq_len vocab_size"]` | LMs (full sequence) |
| `prefill_response_logits` | `Float[Tensor, "bsz response_seq_len vocab_size"]` | Loss has `require_target_prefill=True` |
| `full_hidden_states` | `Float[Tensor, "bsz n_layers seq_len d_model"]` | Loss has `require_hidden_states=True` |
| `full_attentions` | `Float[Tensor, "bsz n_layers n_heads seq_len seq_len"]` | Loss has `require_attentions=True` |
| `generated_response_strs` | `List[str]` | Loss has `require_generation=True` |
| `response_first_token_logprobs` | `List[Dict[str, float]]` | Loss has `require_first_token_logprobs=True` |
| `output_class_logits` | `Float[Tensor, "bsz n_classes"]` | Classifier models |

**From {py:class}`~tropt.common.ModelInput`** — assembled per step before the forward pass:

| Parameter name | Type | Description |
|---|---|---|
| `input_texts` | `List[str]` | Full triggered input strings (one per candidate) |
| `input_trigger_strs` | `List[str]` | Just the trigger string (one per candidate) |
| `input_trigger_ids` | `Int[Tensor, "bsz trigger_seq_len"]` | Current trigger token IDs |
| `input_slices` | `Dict[SliceKey, slice]` | Position markers (trigger / before / after / appended / last token) |

**From {py:class}`~tropt.common.MessageTargets`** — per-template optimization targets, sliced from the {py:class}`~tropt.common.Targets` passed to {py:meth}`~tropt.optimizer.BaseOptimizer.optimize_trigger`:

| Parameter name | Type | Description |
|---|---|---|
| `target_response_strs` | `str` | Raw target response text |
| `target_response_toks` | `Int[Tensor, "target_seq_len"]` | Tokenized target response |
| `target_response_logits` | `Float[Tensor, "target_seq_len vocab_size"]` | Reference (e.g. teacher) logits |
| `target_vectors` | `Float[Tensor, "d_model"]` | Target embedding vector |
| `target_directions` | `Float[Tensor, "d_model"]` | Target direction in activation space |
| `target_class_idx` | `int` | Target class index (targeted attacks) |
| `true_class_idx` | `int` | True class index (untargeted attacks) |

Parameters with default values in your signature are treated as optional by the resolver — missing data does not raise. Use this for losses that can operate in multiple modes.

If the data you need isn't exposed by any current field, add a new field to `ModelOutput` / `MessageTargets` / `Targets` in [`tropt/common.py`](https://github.com/matanbt/TROPT/blob/main/tropt/common.py) and populate it from the relevant model backend; the resolver will then route it to any loss that names it.


> Want to contribute your loss back to the TROPT package? See [CONTRIBUTING.md](https://github.com/matanbt/TROPT/blob/main/CONTRIBUTING.md) for the registration and testing steps.
