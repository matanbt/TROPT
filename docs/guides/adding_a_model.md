# Adding a New Model

This guide walks you through wrapping a new model backend in TROPT. Pick the section that matches your situation:

- **[Text-access (black-box) model](#adding-a-text-access-black-box-model)** — API-only models where you can query with text and get text/embeddings back. No internal access. This is the most common case. Examples: `EncoderGeminiModel`, `LiteLLMModel`.
- **[Token-access (grey/white-box) model](#adding-a-token-access-greywhite-box-model)** — Backends that expose embedding-level input (you can feed raw embeddings and get logits/gradients). You implement the full compute loop.
- **[HuggingFace model](#adding-a-huggingface-model)** — Any HF-backed model. `_HuggingFaceModelMixins` provides the compute loop; you fill in model-specific parts. Examples: `LMHFModel`, `EncoderHFModel`.

> This guide is self-contained — you can follow it step by step without reading anything else. If you want to understand the *why* behind the design, see [DESIGN.md](../../DESIGN.md) (especially "Pillar 1: Target Model"). For full API reference, see the [models API docs](../api/models.html).

---

## Background

### The Three Method Families

Every model is composed of methods from three families. Understanding these helps you know exactly what you need to implement:

1. **Invoke methods** (`invoke_from_texts`, `invoke_from_tokens`) — stateless forward passes. These are the core model-specific logic: take input, return `ModelOutput`. Every model implements `invoke_from_texts`; only models with embedding-level access implement `invoke_from_tokens`.

2. **Input methods** (`set_inputs_from_tokens`, `set_inputs_from_texts`) — store an `InputsManager` for use during optimization. Default implementations exist for both flows, so most models don't need custom logic here. You only customize this when your backend has special input handling (e.g., HuggingFace's embedding-level prefix caching).

3. **Compute methods** (`compute_loss_from_tokens`, `compute_grad_from_tokens`, etc.) — the methods optimizers actually call. These use the stored inputs (from family 2) and the invoke methods (from family 1) internally. For text-access models, `LossTextAccessMixin` provides `compute_loss_from_texts` for free. For HuggingFace models, `_HuggingFaceModelMixins` provides all token-based compute methods. For other token-access backends, you implement these yourself.

**In practice**: for most new models, you implement `invoke_from_texts` and get everything else for free. Token-access models additionally implement `invoke_from_tokens` and `set_inputs_from_tokens`. The compute methods are only hand-written for non-HF token-access backends.

### Base Classes

Every model inherits from one of two base classes in [`tropt/models/model_base.py`](../../tropt/models/model_base.py):

| Base class | Use for | Inference method you implement |
|---|---|---|
| `LMBaseModel` | Language models (text generation) | `invoke_from_texts(input_texts, ...)` |
| `EncoderBaseModel` | Embedding / encoder models | `invoke_from_texts(input_texts, ...)` and `d_model` property |

Each base class defines `__call__` which delegates to its inference method (`invoke_from_texts`). You implement the inference method; `__call__` is already wired up.

Throughout this guide, we use **"inference method"** to refer to `invoke_from_texts` — the method your base class requires.

### Mixins (Access Levels)

Mixins declare what *type of access* the model exposes. Optimizers check these at init via `model_requirements` to verify compatibility — this is how TROPT enforces that an optimizer only calls methods the model actually supports.

**Naming convention**: `{Value}{InputType}AccessMixin`
- *Value*: what can be computed (`Loss`, `Gradient`, `Logits`)
- *InputType*: what goes in (`Token` for white/grey-box, `Text` for black-box)

**Token-access mixins** — require embedding-level access:

| Mixin | Access level | Methods you implement |
|---|---|---|
| `LossTokenAccessMixin` | grey-box | `compute_loss_from_tokens` |
| `GradientTokenAccessMixin` | white-box | `compute_grad_from_tokens` |
| `LogitsTokenAccessMixin` | white-box | `compute_logits_from_tokens` |
| `GradientEmbedAccessMixin` | white-box | `compute_grad_from_embeds` |

All token-access mixins inherit from `TokenAccessMixin`, which requires:
- **`tokenizer`** property — HuggingFace `PreTrainedTokenizer` or [`BaseTokenizer`](../../tropt/models/model_base.py) subclass.
- **`set_inputs_from_tokens(templates, targets)`** — builds and stores an `InputsManager` for the optimization run.

You implement `tokenizer` and `set_inputs_from_tokens` **once**, regardless of how many token mixins you include.

**Text-access mixin** — black-box text I/O:

| Mixin | Access level | Methods you implement |
|---|---|---|
| `LossTextAccessMixin` | black-box | *nothing* — fully implemented; it calls your `__call__` internally |

`LossTextAccessMixin` provides `compute_loss_from_texts` and `set_inputs_from_texts` out of the box. It works by calling `self(input_texts)` and passing the resulting `ModelOutput` through the [unified loss resolution system](../../tropt/loss/resolution.py).

A model can include both token-access and text-access mixins — see `LMHFModel` and `EncoderHFModel` for examples.

### ModelOutput

[`ModelOutput`](../../tropt/common.py) is a dataclass that standardizes what your model returns. All fields are optional — populate only the ones your backend can provide:

```python
@dataclass
class ModelOutput:
    output_embeddings: ...   # Encoder models
    output_logits: ...       # LMs (prefill logits)
    response_logits: ...     # LMs (response-region logits)
    output_hidden_states: ...
    output_attentions: ...
    generated_response_strs: ...  # LMs (generated text)
    generated_response_ids: ...
    generated_response_logits: ...
    full_template_strs: ...
    full_template_ids: ...
```

The fields you populate determine which loss types are compatible with your model. For example, `output_embeddings` enables `EmbeddingBasedLoss` (e.g., `SimilarityLoss`), while `generated_response_strs` enables `TextBasedLoss` (e.g., `ResponseLMScoreLoss`). The [loss resolution system](../../tropt/loss/resolution.py) validates this at runtime and raises clear errors if a required field is missing.

---

## Adding a Text-Access (Black-box) Model

Use this path when you can only query the model with text and receive text or embeddings back — no access to logits, gradients, or internal representations.

Your class will inherit from:
- A **base class**: `EncoderBaseModel` (for embedding models) or `LMBaseModel` (for language models)
- The **`LossTextAccessMixin`** mixin — which provides `compute_loss_from_texts` and `set_inputs_from_texts` for free

**Existing examples**: [`EncoderGeminiModel`](../../tropt/models/google/encoder.py), [`LiteLLMModel`](../../tropt/models/litellm_proxy/lm.py).

### What to implement

You implement a single method — the inference method required by your base class. Here's an LM example:

```python
def invoke_from_texts(self, input_texts: List[str], **kwargs) -> ModelOutput:
    responses = self._client.complete(input_texts)  # your backend call

    self._update_usage_stats(
        tokens=...,  # extract from backend response
        forward_calls=1,
        forward_samples=len(input_texts),
    )

    return ModelOutput(generated_response_strs=responses)
```

Key rules:
- Input is always `List[str]`. Output is always a `ModelOutput` — the fields you populate determine which loss types are compatible (see [ModelOutput](#modeloutput)).
- **Call `_update_usage_stats` (or `_update_invoke_stats` for models that support FLOP counting) at the same call site as the backend call** — not from higher-level wrappers. This avoids double-counting.

Everything else — `set_inputs_from_texts`, `compute_loss_from_texts` — is provided by `LossTextAccessMixin`.

### Minimal class skeleton

```python
from tropt.common import ModelOutput
from tropt.models import LMBaseModel, LossTextAccessMixin


class MyLMModel(LMBaseModel, LossTextAccessMixin):

    def __init__(self, model_name, api_key=None):
        self._client = ...   # initialize your API client
        self._model_name = model_name

    def invoke_from_texts(self, input_texts: List[str], **kwargs):
        responses = self._client.complete(input_texts)
        self._update_usage_stats(
            tokens=...,
            forward_calls=1,
            forward_samples=len(input_texts),
        )
        return ModelOutput(generated_response_strs=responses)
```

For complete working examples, see [`LiteLLMModel`](../../tropt/models/litellm_proxy/lm.py) (LM) or [`EncoderGeminiModel`](../../tropt/models/google/encoder.py) (encoder).

### File placement

Create a directory under `tropt/models/` for your backend:

```
tropt/models/
├── huggingface/
├── openai/
├── google/
├── litellm_proxy/
└── my_backend/          # your new backend
    ├── __init__.py
    └── encoder.py       # or lm.py
```

Then export your class from [`tropt/models/__init__.py`](../../tropt/models/__init__.py).

---

## Adding a Token-Access (Grey/White-box) Model

Use this when your backend supports embedding-level input — you can pass raw input embeddings (not just text) and receive model outputs like logits or gradients. This is the path for non-HuggingFace backends with internal access.

> If your model is HuggingFace-based, skip to [Adding a HuggingFace Model](#adding-a-huggingface-model) — the `_HuggingFaceModelMixins` class already provides the compute loop.

### The InputsManager

The `InputsManager` pre-processes text templates once (splitting at the `{{OPTIMIZED_TRIGGER}}` placeholder, tokenizing, embedding) and then efficiently inserts candidate triggers at each optimization step via `get_triggered_inputs(trigger_ids, chosen_template_idx)`, which returns a [`ModelInput`](../../tropt/common.py) dataclass.

The default [`DefaultTokenInputManager`](../../tropt/models/inputs_manager.py) works with any tokenizer supporting the `BaseTokenizer` interface — it decodes trigger IDs to strings and reconstructs full texts. The HuggingFace backend uses [`_HFTokenInputManager`](../../tropt/models/huggingface/base.py), which overrides this with embedding-level input construction, attention masks, prefix caching, and position slicing.

### The setup-then-compute pattern

Token-access follows a two-step pattern that separates *setup* from *computation*:

1. **`set_inputs_from_tokens(templates, targets)`** — called by the optimizer **once** before `optimize_trigger()` begins. Tokenizes the templates, splits them at the `{{OPTIMIZED_TRIGGER}}` placeholder, and stores an `InputsManager` on the model.
2. **`compute_{value}_from_tokens(candidate_trigger_ids, ...)`** — called **repeatedly** at each optimization step. Uses the stored manager to assemble full inputs for each candidate trigger, runs the model, and returns the result.

This split matters because templates are fixed for an entire run. Pre-processing them once avoids redundant tokenization and embedding work at every step.

Cleanup is handled by `reset_inputs_from_tokens()` (provided by `TokenAccessMixin`), which the optimizer calls at the end of `optimize_trigger()`. You don't need to implement or call it yourself.

### What to implement

**1. Inference method** — Same as for text-access models (see [above](#what-to-implement)).

**2. `tokenizer` property** — Required by `TokenAccessMixin`. Must be a HuggingFace `PreTrainedTokenizer` or a [`BaseTokenizer`](../../tropt/models/model_base.py) subclass. The optimizer uses it to encode/decode triggers.

```python
@property
def tokenizer(self):
    return self._tokenizer
```

If your backend doesn't use a HuggingFace tokenizer, implement the `BaseTokenizer` interface — see [`OpenAITokenizer`](../../tropt/models/openai/encoder.py) for an example wrapping `tiktoken`.

**3. `set_inputs_from_tokens`** — Tokenize the templates and construct your `InputsManager`. Store it via `self._token_input_manager`. In most cases you can use the `DefaultTokenInputManager`, which works with any `BaseTokenizer`:

```python
def set_inputs_from_tokens(self, templates: List[str], targets: Targets = None) -> None:
    tok_ids = self.tokenizer(templates, add_special_tokens=True)["input_ids"]
    self._token_input_manager = DefaultTokenInputManager(
        tok_ids=tok_ids,
        tokenizer=self.tokenizer,
        targets=targets,
        optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER,
    )
```

Only subclass or replace the input manager if your backend needs custom input construction (e.g., embedding-level assembly — see HuggingFace's `_HFTokenInputManager`).

**4. `compute_{value}_from_tokens`** — Implement one method per token-access mixin you include. Each compute method should use your `invoke_from_tokens` internally to run the forward pass. The structure is the same for all: loop over templates, call `get_triggered_inputs` to get a `ModelInput`, invoke the model, and return the result.

An important convention: **loss is computed per-template, not across templates**. Each template may have its own target, so we never mix templates in a single loss call. The per-template losses are aggregated (averaged) afterward.

`compute_loss_from_tokens` (required by `LossTokenAccessMixin`) is the most common:

```python
@torch.no_grad()
def compute_loss_from_tokens(
    self,
    candidate_trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"],
    loss_func: BaseLoss,
    keep_message_dim: bool = False,
) -> Float[Tensor, "n_candidates"]:
    input_manager = self._token_input_manager
    n_templates = input_manager.n_templates
    all_losses = [[] for _ in range(n_templates)]

    for template_idx in range(n_templates):
        for batch in batched(candidate_trigger_ids):  # implement your own batching
            model_input = input_manager.get_triggered_inputs(
                trigger_ids=batch, chosen_template_idx=template_idx
            )
            model_output = ...  # run your backend on model_input
            self._update_usage_stats(forward_calls=1, forward_samples=len(batch), tokens=...)
            loss = resolve_and_compute_loss(model_output, model_input, loss_func)
            all_losses[template_idx].append(loss)

    losses = torch.stack([torch.cat(l) for l in all_losses])  # (n_templates, n_candidates)
    return losses if keep_message_dim else losses.mean(dim=0)
```

The key line is `resolve_and_compute_loss(model_output, model_input, loss_func)` — this is the [unified loss resolution](../../tropt/loss/resolution.py) function that dispatches to the correct loss computation based on loss type. You don't implement loss logic yourself; you just provide the data via `ModelOutput` and `ModelInput`.

Note that it is `compute_loss_*` duty to wrap/not wrap computations with `no_grad()`, for efficeincy. We assume that all calls to `compute_loss_*` requires not grad, as there is a dedicated method fo grad computation.

`compute_grad_from_tokens` (required by `GradientTokenAccessMixin`) has the same per-template loop structure, but returns the **gradient of the loss w.r.t. the token input** instead of the loss itself. The returned tensor has shape `(n_candidates, trigger_seq_len, vocab_size)` — one gradient value per token position per vocabulary entry, telling the optimizer which substitutions would most reduce the loss. How you compute this gradient is up to your backend.

### FLOP counting

TROPT supports optional FLOP counting via invoke-level tracking (see [`tropt/models/flop_counter.py`](../../tropt/models/flop_counter.py)), using the [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361) approximation. Cheap and deterministic. Requires `_model` to be a HuggingFace `PreTrainedModel`.

FLOP counting is handled entirely inside `invoke_from_tokens` / `invoke_from_texts` — these are the model-call bottleneck. To support it:

1. **Call `_update_invoke_stats`** after each raw model call inside your invoke methods, passing `n_tokens`, `n_samples`, and optionally `count_backward`.
2. For **gradient methods**: pass `count_backward=True` to `invoke_from_tokens` so it records the backward FLOPs correctly.

`set_flop_counting("manual")` will raise a `TypeError` if `_model` is not a HuggingFace `PreTrainedModel`.

### File placement

Same as for text-access models — one directory per backend, exported from `tropt/models/__init__.py`.

---

## Adding a HuggingFace Model

HuggingFace models get the full token-access compute loop for free via [`_HuggingFaceModelMixins`](../../tropt/models/huggingface/base.py). You only implement the model-specific parts.

**Existing examples**: [`LMHFModel`](../../tropt/models/huggingface/lm.py), [`EncoderHFModel`](../../tropt/models/huggingface/encoder.py).

### What `_HuggingFaceModelMixins` provides

These methods are fully implemented and call `invoke_from_tokens` internally:

- **`compute_loss_from_tokens`** — batched forward pass over all candidate triggers with automatic OOM-safe batch size reduction.
- **`compute_grad_from_tokens`** — gradient w.r.t. one-hot token representations. Handles both hard (discrete) and soft (probabilistic) triggers.
- **`compute_grad_from_embeds`** — gradient w.r.t. continuous trigger embeddings (for soft optimization).
- **`effective_embedding_matrix`** — computes the actual embedding matrix used by the model. This matters because some models (e.g., Gemma) apply scaling inside the embedding layer, so a plain `embedding_layer.weight` lookup gives incorrect embeddings.
- **`cast_to_model_tokenizer`** — cross-tokenizer casting (used by hybrid optimizers like RASLITE+).

All of these call **`invoke_from_tokens`** internally — the one method you must implement.

The mixin also handles the template loop, batching, and loss resolution via `resolve_and_compute_loss`. **FLOP counting** (see [FLOP counting](#flop-counting) above) is handled inside `invoke_from_tokens` via `_update_invoke_stats`. Since `_model` is a `PreTrainedModel`, `set_flop_counting("manual")` works out of the box — you just need your `invoke_from_tokens` to call `_update_invoke_stats` with the correct token count and `count_backward` flag.

### What you implement

**1. `__init__`** — Load the HF model and set these fields (expected by `_HuggingFaceModelMixins`):

```python
self._model                    # transformers.PreTrainedModel (or SentenceTransformer)
self._embedding_layer          # nn.Module — model.get_input_embeddings()
self._tokenizer                # HF tokenizer
self._forward_pass_batch_size  # int — starting batch size for loss computation
self._backward_pass_batch_size # int — starting batch size for gradient computation
```

Both batch sizes are automatically reduced on OOM. Set them high and let the handler dial them down.

Set the model to **eval mode** and disable gradients on its parameters — optimization gradients flow through the one-hot input, not the model weights:

```python
self._model.eval()
for param in self._model.parameters():
    param.requires_grad = False
```

Register the trigger placeholder as a special token so it tokenizes as a single ID:

```python
self._tokenizer.add_special_tokens(
    {"additional_special_tokens": [OPTIMIZED_TRIGGER_PLACEHOLDER]}
)
```

**2. `invoke_from_tokens`** — The single entry point for all white/grey-box forward passes. Receives the input embeddings and attention mask (unpacked from `ModelInput` by the caller), runs the model, and returns a `ModelOutput`.

```python
def invoke_from_tokens(self, input_embeds, input_attention_mask,
                       input_prefix_cache_kwargs=None, input_slices=None,
                       count_backward=False, **kwargs) -> ModelOutput:
    outputs = self._model(
        inputs_embeds=input_embeds,
        attention_mask=input_attention_mask,
        **(input_prefix_cache_kwargs or {}),
    )
    self._update_invoke_stats(
        n_tokens=int(input_attention_mask.sum().item()),
        n_samples=input_embeds.shape[0],
        count_backward=count_backward,
    )
    return ModelOutput(
        output_logits=outputs.logits,   # populate what your model provides
    )
```

The `count_backward` flag is set to `True` by gradient methods (`compute_grad_from_tokens`, `compute_grad_from_embeds`) so that FLOPs include the backward pass cost.

**3. `set_inputs_from_tokens`** — Build an [`_HFTokenInputManager`](../../tropt/models/huggingface/base.py) (or a model-specific subclass) and store it:

```python
def set_inputs_from_tokens(self, templates: List[str], targets: Targets = None) -> None:
    tok_ids = self.tokenizer(templates, add_special_tokens=True)["input_ids"]
    self._token_input_manager = _HFTokenInputManager(
        tok_ids=tok_ids,
        model=self._model,
        tokenizer=self.tokenizer,
        embed_func=self._embedding_layer,
        optimized_trigger_placeholder=OPTIMIZED_TRIGGER_PLACEHOLDER,
        targets=targets,
    )
```

LM models typically apply the chat template and tokenize target responses before building the manager — see [`LMHFModel.set_inputs_from_tokens`](../../tropt/models/huggingface/lm.py) for the full pattern. You may also subclass `_HFTokenInputManager` if your model has special target handling — see [`LMHFTokenInputManager`](../../tropt/models/huggingface/lm.py) (which auto-appends target response embeddings for prefill-based losses) and [`EncoderHFTokenInputManager`](../../tropt/models/huggingface/encoder.py).

**4. Inference method** — The public method called by `__call__`. This is a **separate code path** from `invoke_from_tokens` — it handles plain-text evaluation, not optimization. See [`LMHFModel.invoke_from_texts`](../../tropt/models/huggingface/lm.py) or [`EncoderHFModel.invoke_from_texts`](../../tropt/models/huggingface/encoder.py) for full examples.

### Class skeleton

```python
from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER, ModelInput, ModelOutput, Targets
from tropt.loss.base import BaseLoss
from tropt.models import (
    LMBaseModel,
    GradientTokenAccessMixin,
    LogitsTokenAccessMixin,
    LossTextAccessMixin,
    LossTokenAccessMixin,
)
from tropt.models.huggingface.base import _HFTokenInputManager, _HuggingFaceModelMixins
from tropt.models.model_mixins import GradientEmbedAccessMixin


class MyHFLMModel(
    LMBaseModel,
    _HuggingFaceModelMixins,
    # token-level access:
    LossTokenAccessMixin,
    GradientTokenAccessMixin,
    LogitsTokenAccessMixin,
    GradientEmbedAccessMixin,
    # text-level access:
    LossTextAccessMixin,
):
    def __init__(self, model_name, device=None, dtype=None):
        self._model = ...                        # AutoModelForCausalLM.from_pretrained(...)
        self._embedding_layer = ...              # self._model.get_input_embeddings()
        self._tokenizer = ...                    # AutoTokenizer.from_pretrained(...)
        self._forward_pass_batch_size = 512
        self._backward_pass_batch_size = 32

        self._tokenizer.add_special_tokens(
            {"additional_special_tokens": [OPTIMIZED_TRIGGER_PLACEHOLDER]}
        )

    @property
    def tokenizer(self): return self._tokenizer

    @property
    def device(self): return self._model.device

    def set_inputs_from_tokens(self, templates, targets=None): ...

    def invoke_from_tokens(self, input_embeds, input_attention_mask,
                           input_prefix_cache_kwargs=None, input_slices=None,
                           count_backward=False, **kwargs): ...

    def invoke_from_texts(self, input_texts, **kwargs): ...
```

You do **not** implement `compute_loss_from_tokens`, `compute_grad_from_tokens`, or `compute_grad_from_embeds` — those come from `_HuggingFaceModelMixins`.

---

## Checklist

After implementing your model:

1. **Export** — Add your class to [`tropt/models/__init__.py`](../../tropt/models/__init__.py).
2. **Test** — Write tests covering initialization, the inference method, and each mixin method. Test both single and multi-template cases. See `tests/models/` for examples.
3. **Verify optimizer compatibility** — Instantiate an optimizer that requires your model's mixins and confirm `model_requirements` validation passes.
4. **Usage stats** — Confirm `invoke_from_tokens` calls `_update_invoke_stats` with `n_tokens`, `n_samples`, and `count_backward`. Gradient methods should pass `count_backward=True` to `invoke_from_tokens`. For HuggingFace models this is already handled by `_HuggingFaceModelMixins`. This takes care of usage tracking and FLOPs.
