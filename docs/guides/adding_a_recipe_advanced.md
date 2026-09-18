# Composing a Recipe: Advanced

[Composing a Recipe](adding_a_recipe.md) built one attack end-to-end: a white-box,
gradient-based jailbreak ({py:class}`~tropt.model.LMHFModel` +
{py:class}`~tropt.loss.PrefillCELoss` + {py:class}`~tropt.optimizer.GCGOptimizer`).
That is a single composition *family*; the Recipe Hub spans several more. This guide
catalogs the other patterns as short, self-contained entries — what each is for, the
one wiring detail that makes it work, and a canonical recipe to copy from.

The organizing principle is the **access contract**: every optimizer declares
`model_requirements` (a tuple of access *mixins*), and a pattern's model must implement
them. Choosing a pattern is mostly choosing a compatible *(model, loss, optimizer)*
triple — see [Recipe Component Compatibility](adding_a_recipe.md#recipe-component-compatibility).
For the full, always-current set of recipes, browse `tropt/recipe_hub/`.


## Black-box attacks (text access only)

When you can only send text and read text back — a hosted API, a model behind a
guardrail — the optimizer must be gradient-free. These recipes require only
{py:class}`~tropt.model.LossTextAccessMixin`: the optimizer proposes candidate triggers,
the model runs them as text, and a *text* loss scores the output. TROPT's black-box
optimizers are {py:class}`~tropt.optimizer.RandomSearchOptimizer` (block-random search)
and {py:class}`~tropt.optimizer.BeamSearchOptimizer` (LM-guided beam search).

Two things change versus the white-box recipe:

- **The victim is API-backed.** Use {py:class}`~tropt.model.LiteLLMModel` (any
  OpenAI / Anthropic / … model via LiteLLM) instead of `LMHFModel`. It exposes a
  tiktoken tokenizer for the optimizer's token-level mutations, but no gradients.
- **The loss reads generated text or first-token logprobs, not prefill logits.**
  For example {py:class}`~tropt.loss.FirstTokenNLLLoss` (maximize the logprob of a
  target first token, read from the API's top-k logprobs),
  {py:class}`~tropt.loss.InputFluencyLoss` / response-harmfulness judges
  (`require_generation`, non-differentiable), or
  {py:class}`~tropt.loss.ExternalTriggerPerplexityLoss`.

```python
from tropt.model import LiteLLMModel
from tropt.loss import FirstTokenNLLLoss
from tropt.optimizer import RandomSearchOptimizer

model = LiteLLMModel(model_name="openai/gpt-4.1-mini")      # black-box victim
optimizer = RandomSearchOptimizer(model=model, loss=FirstTokenNLLLoss(target_token="Sure"))
```

Canonical recipes: {py:func}`~tropt.recipe_hub.prs__andriushchenko2024` (logprob random
search), {py:func}`~tropt.recipe_hub.beast__sadasivan2024` (beam search),
{py:func}`~tropt.recipe_hub.ral__sitawarin2024` (proxy-free RS).


## Oracle loss (no target model)

Some losses are *self-contained oracles*: they score the trigger text on its own — an
external API scoring the candidate, a custom metric, a pairwise judge — so there is no
target model to query. Pair such a loss with {py:class}`~tropt.model.PassOnModel`, a
model with no weights that simply forwards candidate triggers to the loss (it still
exposes a tokenizer for the search), and any `LossTextAccessMixin` optimizer.

```python
from tropt.model import PassOnModel
from tropt.optimizer import RandomSearchOptimizer

optimizer = RandomSearchOptimizer(model=PassOnModel(), loss=my_oracle_loss)
```

Canonical recipes: {py:func}`~tropt.recipe_hub.rs_oracle` (hill-climb any trigger-only
loss, e.g. {py:class}`~tropt.loss.ExternalTriggerPerplexityLoss`) and
{py:func}`~tropt.recipe_hub.ask_for_directions__zhang2025` (the victim itself acts as a
pairwise judge *inside* the loss).

```{admonition} Gotcha — relative/pairwise losses have no global scale
:class: warning

A loss like {py:class}`~tropt.loss.PairwiseRelativeOracleLoss` only ranks a challenger
*against the current trigger*; its absolute value is meaningless. Use an optimizer that
keeps the best candidate *per step* ({py:class}`~tropt.optimizer.RandomSearchOptimizer`)
rather than one that tracks a global best-loss, and set `n_candidates=1` so each step is
exactly one incumbent-vs-challenger comparison.
```


## Embedding / retrieval attacks

Here the target is an **embedding model**, and the objective is a *vector*, not a
response. Use {py:class}`~tropt.model.EncoderHFModel` or
{py:class}`~tropt.model.EncoderOpenAIModel` with {py:class}`~tropt.loss.SimilarityLoss`,
and pass a target embedding via `Targets(target_vectors=...)` rather than
`target_response_strs`. This is TROPT's largest recipe family (corpus-poisoning /
retrieval attacks), and it works white-box or black-box.

Canonical recipes: {py:func}`~tropt.recipe_hub.rs_emb` (black-box RS on an embedding
model), {py:func}`~tropt.recipe_hub.gasliteplus_encoder`,
{py:func}`~tropt.recipe_hub.rasliteplus`.


## Continuous / relaxation optimizers

These optimize in the model's *embedding space* and project back to discrete tokens, so
they need {py:class}`~tropt.model.GradientEmbedAccessMixin` — a different access level
than the token-level gradient GCG uses. Use
{py:class}`~tropt.optimizer.SoftPromptOptimizer`,
{py:class}`~tropt.optimizer.PEZOptimizer`, or
{py:class}`~tropt.optimizer.GBDAOptimizer`.

Canonical recipes: {py:func}`~tropt.recipe_hub.soft_prompt__schwinn2024`,
{py:func}`~tropt.recipe_hub.pez__wen2023`, {py:func}`~tropt.recipe_hub.gbda__guo2021`.


## Hidden-state and classifier objectives

Some attacks optimize an *internal* signal rather than the output text:
{py:class}`~tropt.loss.SteeringActivationLoss` steers hidden activations toward a
direction (`require_hidden_states`), and misclassification losses target a classifier's
decision. The model must expose the matching output (hidden states / class logits).

Canonical recipes: {py:func}`~tropt.recipe_hub.iris__huang2025`,
{py:func}`~tropt.recipe_hub.mac__zhang2024`.


## Surrogate / proxy models (transfer)

When the target is hard to differentiate (or black-box), some optimizers optimize
against a *separate* white-box surrogate and transfer the result. Optimizers such as
{py:class}`~tropt.optimizer.PALOptimizer` and `QCGOptimizer` accept a `proxy_model`
distinct from the target — the base guide passes `proxy_model=model` for simplicity, but
any surrogate LM works.

Canonical recipes: {py:func}`~tropt.recipe_hub.pal__sitawarin2024`,
{py:func}`~tropt.recipe_hub.qcg__hayase2024`.


---

Any *(model, loss, optimizer)* triple that satisfies the access contract is fair game;
these patterns are the common, tested shapes. When in doubt, try the combination — TROPT
reports an incompatibility at initialization or first run. See the
[Compatibility Matrix](compatibility_matrix.md) for a rough automated list.
