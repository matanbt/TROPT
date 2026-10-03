"""
Safety Operator: optimizing a safety instruction (SI) via its operator's eigenvalue (Dherin et al., 2026).
    https://arxiv.org/abs/2609.36434

Instantiates `ContrastiveSteeringOperatorLoss` to improve an SI: its operator eigenvalue is pushed up on harmful
queries and toward ~1 (dormant) on harmless ones. Four recipes, one per trigger optimizer in the paper:
- `safety_operator_soft`: the whole SI as soft embeddings (Soft).
- `safety_operator_mixed`: frozen SI + soft suffix (Mixed).
- `safety_operator_gcg`: frozen SI + discrete GCG suffix (Hard GCG).
- `safety_operator_hardr`: frozen SI + readable discrete suffix (HardR).

We use the first `n_queries` JailbreakBench harmful/benign pairs (paper: the 100-pair train split of its 300-pair JBB-Ext, with separate eval/test splits).


SI placement: the paper's prompt is [α, Y, β] (α: system prefix, "SYSTEM:" or empty; Y: SI; β: query),
with the turn placement unspecified. We use α = "" and put Y at the start of the user turn, as TROPT
templates are user messages. This does not harm reproduction (the paper's Gemma 3 folds system
messages into the user turn identically), but other models would see Y outside their system role.
"""

import random
from typing import List, Optional

import torch
from datasets import load_dataset

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER, Targets
from tropt.loss import ContrastiveSteeringOperatorLoss
from tropt.model.huggingface.lm import LMHFModel
from tropt.optimizer import OptimizerResult
from tropt.optimizer.gcgplus_optimizer import GCGPlusOptimizer
from tropt.optimizer.soft_optimizer import SoftPromptOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.optimizer.utils.token_initializers import get_printable_random_trigger
from tropt.tracker import BaseTracker

# The paper's SI.
_SI = (
    "You are a helpful, honest, and harmless AI assistant: provide clear and effective answers, "
    "truthfully admit your limitations and uncertainties, and strictly refuse any requests that "
    "solicit dangerous, toxic, or unethical content."
)
_TOKEN_CONSTRAINTS = TokenConstraints()


def _build_data(
    model: LMHFModel,
    n_queries: int,
    with_si: bool,
    harmful_queries: Optional[List[str]] = None,
    harmless_queries: Optional[List[str]] = None,
    safety_instruction: str = _SI,
) -> tuple[List[str], Targets]:
    """Templates `[SI ][trigger]\\n\\n[query]` over the first `n_queries` JailbreakBench harmful/benign pairs,
    plus targets: A_clean (MLP inputs of the bare query) and harmful labels.

    Harmful templates come first, then harmless. Per template, `target_hidden_states` is A_clean (the last-token
    pre-MLP hidden states of the bare query, without the SI) and `target_class_idx` is 1 (harmful) / 0 (harmless).

    Deviations from the paper's data (JBB-Ext, App. H):
    - Only the 100 original JailbreakBench pairs (paper adds 200 synthetic pairs).
    - No stratified train/eval/test split: we take the first `n_queries` pairs, none held out.
    - Per-step harmful/harmless queries are drawn independently, not as matched pairs.

    Args:
        n_queries: Number of harmful and of harmless queries (each); the first `n_queries` JailbreakBench
            pairs (paper: the 100-pair train split of its 300-pair JBB-Ext).
        with_si: Prepend the (frozen) SI before the trigger; False when the trigger is the SI itself.
        harmful_queries, harmless_queries: Custom queries replacing JailbreakBench (then `n_queries` is ignored).
        safety_instruction: Custom SI replacing the paper's SI.
    """

    # Load queries
    if harmful_queries is None and harmless_queries is None:
        jbb = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
        assert 0 < n_queries <= len(jbb["harmful"]), f"`n_queries` must be in [1, {len(jbb['harmful'])}]."
        harmful_queries, harmless_queries = jbb["harmful"]["Goal"][:n_queries], jbb["benign"]["Goal"][:n_queries]
    assert harmful_queries and harmless_queries, "Need at least one harmful and one harmless query (pass both lists)."
    queries = list(harmful_queries) + list(harmless_queries)

    # Compute the last-token pre-MLP activations w/o SI (paper's A_clean)
    a_clean = []
    with torch.no_grad():
        for query in queries:
            input_ids = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": query}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **model.chat_template_kwargs,
            )["input_ids"].to(model.device)
            out = model.invoke_from_tokens(input_ids=input_ids, require_pre_mlp_hidden_states=True)
            assert out.full_pre_mlp_hidden_states is not None
            a_clean.append(out.full_pre_mlp_hidden_states[0, :, -1, :])  # (n_layers, d_model)

    prefix = safety_instruction + " " if with_si else ""
    templates = [f"{prefix}{OPTIMIZED_TRIGGER_PLACEHOLDER}\n\n{q}" for q in queries]
    targets = Targets(
        target_hidden_states=torch.stack(a_clean),
        target_class_idx=[1] * len(harmful_queries) + [0] * len(harmless_queries),
    )  # TODO(tropt-feature) in the future we should support template with system and multi-turns (with dict); this would've allowed us here to use the SI as a system message
    return templates, targets


def _one_harmful_one_harmless_batch_sampler(templates: List[str], targets: Targets) -> List[int]:
    """Per-step batch of the paper: one random harmful and one random harmless template."""
    labels = targets.target_class_idx
    return [random.choice([i for i, c in enumerate(labels) if c == label]) for label in (1, 0)]


def safety_operator_soft(
    model_name: str = "google/gemma-3-1b-it",
    n_queries: int = 100,
    suppression_weight: float = 10.0,
    targeted_layers: slice = slice(12, 24),  # paper's layer range (Gemma 3 1B)
    override_data_args: Optional[dict] = None,
    model_obj: Optional[LMHFModel] = None,
    tracker: Optional[BaseTracker] = None,
    seed: Optional[int] = None,
) -> OptimizerResult:
    """Optimizes the whole SI's embeddings (paper's Soft variant).

    Deviations from the paper (App. C):
    - No L2 drift regularizer (paper: λ_reg = 0.5 on ||E_Y - E_Y0||²_F / (M·d)).

    Args:
        n_queries: Number of harmful and of harmless JailbreakBench queries (each) to optimize over.
        suppression_weight: ρ, weight of the harmless "keep λ≈1" term; lower means more refusal.
        targeted_layers: Layers to average the loss over (0-based, end-exclusive); must fit the model.
        override_data_args: Optional overrides for `_build_data`: `harmful_queries` / `harmless_queries` (custom
            queries instead of JailbreakBench) and/or `safety_instruction` (custom SI instead of the paper's).
    """
    initial_trigger = (override_data_args or {}).get("safety_instruction", _SI)  # optimized embeddings init from the SI
    model = model_obj or LMHFModel(model_name=model_name)
    templates, targets = _build_data(model, n_queries, with_si=False, **(override_data_args or {}))
    optimizer = SoftPromptOptimizer(
        model=model,
        loss=ContrastiveSteeringOperatorLoss(suppression_weight=suppression_weight, targeted_layers=targeted_layers),
        tracker=tracker,
        seed=seed,
        num_steps=250,
        learning_rate=0.005,
        template_batch_sampler=_one_harmful_one_harmless_batch_sampler,
    )
    return optimizer.optimize_trigger(templates=templates, targets=targets, initial_trigger=initial_trigger)


def safety_operator_mixed(
    model_name: str = "google/gemma-3-1b-it",
    n_queries: int = 100,
    suppression_weight: float = 2000.0,
    targeted_layers: slice = slice(8, 24),  # paper's layer range (Gemma 3 1B)
    trigger_len: int = 10,
    override_data_args: Optional[dict] = None,
    model_obj: Optional[LMHFModel] = None,
    tracker: Optional[BaseTracker] = None,
    seed: Optional[int] = None,
) -> OptimizerResult:
    """Optimizes a soft suffix after the frozen SI (paper's Mixed variant).

    Deviations from the paper (App. D):
    - No L2 drift regularizer.

    Args:
        n_queries: Number of harmful and of harmless JailbreakBench queries (each) to optimize over.
        suppression_weight: ρ, weight of the harmless "keep λ≈1" term; lower means more refusal.
        targeted_layers: Layers to average the loss over (0-based, end-exclusive); must fit the model.
        override_data_args: Optional overrides for `_build_data`: `harmful_queries` / `harmless_queries` (custom
            queries instead of JailbreakBench) and/or `safety_instruction` (custom SI instead of the paper's).
        trigger_len: Number of suffix tokens (between the paper's SI and the query); randomly initialized.
    """
    model = model_obj or LMHFModel(model_name=model_name)
    templates, targets = _build_data(model, n_queries, with_si=True, **(override_data_args or {}))
    optimizer = SoftPromptOptimizer(
        model=model,
        loss=ContrastiveSteeringOperatorLoss(suppression_weight=suppression_weight, targeted_layers=targeted_layers),
        tracker=tracker,
        seed=seed,
        num_steps=250,
        learning_rate=0.005,
        template_batch_sampler=_one_harmful_one_harmless_batch_sampler,
    )
    # drawn after the optimizer sets the seed
    initial_trigger = get_printable_random_trigger(trigger_len, tokenizer=model.tokenizer, token_constraints=_TOKEN_CONSTRAINTS)
    return optimizer.optimize_trigger(templates=templates, targets=targets, initial_trigger=initial_trigger)


def safety_operator_gcg(
    model_name: str = "google/gemma-3-1b-it",
    n_queries: int = 100,
    suppression_weight: float = 5.0,
    targeted_layers: slice = slice(8, 24),  # paper's layer range (Gemma 3 1B)
    trigger_len: int = 10,
    override_data_args: Optional[dict] = None,
    model_obj: Optional[LMHFModel] = None,
    tracker: Optional[BaseTracker] = None,
    seed: Optional[int] = None,
) -> OptimizerResult:
    """Optimizes a discrete GCG suffix after the frozen SI (paper's Hard GCG variant).

    Args:
        n_queries: Number of harmful and of harmless JailbreakBench queries (each) to optimize over.
        suppression_weight: ρ, weight of the harmless "keep λ≈1" term; lower means more refusal.
        targeted_layers: Layers to average the loss over (0-based, end-exclusive); must fit the model.
        override_data_args: Optional overrides for `_build_data`: `harmful_queries` / `harmless_queries` (custom
            queries instead of JailbreakBench) and/or `safety_instruction` (custom SI instead of the paper's).
        trigger_len: Number of suffix tokens (between the paper's SI and the query), randomly initialized.
    """
    model = model_obj or LMHFModel(model_name=model_name)
    templates, targets = _build_data(model, n_queries, with_si=True, **(override_data_args or {}))
    optimizer = GCGPlusOptimizer(
        model=model,
        loss=ContrastiveSteeringOperatorLoss(suppression_weight=suppression_weight, targeted_layers=targeted_layers),
        tracker=tracker,
        seed=seed,
        num_steps=250,
        n_candidates=128,
        sample_topk=128,
        token_constraints=_TOKEN_CONSTRAINTS,
        template_batch_sampler=_one_harmful_one_harmless_batch_sampler,
    )
    # drawn after the optimizer sets the seed
    initial_trigger = get_printable_random_trigger(trigger_len, tokenizer=model.tokenizer, token_constraints=_TOKEN_CONSTRAINTS)
    return optimizer.optimize_trigger(templates=templates, targets=targets, initial_trigger=initial_trigger)


def safety_operator_hardr(
    model_name: str = "google/gemma-3-1b-it",
    n_queries: int = 100,
    suppression_weight: float = 350.0,
    targeted_layers: slice = slice(8, 24),  # paper's layer range (Gemma 3 1B)
    readability_weight: float = 0.1,
    initial_trigger: str = (
        "Always maintain safety: carefully analyze the underlying intent before responding, "
        "and strictly decline any requests seeking dangerous, illegal, or harmful assistance."
    ),
    override_data_args: Optional[dict] = None,
    model_obj: Optional[LMHFModel] = None,
    tracker: Optional[BaseTracker] = None,
    seed: Optional[int] = None,
) -> OptimizerResult:
    """Optimizes a readable discrete suffix after the frozen SI (paper's HardR variant).

    Args:
        n_queries: Number of harmful and of harmless JailbreakBench queries (each) to optimize over.
        suppression_weight: ρ, weight of the harmless "keep λ≈1" term; lower means more refusal.
        targeted_layers: Layers to average the loss over (0-based, end-exclusive); must fit the model.
        override_data_args: Optional overrides for `_build_data`: `harmful_queries` / `harmless_queries` (custom
            queries instead of JailbreakBench) and/or `safety_instruction` (custom SI instead of the paper's).
        readability_weight: γ, weight of the suffix's log-likelihood in the token score and candidate ranking.
        initial_trigger: Initial suffix (the paper's).
        model_obj: Pre-loaded model; must have `use_prefix_cache=False` (for the suffix log-probs).
    """
    model = model_obj or LMHFModel(model_name=model_name, use_prefix_cache=False)  # suffix log-probs need the full prefix
    templates, targets = _build_data(model, n_queries, with_si=True, **(override_data_args or {}))
    optimizer = GCGPlusOptimizer(
        model=model,
        loss=ContrastiveSteeringOperatorLoss(suppression_weight=suppression_weight, targeted_layers=targeted_layers),
        tracker=tracker,
        seed=seed,
        num_steps=250,
        n_candidates=128,
        sample_topk=128,
        token_constraints=_TOKEN_CONSTRAINTS,
        template_batch_sampler=_one_harmful_one_harmless_batch_sampler,
        # the loss is a mean over the 1+1 batch (½ the paper's harmful + ρ·harmless sum), hence γ/2
        token_prior_weight=readability_weight / 2,
        sample_temperature=0.5,  # T_mut
    )
    return optimizer.optimize_trigger(templates=templates, targets=targets, initial_trigger=initial_trigger)
