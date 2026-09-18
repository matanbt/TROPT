"""PassOnModel: candidate triggers must reach a trigger-only loss verbatim."""

from dataclasses import dataclass
from typing import ClassVar, List

import torch

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER
from tropt.loss import BaseLoss
from tropt.model import PassOnModel
from tropt.optimizer.rs_optimizer import RandomSearchOptimizer

TOKENIZER_NAME = "google/gemma-3-270m-it"


@dataclass
class _CountLetterLoss(BaseLoss):
    """Oracle stand-in: minimized by triggers packed with 'a'."""

    is_differentiable: ClassVar[bool] = False

    def __call__(self, input_trigger_strs: List[str]) -> torch.Tensor:
        return torch.tensor([-s.count("a") for s in input_trigger_strs], dtype=torch.float32)


def test_trigger_reaches_loss_verbatim():
    model = PassOnModel(tokenizer=TOKENIZER_NAME)
    model.set_inputs_from_texts(templates=[OPTIMIZED_TRIGGER_PLACEHOLDER])

    losses = model.compute_loss_from_texts(["aaa", "b"], loss_func=_CountLetterLoss())

    assert losses.tolist() == [-3.0, 0.0]
    assert model.get_usage_stats()["total_tokens"] > 0


def test_rs_optimizes_against_oracle_loss():
    model = PassOnModel(tokenizer=TOKENIZER_NAME)
    optimizer = RandomSearchOptimizer(
        model=model, loss=_CountLetterLoss(), num_steps=20, n_candidates=32, seed=0,
    )

    result = optimizer.optimize_trigger(
        templates=[OPTIMIZED_TRIGGER_PLACEHOLDER], initial_trigger="zzzzzzzz",
    )

    assert result.best_loss < 0
    assert result.best_trigger_str is not None and "a" in result.best_trigger_str


def test_token_flow_builds_bare_trigger_inputs():
    model = PassOnModel(tokenizer=TOKENIZER_NAME)
    model.set_inputs_from_tokens(templates=[OPTIMIZED_TRIGGER_PLACEHOLDER])
    assert model._token_input_manager is not None

    trigger_ids = model.tokenizer.encode_trigger("hello").unsqueeze(0)
    model_input = model._token_input_manager.get_triggered_inputs(
        chosen_template_idx=0, trigger_ids=trigger_ids
    )

    # No template, no special tokens: the built input is the trigger itself.
    assert model_input.input_texts == ["hello"]
    assert model_input.input_trigger_strs == ["hello"]


@dataclass
class _PreferMoreAsLoss(BaseLoss):
    """Pairwise stand-in mirroring ``PairwiseRelativeOracleLoss``: two inputs
    (incumbent, challenger); the challenger 'wins' (gets -1.0) iff it has strictly
    more 'a's. One input -> ``[1.0]`` (no comparison). Enforces exactly two."""

    is_differentiable: ClassVar[bool] = False

    def __call__(self, input_trigger_strs: List[str]) -> torch.Tensor:
        if len(input_trigger_strs) < 2:
            return torch.ones(1)
        assert len(input_trigger_strs) == 2, "pairwise loss expects exactly two inputs"
        ref, chal = input_trigger_strs[0].count("a"), input_trigger_strs[1].count("a")
        preferred = [1.0, -1.0] if chal > ref else [-1.0, 1.0]
        return torch.tensor(preferred, dtype=torch.float32)


def test_pairwise_relative_loss_hill_climb():
    """RandomSearchOptimizer always prepends the incumbent, so n_candidates=1 gives
    exactly one incumbent-vs-challenger pair; the surviving trigger only gains 'a's."""
    from tropt.optimizer.utils.token_constraints import TokenConstraints

    model = PassOnModel(tokenizer=TOKENIZER_NAME)
    optimizer = RandomSearchOptimizer(
        model=model,
        loss=_PreferMoreAsLoss(),
        num_steps=40,
        n_candidates=1,  # + the always-prepended incumbent => one pairwise query
        patience=0,  # no restarts: pure monotone climb
        token_constraints=TokenConstraints(disallow_non_ascii=True),
        seed=0,
    )

    result = optimizer.optimize_trigger(
        templates=[OPTIMIZED_TRIGGER_PLACEHOLDER], initial_trigger="aazzzzzz",
    )

    # A challenger is accepted only with strictly more 'a's, so the final surviving
    # trigger never drops below the initial count.
    assert result.best_trigger_str is not None
    assert result.best_trigger_str.count("a") >= "aazzzzzz".count("a")


def test_pairwise_loss_rejects_more_than_two_candidates():
    """A pairwise loss needs n_candidates=1; a larger batch trips its own assert."""
    import pytest

    model = PassOnModel(tokenizer=TOKENIZER_NAME)
    optimizer = RandomSearchOptimizer(
        model=model, loss=_PreferMoreAsLoss(), num_steps=5, n_candidates=2, patience=0,
    )
    with pytest.raises(AssertionError):
        optimizer.optimize_trigger(
            templates=[OPTIMIZED_TRIGGER_PLACEHOLDER], initial_trigger="aazzzzzz",
        )
