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
