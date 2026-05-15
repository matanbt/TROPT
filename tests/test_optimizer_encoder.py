"""End-to-end happy-path for an encoder-targeting optimizer (GASLITE)."""

import math

from tropt.loss import SimilarityLoss
from tropt.optimizer import GASLITEOptimizer


def test_gaslite_optimizer_end_to_end(
    encoder_model, encoder_templates, encoder_targets
):
    optimizer = GASLITEOptimizer(
        model=encoder_model,
        loss=SimilarityLoss(),
        num_steps=2,
        n_grad=2,
        n_flip=2,
        n_candidates=8,
        seed=0,
    )
    result = optimizer.optimize_trigger(
        templates=encoder_templates,
        initial_trigger="! ! ! ! !",
        targets=encoder_targets,
    )

    assert result.best_trigger_str
    assert isinstance(result.best_loss, float)
    assert math.isfinite(result.best_loss)
    assert result.losses is not None and len(result.losses) >= 1
    assert len(result.losses) <= 2
