"""End-to-end happy-path tests for LM-targeting optimizers.

Each test wires a real (tiny) LM + a real loss + the optimizer, runs 2 steps
with small candidate budgets, and asserts on the shape/finiteness of the
result. Loss-decrease is intentionally not asserted: 2 steps on a 270M model
is too short to guarantee improvement on every seed.
"""

import math

from tropt.loss import PrefillCELoss
from tropt.optimizer import (
    GBDAOptimizer,
    GCGOptimizer,
    GCGPlusOptimizer,
    PALOptimizer,
    QCGOptimizer,
)

INITIAL_TRIGGER = "! ! ! ! !"


def _assert_result_ok(result, expected_num_steps):
    assert result.best_trigger_str, "best_trigger_str should be non-empty"
    assert isinstance(result.best_loss, float)
    assert math.isfinite(result.best_loss)
    assert result.losses is not None and len(result.losses) >= 1
    assert len(result.losses) <= expected_num_steps


def test_gcg_optimizer_end_to_end(tiny_lm, lm_templates, lm_targets):
    optimizer = GCGOptimizer(
        model=tiny_lm,
        loss=PrefillCELoss(),
        num_steps=2,
        n_candidates=8,
        sample_topk=8,
        sample_n_replace=1,
        seed=0,
    )
    result = optimizer.optimize_trigger(
        templates=lm_templates,
        initial_trigger=INITIAL_TRIGGER,
        targets=lm_targets,
    )
    _assert_result_ok(result, expected_num_steps=2)


def test_gcgplus_optimizer_end_to_end(tiny_lm, lm_templates, lm_targets):
    optimizer = GCGPlusOptimizer(
        model=tiny_lm,
        loss=PrefillCELoss(),
        num_steps=2,
        n_candidates=8,
        sample_topk=8,
        seed=0,
    )
    result = optimizer.optimize_trigger(
        templates=lm_templates,
        initial_trigger=INITIAL_TRIGGER,
        targets=lm_targets,
    )
    _assert_result_ok(result, expected_num_steps=2)


def test_gbda_optimizer_end_to_end(tiny_lm, lm_templates, lm_targets):
    optimizer = GBDAOptimizer(
        model=tiny_lm,
        loss=PrefillCELoss(),
        num_steps=2,
        n_grad_samples=2,
        n_final_gumbel_samples=4,
        seed=0,
    )
    result = optimizer.optimize_trigger(
        templates=lm_templates,
        initial_trigger=INITIAL_TRIGGER,
        targets=lm_targets,
    )
    _assert_result_ok(result, expected_num_steps=2)


def test_pal_optimizer_end_to_end(tiny_lm, lm_templates, lm_targets):
    # Self-proxy: model serves as its own proxy (white-box mode).
    optimizer = PALOptimizer(
        model=tiny_lm,
        loss=PrefillCELoss(),
        num_steps=2,
        n_candidates=8,
        sample_topk=8,
        sample_n_replace=1,
        seed=0,
    )
    result = optimizer.optimize_trigger(
        templates=lm_templates,
        initial_trigger=INITIAL_TRIGGER,
        targets=lm_targets,
    )
    _assert_result_ok(result, expected_num_steps=2)


def test_qcg_optimizer_end_to_end(tiny_lm, lm_templates, lm_targets):
    # QCG defaults are tuned for real runs (thousands of candidates); shrink for tests.
    optimizer = QCGOptimizer(
        model=tiny_lm,
        loss=PrefillCELoss(),
        num_steps=2,
        n_proxy_candidates=16,
        n_target_candidates=4,
        buffer_size=4,
        seed=0,
    )
    result = optimizer.optimize_trigger(
        templates=lm_templates,
        initial_trigger=INITIAL_TRIGGER,
        targets=lm_targets,
    )
    _assert_result_ok(result, expected_num_steps=2)
