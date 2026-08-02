"""End-to-end happy-path for an encoder-targeting optimizer (GASLITE)."""

import math

from tropt.common import Targets
from tropt.loss import SimilarityLoss
from tropt.model.huggingface import EncoderHFModel
from tropt.optimizer import GASLITEOptimizer
from tropt.optimizer.soft_optimizer import SoftPromptOptimizer

from tests.conftest import ENCODER_NAME


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


def test_soft_prompt_stays_finite_in_half_precision(encoder_templates):
    """Soft-prompt optimization must not diverge when the model is fp16.

    Adam keeps its state in the parameter's dtype, and in fp16 both `grad ** 2`
    and the default `eps=1e-8` flush to zero -- so the first step divides by
    zero, the prompt becomes +-inf and every later loss is NaN. The rest of this
    suite loads models in float32, where the bug is invisible.
    """
    fp16_encoder = EncoderHFModel(
        model_name=ENCODER_NAME, device="cpu", dtype="float16",
    )
    targets = Targets(
        target_vectors=fp16_encoder(["This product is excellent."]).detach(),
    )

    result = SoftPromptOptimizer(
        model=fp16_encoder, loss=SimilarityLoss(), num_steps=3, seed=0,
    ).optimize_trigger(
        templates=encoder_templates,
        initial_trigger="! ! ! ! !",
        targets=targets,
    )

    assert result.losses is not None and len(result.losses) == 3
    assert all(math.isfinite(loss) for loss in result.losses), result.losses
    assert math.isfinite(result.best_loss)
