"""Import smoke test. Catches packaging / circular-import regressions cheaply."""


def test_top_level_imports():
    import tropt
    from tropt.common import Targets
    from tropt.loss import PrefillCELoss, SimilarityLoss
    from tropt.model import LMHFModel
    from tropt.model.huggingface import EncoderHFModel
    from tropt.optimizer import (
        GASLITEOptimizer,
        GBDAOptimizer,
        GCGOptimizer,
        GCGPlusOptimizer,
        PALOptimizer,
        QCGOptimizer,
    )

    assert tropt is not None
    # Reference each symbol so the linter is satisfied and the names are checked.
    assert all(
        cls is not None
        for cls in (
            Targets,
            PrefillCELoss,
            SimilarityLoss,
            LMHFModel,
            EncoderHFModel,
            GASLITEOptimizer,
            GBDAOptimizer,
            GCGOptimizer,
            GCGPlusOptimizer,
            PALOptimizer,
            QCGOptimizer,
        )
    )
