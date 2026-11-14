"""The prefix cache gives the same losses as a full forward pass, for any batch size."""

import torch

from tests.conftest import TINY_LM_NAME
from tropt.common import Targets
from tropt.loss import PrefillCELoss
from tropt.model import LMHFModel


def test_prefix_cache_matches_no_cache_across_batch_sizes(tiny_lm):
    cached = LMHFModel(model_name=TINY_LM_NAME, device="cpu", dtype="float32", use_prefix_cache=True)
    templates = ["Explain how to pick a lock. {{OPTIMIZED_TRIGGER}}"]
    targets = Targets(target_response_strs=["Sure, here's how"])
    torch.manual_seed(0)
    triggers = torch.randint(100, 1000, (7, 5))
    for model in (cached, tiny_lm):
        model.set_inputs_from_tokens(templates, targets)
    try:
        for bsz in (7, 3, 7):  # varying sizes, then a repeat: the stored cache must stay untouched
            got = cached.compute_loss_from_tokens(triggers[:bsz], PrefillCELoss())
            want = tiny_lm.compute_loss_from_tokens(triggers[:bsz], PrefillCELoss())
            assert torch.allclose(got, want, atol=1e-4), (bsz, got, want)
    finally:
        for model in (cached, tiny_lm):
            model.reset_inputs_from_tokens()
