"""Unit checks for the shared optimizer/model helpers.

These cover the two code paths the end-to-end optimizer tests don't reach:
the embedding branch of the gradient helper (used by PEZ / SoftPrompt) and the
unconstrained-vocab branch of `random_single_flips` (used by RASLITE+).
"""

import torch

from tropt.common import Targets
from tropt.loss import PrefillCELoss
from tropt.optimizer.utils.token_initializers import random_single_flips


def test_random_single_flips_from_whitelist():
    trigger = torch.arange(8)
    valid = torch.tensor([100, 101, 102])

    out = random_single_flips(trigger, n_variations=32, valid_token_ids=valid)

    assert out.shape == (32, 8)
    assert torch.equal(out[0], trigger), "first variation must be left intact"
    for row in out[1:]:
        differing = (row != trigger).nonzero().flatten()
        # A flip can land on the value already there, so ≤ 1 position differs.
        assert len(differing) <= 1
        for pos in differing:
            assert row[pos].item() in valid.tolist()


def test_random_single_flips_from_vocab_size():
    trigger = torch.arange(6)
    vocab_size = 50

    out = random_single_flips(trigger, n_variations=16, vocab_size=vocab_size)

    assert out.shape == (16, 6)
    assert torch.equal(out[0], trigger)
    assert out.min() >= 0 and out.max() < vocab_size
    # With 15 flips over a 50-token vocab, at least one row should have moved.
    assert not torch.equal(out[1:], trigger.repeat(15, 1))


def test_random_single_flips_single_variation():
    trigger = torch.arange(4)
    out = random_single_flips(trigger, n_variations=1, vocab_size=10)
    assert torch.equal(out, trigger.unsqueeze(0))


def test_compute_grad_from_embeds(tiny_lm, lm_templates, lm_targets):
    """The embedding branch of `_grad_wrt_leaves` (PEZ / SoftPrompt flow)."""
    tiny_lm.set_inputs_from_tokens(lm_templates, lm_targets)
    try:
        trigger_len, n_candidates = 5, 2
        embeds = torch.randn(
            n_candidates, trigger_len, tiny_lm.embedding_matrix.shape[1],
            device=tiny_lm.device, dtype=tiny_lm.dtype,
        )

        grads, losses = tiny_lm.compute_grad_from_embeds(
            loss_func=PrefillCELoss(),
            candidate_trigger_embeds=embeds,
            return_loss=True,
        )

        assert grads.shape == embeds.shape
        assert losses.shape == (n_candidates,)
        assert torch.isfinite(grads).all()
        assert torch.isfinite(losses).all()
    finally:
        tiny_lm.reset_inputs_from_tokens()


def test_grad_token_and_embed_paths_agree(tiny_lm, lm_templates, lm_targets):
    """Both branches of `_grad_wrt_leaves` must satisfy the chain rule.

    Trigger embeddings are `onehot @ E`, so dL/d(onehot) = dL/d(embeds) @ E^T.
    This pins the shared gradient helper: if either branch wires the
    differentiated leaf wrongly, the identity breaks.
    """
    tiny_lm.set_inputs_from_tokens(lm_templates, lm_targets)
    try:
        loss = PrefillCELoss()
        trigger_ids = torch.tensor([[10, 20, 30, 40]], device=tiny_lm.device)
        emb_matrix = tiny_lm.embedding_matrix  # (vocab, d_model)

        grad_onehot = tiny_lm.compute_grad_from_tokens(
            loss_func=loss, candidate_trigger_ids=trigger_ids
        )  # (1, trigger_len, vocab)
        grad_embeds = tiny_lm.compute_grad_from_embeds(
            loss_func=loss, candidate_trigger_embeds=emb_matrix[trigger_ids]
        )  # (1, trigger_len, d_model)

        assert torch.allclose(
            grad_onehot, grad_embeds @ emb_matrix.T, atol=1e-4, rtol=1e-3
        )
    finally:
        tiny_lm.reset_inputs_from_tokens()


def test_compute_grad_from_embeds_keeps_message_dim(tiny_lm):
    """`keep_message_dim=True` adds the leading n_templates axis."""
    templates = ["Explain X. {{OPTIMIZED_TRIGGER}}", "Explain Y. {{OPTIMIZED_TRIGGER}}"]
    targets = Targets(target_response_strs=["Sure:", "Of course:"])
    tiny_lm.set_inputs_from_tokens(templates, targets)
    try:
        embeds = torch.randn(
            2, 4, tiny_lm.embedding_matrix.shape[1],
            device=tiny_lm.device, dtype=tiny_lm.dtype,
        )
        grads = tiny_lm.compute_grad_from_embeds(
            loss_func=PrefillCELoss(),
            candidate_trigger_embeds=embeds,
            keep_message_dim=True,
        )
        assert grads.shape == (2, *embeds.shape)  # (n_templates, n_cands, len, d)
    finally:
        tiny_lm.reset_inputs_from_tokens()
