"""`compute_grad_from_tokens(do_gumbel_softmax=True)` (GBDA) returns dL/dtheta for the logits theta, through the
Gumbel-softmax sample y = softmax((theta + g) / tau), not dL/dy. Checked on the very sample drawn (same seed)
against dL/de from `compute_grad_from_embeds` and the softmax Jacobian."""

import torch
import torch.nn.functional as F

from tropt.loss import SimilarityLoss


def test_gumbel_softmax_grad_is_wrt_logits(encoder_model, encoder_templates, encoder_targets):
    model, loss, tau = encoder_model, SimilarityLoss(), 0.5
    model.set_inputs_from_tokens(encoder_templates, encoder_targets)
    embedding_matrix = model.embedding_matrix.detach()
    theta = 3 * torch.randn(2, 5, embedding_matrix.shape[0], generator=torch.Generator().manual_seed(0))

    torch.manual_seed(0)
    grad = model.compute_grad_from_tokens(loss_func=loss, candidate_trigger_probs=theta, do_gumbel_softmax=True,
                                          gumbel_softmax_temp=tau)

    torch.manual_seed(0)  # redraw the same sample
    y = F.gumbel_softmax(theta.to(model.device, model.dtype), tau=tau, dim=-1)
    grad_y = model.compute_grad_from_embeds(loss_func=loss, candidate_trigger_embeds=(y @ embedding_matrix).detach()) @ embedding_matrix.T
    expected = y * (grad_y - (y * grad_y).sum(-1, keepdim=True)) / tau

    assert (grad - expected).norm() < 1e-4 * expected.norm()
