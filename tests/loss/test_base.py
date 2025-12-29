import torch

from tropt.loss.base import SimilarityLoss


def test_similarity_loss():
    loss_fn = SimilarityLoss()
    # vectors: (bsz, d_model)
    # target: (bsz, d_model)
    
    v1 = torch.tensor([[1.0, 0.0]])
    t1 = torch.tensor([[1.0, 0.0]])
    # Cosine sim is 1. Loss is -1.
    loss = loss_fn(v1, t1)
    assert loss.shape == (1,)
    assert torch.allclose(loss, torch.tensor([-1.0]), atol=1e-6)

    # Test batch
    v4 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    t4 = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss = loss_fn(v4, t4)
    assert loss.shape == (2,)
    # First pair: cos sim 1 -> loss -1; second pair: cos sim 0 -> loss 0
    assert torch.allclose(loss, torch.tensor([-1.0, 0.0]), atol=1e-6)

# TODO test shapes of other losses