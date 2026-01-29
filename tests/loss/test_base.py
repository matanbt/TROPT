import torch

from tropt.loss.base import SimilarityLoss, SteeringActivationLoss


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


def test_steering_loss_shape():
    """Test that SteeringActivationLoss returns correct output shape."""
    loss_fn = SteeringActivationLoss()

    bsz, n_layers, seq_len, d_model = 2, 4, 10, 128
    hidden_states = torch.randn(bsz, n_layers, seq_len, d_model)
    target_directions = torch.randn(bsz, d_model)

    loss = loss_fn(hidden_states, target_directions)

    assert loss.shape == (bsz,)
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()


def test_steering_activation_loss_known_values():
    """Test SteeringActivationLoss with known cosine similarity values."""
    loss_fn = SteeringActivationLoss()

    # Create hidden states where all positions and layers have the same direction
    bsz, n_layers, seq_len, d_model = 1, 2, 3, 2

    # Case 1: Perfect alignment (hidden states = target direction)
    direction = torch.tensor([1.0, 0.0])
    hidden_states = direction.view(1, 1, 1, 2).expand(bsz, n_layers, seq_len, d_model)
    target_directions = direction.view(1, 2)

    loss = loss_fn(hidden_states, target_directions)
    # Perfect alignment: cos_sim = 1, loss = -1
    assert torch.allclose(loss, torch.tensor([-1.0]), atol=1e-6)

    # Case 2: Orthogonal (cos_sim = 0)
    hidden_states = torch.tensor([0.0, 1.0]).view(1, 1, 1, 2).expand(bsz, n_layers, seq_len, d_model)
    target_directions = torch.tensor([[1.0, 0.0]])

    loss = loss_fn(hidden_states, target_directions)
    # Orthogonal: cos_sim = 0, loss = 0
    assert torch.allclose(loss, torch.tensor([0.0]), atol=1e-6)

    # Case 3: Opposite direction (cos_sim = -1)
    hidden_states = torch.tensor([-1.0, 0.0]).view(1, 1, 1, 2).expand(bsz, n_layers, seq_len, d_model)
    target_directions = torch.tensor([[1.0, 0.0]])

    loss = loss_fn(hidden_states, target_directions)
    # Opposite: cos_sim = -1, loss = 1
    assert torch.allclose(loss, torch.tensor([1.0]), atol=1e-6)


def test_steering_activation_loss_with_slices():
    """Test SteeringActivationLoss with position slices."""
    loss_fn = SteeringActivationLoss(slc_name="adv")

    bsz, n_layers, seq_len, d_model = 2, 3, 10, 4
    hidden_states = torch.randn(bsz, n_layers, seq_len, d_model)
    target_directions = torch.randn(bsz, d_model)

    # Define slices for each batch element
    slices = [
        {"adv": slice(2, 5)},  # positions 2-4 for first element
        {"adv": slice(3, 7)},  # positions 3-6 for second element
    ]

    loss = loss_fn(hidden_states, target_directions, slices=slices)

    assert loss.shape == (bsz,)
    assert not torch.isnan(loss).any()


def test_steering_enh_loss_targeted_layers():
    """Test SteeringActivationLoss with targeted layers."""
    # Only target middle layers
    loss_fn = SteeringActivationLoss(targeted_layers=slice(1, 3))

    bsz, n_layers, seq_len, d_model = 1, 4, 5, 2

    # Create hidden states with different values in different layers
    # Layers 0,3 are orthogonal to target, layers 1,2 are aligned
    target_direction = torch.tensor([[1.0, 0.0]])

    hidden_states = torch.zeros(bsz, n_layers, seq_len, d_model)
    hidden_states[:, 0, :, :] = torch.tensor([0.0, 1.0])  # Layer 0: orthogonal
    hidden_states[:, 1, :, :] = torch.tensor([1.0, 0.0])  # Layer 1: aligned
    hidden_states[:, 2, :, :] = torch.tensor([1.0, 0.0])  # Layer 2: aligned
    hidden_states[:, 3, :, :] = torch.tensor([0.0, 1.0])  # Layer 3: orthogonal

    loss = loss_fn(hidden_states, target_direction)

    # Should only consider layers 1-2 (aligned), so cos_sim = 1, loss = -1
    assert torch.allclose(loss, torch.tensor([-1.0]), atol=1e-6)


def test_steering_activation_loss_batch():
    """Test SteeringActivationLoss with batch size > 1."""
    loss_fn = SteeringActivationLoss()

    bsz, n_layers, seq_len, d_model = 3, 2, 4, 8
    hidden_states = torch.randn(bsz, n_layers, seq_len, d_model)
    target_directions = torch.randn(bsz, d_model)

    loss = loss_fn(hidden_states, target_directions)

    assert loss.shape == (bsz,)
    assert not torch.isnan(loss).any()
    # Each batch element should have different loss values (with high probability)
    assert not torch.allclose(loss[0], loss[1], atol=1e-3)


# TODO test shapes of other losses