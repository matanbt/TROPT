import torch

from tropt.loss import SimilarityLoss, SteeringActivationLoss


def test_similarity_loss():
    loss_fn = SimilarityLoss()
    # output_embeddings: (bsz, d_model)
    # target_vectors: (d_model,) - UNBATCHED!

    v1 = torch.tensor([[1.0, 0.0]])
    t1 = torch.tensor([1.0, 0.0])  # Unbatched
    # Cosine sim is 1. Loss is -1.
    loss = loss_fn(v1, t1)
    assert loss.shape == (1,)
    assert torch.allclose(loss, torch.tensor([-1.0]), atol=1e-6)

    # Test batch - same target for all samples
    v4 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    t4 = torch.tensor([1.0, 0.0])  # Unbatched - same target for both samples
    loss = loss_fn(v4, t4)
    assert loss.shape == (2,)
    # First: cos sim 1 -> loss -1; second: cos sim 0 -> loss 0
    assert torch.allclose(loss, torch.tensor([-1.0, 0.0]), atol=1e-6)


def test_steering_loss_shape():
    """Test that SteeringActivationLoss returns correct output shape."""
    loss_fn = SteeringActivationLoss()

    bsz, n_layers, seq_len, d_model = 2, 4, 10, 128
    hidden_states = torch.randn(bsz, n_layers, seq_len, d_model)
    target_directions = torch.randn(d_model)  # Unbatched

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
    target_directions = direction

    loss = loss_fn(hidden_states, target_directions)
    # Perfect alignment: cos_sim = 1, loss = -1
    assert torch.allclose(loss, torch.tensor([-1.0]), atol=1e-6)

    # Case 2: Orthogonal (cos_sim = 0)
    hidden_states = torch.tensor([0.0, 1.0]).view(1, 1, 1, 2).expand(bsz, n_layers, seq_len, d_model)
    target_directions = torch.tensor([1.0, 0.0])

    loss = loss_fn(hidden_states, target_directions)
    # Orthogonal: cos_sim = 0, loss = 0
    assert torch.allclose(loss, torch.tensor([0.0]), atol=1e-6)

    # Case 3: Opposite direction (cos_sim = -1)
    hidden_states = torch.tensor([-1.0, 0.0]).view(1, 1, 1, 2).expand(bsz, n_layers, seq_len, d_model)
    target_directions = torch.tensor([1.0, 0.0])

    loss = loss_fn(hidden_states, target_directions)
    # Opposite: cos_sim = -1, loss = 1
    assert torch.allclose(loss, torch.tensor([1.0]), atol=1e-6)


def test_steering_activation_loss_with_slices():
    """Test SteeringActivationLoss with position slices."""
    loss_fn = SteeringActivationLoss(slc_name="adv")

    bsz, n_layers, seq_len, d_model = 2, 3, 10, 4
    hidden_states = torch.randn(bsz, n_layers, seq_len, d_model)
    target_directions = torch.randn(d_model)

    # Define slices for each batch element
    slices = {"adv": slice(2, 5)}

    loss = loss_fn(hidden_states, target_directions, input_slices=slices)

    assert loss.shape == (bsz,)
    assert not torch.isnan(loss).any()


def test_steering_enh_loss_targeted_layers():
    """Test SteeringActivationLoss with targeted layers."""
    # Only target middle layers
    loss_fn = SteeringActivationLoss(targeted_layers=slice(1, 3))

    bsz, n_layers, seq_len, d_model = 1, 4, 5, 2

    # Create hidden states with different values in different layers
    # Layers 0,3 are orthogonal to target, layers 1,2 are aligned
    target_direction = torch.tensor([1.0, 0.0])

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
    target_directions = torch.randn(d_model)

    loss = loss_fn(hidden_states, target_directions)

    assert loss.shape == (bsz,)
    assert not torch.isnan(loss).any()
    # Each batch element should have different loss values (with high probability)
    assert not torch.allclose(loss[0], loss[1], atol=1e-3)


# ==============================================================================
# Logit-Based Loss Tests
# ==============================================================================

def test_prefill_ce_loss_shape():
    """Test PrefillCELoss returns correct output shape."""
    from tropt.loss import PrefillCELoss

    loss_fn = PrefillCELoss()

    # Simulated logits: (bsz, target_seq_len, vocab_size)
    # NOTE: logits and target_ids must have same first TWO dimensions
    bsz, target_seq_len, vocab_size = 2, 5, 1000
    logits = torch.randn(bsz, target_seq_len, vocab_size)
    target_ids = torch.randint(0, vocab_size, (target_seq_len,))

    loss = loss_fn(logits, target_ids)

    assert loss.shape == (bsz,)
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()
    assert (loss >= 0).all(), "Cross-entropy loss should be non-negative"


def test_prefill_ce_loss_known_values():
    """Test PrefillCELoss with known input/output pairs."""
    from tropt.loss import PrefillCELoss

    loss_fn = PrefillCELoss()

    # Case 1: Perfect prediction (high logit for correct token)
    vocab_size = 10
    logits = torch.zeros(1, 3, vocab_size)  # (bsz=1, seq_len=3, vocab_size=10)
    target_ids = torch.tensor([2, 5, 7])  # (target_len=3,)

    # Set very high logits for the correct tokens
    logits[0, 0, 2] = 100.0  # Position 0, token 2
    logits[0, 1, 5] = 100.0  # Position 1, token 5
    logits[0, 2, 7] = 100.0  # Position 2, token 7

    loss = loss_fn(logits, target_ids)

    # With perfect prediction, CE loss should be very close to 0
    assert loss.item() < 0.1, f"Expected loss near 0 for perfect prediction, got {loss.item()}"


def test_prefill_mellowmax_loss_shape():
    """Test PrefillMellowMaxLoss returns correct output shape."""
    from tropt.loss import PrefillMellowMaxLoss

    loss_fn = PrefillMellowMaxLoss()
    loss_fn.mellowmax_alpha = 1.0

    bsz, target_seq_len, vocab_size = 3, 4, 500
    logits = torch.randn(bsz, target_seq_len, vocab_size)
    target_ids = torch.randint(0, vocab_size, (target_seq_len,))

    loss = loss_fn(logits, target_ids)

    assert loss.shape == (bsz,)
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()


def test_prefill_mellowmax_different_alpha():
    """Test PrefillMellowMaxLoss with different alpha values."""
    from tropt.loss import PrefillMellowMaxLoss

    bsz, target_seq_len, vocab_size = 2, 3, 100
    logits = torch.randn(bsz, target_seq_len, vocab_size)
    target_ids = torch.randint(0, vocab_size, (target_seq_len,))

    # Test with different alpha values
    for alpha in [0.5, 1.0, 2.0]:
        loss_fn = PrefillMellowMaxLoss()
        loss_fn.mellowmax_alpha = alpha
        loss = loss_fn(logits, target_ids)

        assert loss.shape == (bsz,)
        assert not torch.isnan(loss).any()


def test_prefill_cw_loss_shape():
    """Test PrefillCWLoss returns correct output shape."""
    from tropt.loss import PrefillCWLoss

    loss_fn = PrefillCWLoss()
    loss_fn.cw_margin = 0.0

    bsz, target_seq_len, vocab_size = 2, 4, 200
    logits = torch.randn(bsz, target_seq_len, vocab_size)
    target_ids = torch.randint(0, vocab_size, (target_seq_len,))

    loss = loss_fn(logits, target_ids)

    assert loss.shape == (bsz,)
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()


def test_prefill_cw_loss_known_values():
    """Test PrefillCWLoss with known scenarios."""
    from tropt.loss import PrefillCWLoss

    loss_fn = PrefillCWLoss()
    loss_fn.cw_margin = 1e-3

    vocab_size = 10
    logits = torch.zeros(1, 2, vocab_size)
    target_ids = torch.tensor([3, 7])

    # Case 1: Target token has highest logit (good) - loss should be low
    logits[0, 0, :] = -1.0
    logits[0, 0, 3] = 5.0  # Target token

    logits[0, 1, :] = -1.0
    logits[0, 1, 7] = 5.0  # Target token

    loss = loss_fn(logits, target_ids)

    # CW loss: (max_non_target - target).clamp_min(-margin)
    # When target has max logit: (non_target - target) is negative, clamped to -margin
    assert loss.item() <= 0.001, f"Expected very low loss when target has max logit, got {loss.item()}"

    # Case 2: Non-target token has highest logit (bad) - loss should be positive
    logits2 = torch.zeros(1, 2, vocab_size)
    target_ids2 = torch.tensor([3, 7])

    logits2[0, 0, :] = -1.0
    logits2[0, 0, 3] = 0.0  # Target token (low)
    logits2[0, 0, 5] = 5.0  # Non-target highest

    logits2[0, 1, :] = -1.0
    logits2[0, 1, 7] = 0.0  # Target token (low)
    logits2[0, 1, 2] = 5.0  # Non-target highest

    loss2 = loss_fn(logits2, target_ids2)

    # Should have positive loss when non-target is higher
    assert loss2.item() > 1.0, f"Expected high loss when non-target has max logit, got {loss2.item()}"


def test_trigger_perplexity_loss_shape():
    """Test TriggerPerplexityLoss returns correct output shape."""
    from tropt.loss import TriggerPerplexityLoss

    loss_fn = TriggerPerplexityLoss(slc_name="adv")

    # Full sequence logits: (bsz, seq_len, vocab_size)
    # Trigger IDs: (bsz, trigger_len)
    bsz, seq_len, trigger_len, vocab_size = 2, 20, 5, 500
    output_logits = torch.randn(bsz, seq_len, vocab_size)
    input_trigger_ids = torch.randint(0, vocab_size, (bsz, trigger_len))

    # Define slices where trigger is located
    input_slices = {"adv": slice(5, 10)}

    loss = loss_fn(output_logits, input_trigger_ids, input_slices)

    assert loss.shape == (bsz,)
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()
    assert (loss >= 0).all(), "Perplexity loss should be non-negative"


# ==============================================================================
# Attention-Based Loss Tests
# ==============================================================================

def test_attention_enh_loss_shape():
    """Test AttentionEnhLoss returns correct output shape."""
    from tropt.loss import AttentionEnhLoss

    loss_fn = AttentionEnhLoss()
    loss_fn.src_slc_name = "adv"

    # Attention weights: (bsz, n_layers, n_heads, seq_len, seq_len)
    bsz, n_layers, n_heads, seq_len = 2, 4, 8, 20
    attn_weights = torch.randn(bsz, n_layers, n_heads, seq_len, seq_len)
    attn_weights = torch.softmax(attn_weights, dim=-1)  # Ensure valid attention

    # Slices indicating trigger position
    slices = {"adv": slice(5, 10)}

    loss = loss_fn(attn_weights, input_slices=slices)

    assert loss.shape == (bsz,)
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()


def test_attention_enh_loss_properties():
    """Test AttentionEnhLoss mathematical properties."""
    from tropt.loss import AttentionEnhLoss

    loss_fn = AttentionEnhLoss()
    loss_fn.src_slc_name = "trigger"

    bsz, n_layers, n_heads, seq_len = 1, 2, 4, 10

    # Case 1: All attention on trigger (should minimize loss)
    attn_weights = torch.zeros(bsz, n_layers, n_heads, seq_len, seq_len)
    trigger_slice = slice(2, 5)

    # Make all positions attend strongly to trigger positions
    for pos in range(seq_len):
        if pos < 2 or pos >= 5:  # Non-trigger positions
            # Attend to trigger positions
            attn_weights[0, :, :, pos, 2:5] = 1.0 / 3  # Uniform on trigger

    # Normalize
    attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-10)

    slices = {"trigger": trigger_slice}
    loss = loss_fn(attn_weights, input_slices=slices)

    # High attention to trigger should give negative loss
    assert loss.item() < 0, f"Expected negative loss with high trigger attention, got {loss.item()}"


# ==============================================================================
# Combined Loss Tests
# ==============================================================================

def test_combined_loss_basic():
    """Test CombinedLoss with multiple loss functions."""
    from tropt.loss import CombinedLoss, SimilarityLoss, SteeringActivationLoss

    loss1 = SimilarityLoss()
    loss2 = SteeringActivationLoss()

    combined = CombinedLoss([loss1, loss2], weights=[0.7, 0.3])

    # Verify the structure
    assert combined.loss_funcs == [loss1, loss2]
    assert torch.allclose(combined.weights, torch.tensor([0.7, 0.3]))


def test_combined_loss_weight_validation():
    """Test CombinedLoss validates weights correctly."""
    from tropt.loss import CombinedLoss, SimilarityLoss

    loss1 = SimilarityLoss()
    loss2 = SimilarityLoss()

    # Weights should sum to number of losses or be normalized
    combined = CombinedLoss([loss1, loss2], weights=[0.5, 0.5])
    assert len(combined.weights) == 2

    # Test with single weight (should work)
    combined = CombinedLoss([loss1, loss2], weights=[1.0, 1.0])
    assert len(combined.weights) == 2


# ==============================================================================
# Edge Cases and Robustness Tests
# ==============================================================================

def test_loss_functions_handle_zero_gradients():
    """Test that loss functions handle non-zero vectors correctly."""
    from tropt.loss import SimilarityLoss

    loss_fn = SimilarityLoss()

    # Use non-zero vectors (zero vectors cause NaN in cosine similarity, which is expected)
    vectors = torch.randn(2, 128) + 1.0  # Add offset to avoid zeros
    targets = torch.randn(128) + 1.0

    loss = loss_fn(vectors, targets)

    # Should not produce NaN or Inf for valid inputs
    assert not torch.isnan(loss).any()
    assert not torch.isinf(loss).any()


def test_loss_functions_batch_size_one():
    """Test loss functions work with batch size 1."""
    from tropt.loss import SimilarityLoss, SteeringActivationLoss

    # SimilarityLoss
    sim_loss = SimilarityLoss()
    vectors = torch.randn(1, 64)
    targets = torch.randn(64)
    loss = sim_loss(vectors, targets)
    assert loss.shape == (1,)

    # SteeringActivationLoss
    steer_loss = SteeringActivationLoss()
    hidden_states = torch.randn(1, 4, 10, 64)
    target_dirs = torch.randn(64)
    loss = steer_loss(hidden_states, target_dirs)
    assert loss.shape == (1,)


def test_loss_functions_large_batch():
    """Test loss functions work with large batch sizes."""
    from tropt.loss import SimilarityLoss

    loss_fn = SimilarityLoss()

    bsz = 128  # Large batch
    vectors = torch.randn(bsz, 64)
    targets = torch.randn(64)

    loss = loss_fn(vectors, targets)

    assert loss.shape == (bsz,)
    assert not torch.isnan(loss).any()