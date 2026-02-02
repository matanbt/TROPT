import pytest
import torch

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER, SliceKey, Targets
from tropt.loss.base import PrefillCELoss, SteeringActivationLoss
from tropt.models.huggingface.lm import LMHFModel


@pytest.fixture(scope="module")
def lm_model():
    model_name = "google/gemma-3-270m-it"
    try:
        model = LMHFModel(model_name, device="cpu")
        return model
    except Exception as e:
        pytest.skip(f"Could not load model {model_name}: {e}")

def test_lm_init(lm_model):
    assert lm_model is not None
    assert lm_model.model is not None
    assert lm_model.tokenizer is not None

def test_lm_prepare_token_inputs(lm_model):
    texts = [f"Hello {OPTIMIZED_TRIGGER_PLACEHOLDER} World"]
    targets = Targets(target_response_strs=["Sure, the world is hello"])

    inputs, trigger_ids = lm_model.prepare_token_inputs(texts, targets, initial_trigger="test....")

    assert inputs is not None
    assert isinstance(trigger_ids, torch.Tensor)
    assert trigger_ids.ndim == 2  # (n_messages, trigger_length)
    assert trigger_ids.shape[0] == 1  # n_messages
    assert inputs.n_messages == 1

def test_lm_prepare_token_multi_inputs(lm_model):
    texts = [f"Hello {OPTIMIZED_TRIGGER_PLACEHOLDER} World", f"Hi! Be honest. {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    targets = Targets(target_response_strs=["Sure, the world is hello", "Sure."])

    inputs, trigger_ids = lm_model.prepare_token_inputs(texts, targets, initial_trigger="test....")

    assert inputs is not None
    assert isinstance(trigger_ids, torch.Tensor)
    assert trigger_ids.ndim == 2 and trigger_ids.shape[0] == 1
    assert inputs.n_messages == 2

    targets = Targets(target_response_strs=["Target output", "Another target output"])
    inputs, trigger_ids = lm_model.prepare_token_inputs(texts, targets)
    
    # sample 2 cand triggers
    n_candidates = 2
    candidate_ids = torch.randint(0, lm_model.tokenizer.vocab_size, (n_candidates, trigger_ids.shape[1]))
    loss_fn = PrefillCELoss()

    losses = lm_model.compute_loss_from_tokens(candidate_ids, inputs, loss_fn)

    assert isinstance(losses, torch.Tensor)
    assert losses.shape == (n_candidates,)
    assert not torch.isnan(losses).any()

def test_lm_compute_grad(lm_model):
    texts = [f"Hello {OPTIMIZED_TRIGGER_PLACEHOLDER} World"]
    targets = Targets(target_response_strs=["Target output"])
    inputs, trigger_ids = lm_model.prepare_token_inputs(texts, targets)

    n_candidates = 2
    candidate_ids = torch.randint(0, lm_model.tokenizer.vocab_size, (n_candidates, trigger_ids.shape[1]))
    loss_fn = PrefillCELoss()

    grads = lm_model.compute_grad_from_tokens(inputs=inputs, loss_func=loss_fn, candidate_trigger_ids=candidate_ids)

    assert isinstance(grads, torch.Tensor)
    assert grads.shape == (n_candidates, trigger_ids.shape[1], lm_model.tokenizer.vocab_size)
    assert not torch.isnan(grads).any()

# --- _HFTokenInputsManager Tests ---
# We use the LM model to create the manager instance for testing

def test_inputs_manager_initialization(lm_model):
    texts = [f"A {OPTIMIZED_TRIGGER_PLACEHOLDER} B", f"C {OPTIMIZED_TRIGGER_PLACEHOLDER} D"]
    targets = Targets(target_response_strs=["T1", "T2"])
    inputs, _ = lm_model.prepare_token_inputs(texts, targets)

    n_messages = len(texts)
    assert inputs.n_messages == n_messages
    assert len(inputs.before_ids) == n_messages
    assert len(inputs.after_ids) == n_messages
    assert inputs.vocab_size == lm_model.tokenizer.vocab_size
    assert len(inputs.targets.target_response_toks) == n_messages

def test_inputs_manager_get_triggered_inputs(lm_model):
    texts = [f"A {OPTIMIZED_TRIGGER_PLACEHOLDER} B", f"C {OPTIMIZED_TRIGGER_PLACEHOLDER} D", f"E {OPTIMIZED_TRIGGER_PLACEHOLDER} F"]
    targets = Targets(target_response_strs=["T1", "T2", "T3"])
    n_messages = len(texts)

    inputs, trigger_ids = lm_model.prepare_token_inputs(texts, targets)

    n_candidates, trigger_len = 2, trigger_ids.shape[1]
    trigger_ids = torch.randint(0, 100, (n_candidates, trigger_len))

    # Loop over messages and test each one individually
    for message_idx in range(n_messages):
        res = inputs.get_triggered_inputs(trigger_ids=trigger_ids, chosen_message_idx=message_idx)

        assert res.input_embeds is not None
        assert res.input_attention_mask is not None
        assert res.targets is not None

        # inputs_embeds: (n_candidates, seq_len, embd_dim) -- message dim removed
        assert res.input_embeds.dim() == 3
        assert res.input_embeds.shape[0] == n_candidates

        # attention mask: (n_candidates, seq_len)
        assert res.input_attention_mask.dim() == 2
        assert res.input_attention_mask.shape[0] == n_candidates

        # Check targets expansion
        assert res.targets is not None
        assert res.targets.target_response_toks is not None

        tgt = res.targets.target_response_toks
        assert isinstance(tgt, torch.Tensor)

        tgt_slices = res.input_slices
        assert isinstance(tgt_slices, dict)
        assert isinstance(tgt_slices[SliceKey.TRIGGER], slice)
        assert tgt_slices[SliceKey.TRIGGER].stop - tgt_slices[SliceKey.TRIGGER].start == trigger_len

def test_inputs_manager_chosen_message(lm_model):
    texts = [f"A {OPTIMIZED_TRIGGER_PLACEHOLDER} B", f"C {OPTIMIZED_TRIGGER_PLACEHOLDER} D"]
    targets = Targets(target_response_strs=["T1", "T2"])
    inputs, _ = lm_model.prepare_token_inputs(texts, targets)

    n_candidates, trigger_len = 2, 3
    trigger_ids = torch.randint(0, 100, (n_candidates, trigger_len))

    # Choose message 1
    res = inputs.get_triggered_inputs(trigger_ids=trigger_ids, chosen_message_idx=1)

    # inputs_embeds: (n_candidates, seq_len, embd_dim) -- message dim removed?
    assert res.input_embeds.dim() == 3
    assert res.input_embeds.shape[0] == n_candidates

    # Targets should be for single message now
    # get_message_from_batched_targets selects the item at chosen_message_idx
    tgt = res.targets.target_response_toks
    assert isinstance(tgt, torch.Tensor)

def test_lm_steering_loss(lm_model):
    """Test SteeringEnhLoss integration with LM model."""
    texts = [f"Hello {OPTIMIZED_TRIGGER_PLACEHOLDER} World"]

    # Create a random target direction
    d_model = lm_model.model.config.hidden_size
    target_direction = torch.randn(1, d_model)

    targets = Targets(target_directions=target_direction)
    inputs, trigger_ids = lm_model.prepare_token_inputs(texts, targets)

    # Sample 2 candidate triggers
    n_candidates = 2
    candidate_ids = torch.randint(0, lm_model.tokenizer.vocab_size, (n_candidates, trigger_ids.shape[1]))

    # Create steering loss targeting middle layers
    loss_fn = SteeringActivationLoss(targeted_layers=slice(5, 10))

    # Compute loss
    losses = lm_model.compute_loss_from_tokens(candidate_ids, inputs, loss_fn)

    assert isinstance(losses, torch.Tensor)
    assert losses.shape == (n_candidates,)
    assert not torch.isnan(losses).any()
    assert not torch.isinf(losses).any()

def test_lm_steering_loss_multi_message(lm_model):
    """Test SteeringActivationLoss with multiple messages."""
    texts = [
        f"Hello {OPTIMIZED_TRIGGER_PLACEHOLDER} World",
        f"Hi! {OPTIMIZED_TRIGGER_PLACEHOLDER}"
    ]

    # Create target directions for each message
    d_model = lm_model.model.config.hidden_size
    target_directions = torch.randn(2, d_model)

    targets = Targets(target_directions=target_directions)
    inputs, trigger_ids = lm_model.prepare_token_inputs(texts, targets)

    # Sample candidates
    n_candidates = 3
    candidate_ids = torch.randint(0, lm_model.tokenizer.vocab_size, (n_candidates, trigger_ids.shape[1]))

    loss_fn = SteeringActivationLoss()

    losses = lm_model.compute_loss_from_tokens(candidate_ids, inputs, loss_fn)

    assert isinstance(losses, torch.Tensor)
    assert losses.shape == (n_candidates,)
    assert not torch.isnan(losses).any()
