import pytest
import torch

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER
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
    targets = {"target_outputs": ["Sure, the world is hello"]}

    inputs, trigger_ids = lm_model.prepare_token_inputs(texts, targets, initial_trigger="test....")

    assert inputs is not None
    assert isinstance(trigger_ids, torch.Tensor)
    assert trigger_ids.ndim == 2  # (n_messages, trigger_length)
    assert trigger_ids.shape[0] == 1  # n_messages
    assert inputs.n_messages == 1

def test_lm_prepare_token_multi_inputs(lm_model):
    texts = [f"Hello {OPTIMIZED_TRIGGER_PLACEHOLDER} World", f"Hi! Be honest. {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    targets = {"target_outputs": ["Sure, the world is hello", "Sure."]}

    inputs, trigger_ids = lm_model.prepare_token_inputs(texts, targets, initial_trigger="test....")

    assert inputs is not None
    assert isinstance(trigger_ids, torch.Tensor)
    assert trigger_ids.ndim == 2 and trigger_ids.shape[0] == 1
    assert inputs.n_messages == 2

def test_lm_compute_loss(lm_model):
    texts = [f"Hello {OPTIMIZED_TRIGGER_PLACEHOLDER} World"]
    targets = {"target_outputs": ["Target output"]}
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
    targets = {"target_outputs": ["Target output"]}
    inputs, trigger_ids = lm_model.prepare_token_inputs(texts, targets)

    n_candidates = 2
    candidate_ids = torch.randint(0, lm_model.tokenizer.vocab_size, (n_candidates, trigger_ids.shape[1]))
    loss_fn = PrefillCELoss()

    grads = lm_model.compute_grad_from_tokens(candidate_ids, inputs, loss_fn)

    assert isinstance(grads, torch.Tensor)
    assert grads.shape == (n_candidates, trigger_ids.shape[1], lm_model.tokenizer.vocab_size)
    assert not torch.isnan(grads).any()

# --- _HFTokenInputsManager Tests ---
# We use the LM model to create the manager instance for testing

def test_inputs_manager_initialization(lm_model):
    texts = [f"A {OPTIMIZED_TRIGGER_PLACEHOLDER} B", f"C {OPTIMIZED_TRIGGER_PLACEHOLDER} D"]
    targets = {"target_outputs": ["T1", "T2"]}
    inputs, _ = lm_model.prepare_token_inputs(texts, targets)

    n_messages = len(texts)
    assert inputs.n_messages == n_messages
    assert len(inputs.before_ids) == n_messages
    assert len(inputs.after_ids) == n_messages
    assert inputs.vocab_size == lm_model.tokenizer.vocab_size
    assert len(inputs.targets["target_outputs_toks"]) == n_messages

def test_inputs_manager_get_triggered_inputs(lm_model):
    texts = [f"A {OPTIMIZED_TRIGGER_PLACEHOLDER} B", f"C {OPTIMIZED_TRIGGER_PLACEHOLDER} D", f"E {OPTIMIZED_TRIGGER_PLACEHOLDER} F"]
    targets = {"target_outputs": ["T1", "T2", "T3"]}
    n_messages = len(texts)

    inputs, trigger_ids = lm_model.prepare_token_inputs(texts, targets)

    n_candidates, trigger_len = 2, trigger_ids.shape[1]
    trigger_ids = torch.randint(0, 100, (n_candidates, trigger_len))

    # Loop over messages and test each one individually
    for message_idx in range(n_messages):
        res = inputs.get_triggered_inputs(trigger_ids=trigger_ids, chosen_message_idx=message_idx)

        assert {"inputs_embeds", "attention_mask", "targets"}.issubset(res.keys())

        # inputs_embeds: (n_candidates, seq_len, embd_dim) -- message dim removed
        assert res["inputs_embeds"].dim() == 3
        assert res["inputs_embeds"].shape[0] == n_candidates

        # attention mask: (n_candidates, seq_len)
        assert res["attention_mask"].dim() == 2
        assert res["attention_mask"].shape[0] == n_candidates

        # Check targets expansion
        assert "targets" in res
        assert "target_outputs_toks" in res["targets"]

        tgt = res["targets"]["target_outputs_toks"]
        assert isinstance(tgt, torch.Tensor)
        assert tgt.shape[0] == n_candidates

        tgt_slices = res["targets"]["slices"]
        assert isinstance(tgt_slices, list) and len(tgt_slices) == n_candidates
        from tropt.models.inputs import SliceKey

        tgt_slices_0 = tgt_slices[0]
        assert isinstance(tgt_slices_0, dict)
        assert isinstance(tgt_slices_0[SliceKey.TRIGGER], slice)
        assert tgt_slices_0[SliceKey.TRIGGER].stop - tgt_slices_0[SliceKey.TRIGGER].start == trigger_len

def test_inputs_manager_chosen_message(lm_model):
    texts = [f"A {OPTIMIZED_TRIGGER_PLACEHOLDER} B", f"C {OPTIMIZED_TRIGGER_PLACEHOLDER} D"]
    targets = {"target_outputs": ["T1", "T2"]}
    inputs, _ = lm_model.prepare_token_inputs(texts, targets)

    n_candidates, trigger_len = 2, 3
    trigger_ids = torch.randint(0, 100, (n_candidates, trigger_len))

    # Choose message 1
    res = inputs.get_triggered_inputs(trigger_ids=trigger_ids, chosen_message_idx=1)

    # inputs_embeds: (n_candidates, seq_len, embd_dim) -- message dim removed?
    assert res["inputs_embeds"].dim() == 3
    assert res["inputs_embeds"].shape[0] == n_candidates

    # Targets should be for single message now
    # get_message_from_batched_targets selects the item at chosen_message_idx
    tgt = res["targets"]["target_outputs_toks"]
    assert isinstance(tgt, torch.Tensor)
    assert tgt.shape[0] == n_candidates

def test_lm_steering_loss(lm_model):
    """Test SteeringEnhLoss integration with LM model."""
    texts = [f"Hello {OPTIMIZED_TRIGGER_PLACEHOLDER} World"]

    # Create a random target direction
    d_model = lm_model.model.config.hidden_size
    target_direction = torch.randn(1, d_model)

    targets = {"target_directions": target_direction}
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

    targets = {"target_directions": target_directions}
    inputs, trigger_ids = lm_model.prepare_token_inputs(texts, targets)

    # Sample candidates
    n_candidates = 3
    candidate_ids = torch.randint(0, lm_model.tokenizer.vocab_size, (n_candidates, trigger_ids.shape[1]))

    loss_fn = SteeringActivationLoss()

    losses = lm_model.compute_loss_from_tokens(candidate_ids, inputs, loss_fn)

    assert isinstance(losses, torch.Tensor)
    assert losses.shape == (n_candidates,)
    assert not torch.isnan(losses).any()
