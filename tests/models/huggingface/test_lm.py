import pytest
import torch

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER
from tropt.loss.base import PrefillCELoss
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

# --- HFTokenInputsManager Tests ---
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

    res = inputs.get_triggered_inputs(trigger_ids=trigger_ids)

    assert {"inputs_embeds", "attention_mask", "targets"}.issubset(res.keys())

    # inputs_embeds: (n_messages, n_candidates, seq_len, embd_dim)
    assert res["inputs_embeds"].shape[0] == n_messages
    assert res["inputs_embeds"].shape[1] == n_candidates

    # attention mask: (n_messages, n_candidates, seq_len)
    assert res["attention_mask"].shape[0] == n_messages
    assert res["attention_mask"].shape[1] == n_candidates

    # Check targets expansion
    assert "targets" in res
    assert "target_outputs_toks" in res["targets"]

    tgt = res["targets"]["target_outputs_toks"]
    assert isinstance(tgt, list)
    assert len(tgt) == n_messages
    assert tgt[0].shape[0] == n_candidates

    tgt_slices = res["targets"]["slices"]
    assert isinstance(tgt, list) and len(tgt_slices) == n_messages
    tgt_slices_msg1 = tgt_slices[0]
    assert isinstance(tgt_slices_msg1, list) and len(tgt_slices_msg1) == n_candidates
    tgt_slices_msg1_0 = tgt_slices_msg1[0]
    assert isinstance(tgt_slices_msg1_0, dict)
    assert isinstance(tgt_slices_msg1_0['adv'], slice)
    assert tgt_slices_msg1_0['adv'].stop - tgt_slices_msg1_0['adv'].start == trigger_len

def test_inputs_manager_batch_slice(lm_model):
    texts = [f"A {OPTIMIZED_TRIGGER_PLACEHOLDER} B"]
    targets = {"target_outputs": ["T1"]}
    inputs, _ = lm_model.prepare_token_inputs(texts, targets)
    
    n_candidates, trigger_len = 4, 3
    trigger_ids = torch.randint(0, 100, (n_candidates, trigger_len))
    
    # Slice first 2
    batch_slice = slice(0, 2)
    batch_size = batch_slice.stop - batch_slice.start
    res = inputs.get_triggered_inputs(trigger_ids=trigger_ids, batch_slice=batch_slice)
    assert res["inputs_embeds"].shape[1] == batch_size
    assert res["targets"]["target_outputs_toks"][0].shape[0] == batch_size

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
