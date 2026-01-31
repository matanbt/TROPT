import pytest
import torch

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER
from tropt.loss.base import SimilarityLoss
from tropt.models.huggingface.encoder import EncoderHFModel

# TODO review & consider dropping the mocks

@pytest.fixture(scope="module")
def encoder_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        model = EncoderHFModel(model_name, device="cpu")
        return model
    except Exception as e:
        pytest.skip(f"Could not load model {model_name}: {e}")

def test_encoder_init(encoder_model):
    assert encoder_model is not None
    assert encoder_model.d_model == 384  # specific to all-MiniLM-L6-v2
    assert encoder_model.model is not None
    assert encoder_model.tokenizer is not None

def test_encoder_prepare_token_inputs(encoder_model):
    texts = [f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    targets = {"target_vectors": torch.randn(1, encoder_model.d_model)}

    inputs, trigger_ids = encoder_model.prepare_token_inputs(texts, targets, initial_trigger="init")

    assert inputs is not None
    assert isinstance(trigger_ids, torch.Tensor)
    assert trigger_ids.ndim == 2  # (n_messages, trigger_length)
    assert trigger_ids.shape[0] == 1  # n_messages (shared trigger)
    assert inputs.n_messages == 1

def test_encoder_prepare_token_multi_inputs(encoder_model):
    texts = [f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}", f"Second: {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    targets = {"target_vectors": torch.randn(2, encoder_model.d_model)}

    inputs, trigger_ids = encoder_model.prepare_token_inputs(texts, targets, initial_trigger="init")

    assert inputs is not None
    assert isinstance(trigger_ids, torch.Tensor)
    assert trigger_ids.ndim == 2 and trigger_ids.shape[0] == 1
    assert inputs.n_messages == 2

def test_encoder_compute_loss(encoder_model):
    texts = [f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    targets = {"target_vectors": torch.randn(1, encoder_model.d_model)}
    inputs, trigger_ids = encoder_model.prepare_token_inputs(texts, targets)

    n_candidates = 2
    # trigger_ids from prepare_token_inputs is (1, len)
    candidate_ids = torch.randint(0, encoder_model.tokenizer.vocab_size, (n_candidates, trigger_ids.shape[1]))
    loss_fn = SimilarityLoss()

    losses = encoder_model.compute_loss_from_tokens(candidate_ids, inputs, loss_fn)

    assert isinstance(losses, torch.Tensor)
    assert losses.shape == (n_candidates,)
    assert not torch.isnan(losses).any()

def test_encoder_compute_grad(encoder_model):
    texts = [f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    targets = {"target_vectors": torch.randn(1, encoder_model.d_model)}
    inputs, trigger_ids = encoder_model.prepare_token_inputs(texts, targets)

    n_candidates = 2
    candidate_ids = torch.randint(0, encoder_model.tokenizer.vocab_size, (n_candidates, trigger_ids.shape[1]))
    loss_fn = SimilarityLoss()

    grads = encoder_model.compute_grad_from_tokens(candidate_ids, inputs, loss_fn)

    assert isinstance(grads, torch.Tensor)
    assert grads.shape == (n_candidates, trigger_ids.shape[1], encoder_model.tokenizer.vocab_size)
    assert not torch.isnan(grads).any()

# --- _HFTokenInputsManager Tests (Encoder Context) ---

def test_inputs_manager_initialization(encoder_model):
    texts = [f"A {OPTIMIZED_TRIGGER_PLACEHOLDER} B", f"C {OPTIMIZED_TRIGGER_PLACEHOLDER} D"]
    targets = {"target_vectors": torch.randn(2, encoder_model.d_model)}
    inputs, _ = encoder_model.prepare_token_inputs(texts, targets)

    n_messages = len(texts)
    assert inputs.n_messages == n_messages
    assert len(inputs.before_ids) == n_messages
    assert len(inputs.after_ids) == n_messages
    assert inputs.vocab_size == encoder_model.tokenizer.vocab_size
    # Check target vectors are present
    assert "target_vectors" in inputs.targets
    assert inputs.targets["target_vectors"].shape == (n_messages, encoder_model.d_model)

def test_inputs_manager_get_triggered_inputs(encoder_model):
    texts = [f"A {OPTIMIZED_TRIGGER_PLACEHOLDER} B", f"C {OPTIMIZED_TRIGGER_PLACEHOLDER} D"]
    targets = {"target_vectors": torch.randn(2, encoder_model.d_model)}
    n_messages = len(texts)

    inputs, trigger_ids = encoder_model.prepare_token_inputs(texts, targets)

    n_candidates = 2
    candidate_trigger_ids = torch.randint(0, 100, (n_candidates, trigger_ids.shape[1]))

    # Loop over messages and test each one individually
    for message_idx in range(n_messages):
        res = inputs.get_triggered_inputs(trigger_ids=candidate_trigger_ids, chosen_message_idx=message_idx)

        assert {"inputs_embeds", "attention_mask", "targets"}.issubset(res.keys())

        # inputs_embeds: (n_candidates, seq_len, embd_dim) -- message dim removed
        assert res["inputs_embeds"].dim() == 3
        assert res["inputs_embeds"].shape[0] == n_candidates

        # attention mask: (n_candidates, seq_len)
        assert res["attention_mask"].dim() == 2
        assert res["attention_mask"].shape[0] == n_candidates

        # Check targets expansion
        assert "targets" in res
        assert "target_vectors" in res["targets"]

        from tropt.models.inputs import SliceKey

        tgt_slices_0 = res["targets"]["slices"][0]  # candidate 0
        assert isinstance(tgt_slices_0, dict)
        assert isinstance(tgt_slices_0[SliceKey.TRIGGER], slice)
        # Trigger slice length should match the actual trigger length (default is 20 tokens)
        assert tgt_slices_0[SliceKey.TRIGGER].stop - tgt_slices_0[SliceKey.TRIGGER].start == trigger_ids.shape[1]

def test_inputs_manager_chosen_message(encoder_model):
    texts = [f"A {OPTIMIZED_TRIGGER_PLACEHOLDER} B", f"C {OPTIMIZED_TRIGGER_PLACEHOLDER} D"]
    targets = {"target_vectors": torch.randn(2, encoder_model.d_model)}
    inputs, trigger_ids = encoder_model.prepare_token_inputs(texts, targets)

    n_candidates = 2
    candidate_trigger_ids = torch.randint(0, 100, (n_candidates, trigger_ids.shape[1]))

    # Choose message 1
    res = inputs.get_triggered_inputs(trigger_ids=candidate_trigger_ids, chosen_message_idx=1)

    # inputs_embeds: (n_candidates, seq_len, embd_dim) -- message dim removed
    assert res["inputs_embeds"].dim() == 3
    assert res["inputs_embeds"].shape[0] == n_candidates

    # Targets should be for single message now
    tgt_vecs = res["targets"]["target_vectors"]
    # Should be (n_candidates, d_model)
    assert isinstance(tgt_vecs, torch.Tensor)
    assert tgt_vecs.shape[0] == n_candidates
    assert tgt_vecs.shape[1] == encoder_model.d_model