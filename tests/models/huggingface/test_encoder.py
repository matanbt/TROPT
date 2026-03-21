import pytest
import torch

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER, Targets
from tropt.loss import SimilarityLoss
from tropt.model.huggingface.encoder import EncoderHFModel

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
    assert encoder_model._model is not None
    assert encoder_model.tokenizer is not None

def test_encoder_set_token_inputs(encoder_model):
    texts = [f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    targets = Targets(target_vectors=torch.randn(1, encoder_model.d_model))

    encoder_model.set_inputs_from_tokens(texts, targets)
    trigger_ids = encoder_model.tokenizer(
        "init", add_special_tokens=False, return_tensors="pt"
    )["input_ids"]

    assert encoder_model._token_input_manager is not None
    assert isinstance(trigger_ids, torch.Tensor)
    assert trigger_ids.ndim == 2  # (1, trigger_length)
    assert trigger_ids.shape[0] == 1  # shared trigger
    assert encoder_model._token_input_manager.n_templates == 1

    encoder_model.reset_inputs_from_tokens()

def test_encoder_set_token_multi_inputs(encoder_model):
    texts = [f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}", f"Second: {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    targets = Targets(target_vectors=torch.randn(2, encoder_model.d_model))

    encoder_model.set_inputs_from_tokens(texts, targets)
    trigger_ids = encoder_model.tokenizer(
        "init", add_special_tokens=False, return_tensors="pt"
    )["input_ids"]

    assert encoder_model._token_input_manager is not None
    assert isinstance(trigger_ids, torch.Tensor)
    assert trigger_ids.ndim == 2 and trigger_ids.shape[0] == 1
    assert encoder_model._token_input_manager.n_templates == 2

    encoder_model.reset_inputs_from_tokens()

def test_encoder_compute_loss(encoder_model):
    texts = [f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    targets = Targets(target_vectors=torch.randn(1, encoder_model.d_model))
    encoder_model.set_inputs_from_tokens(texts, targets)
    trigger_ids = encoder_model.tokenizer(
        "init", add_special_tokens=False, return_tensors="pt"
    )["input_ids"]

    n_candidates = 2
    # trigger_ids is (1, len)
    candidate_ids = torch.randint(0, encoder_model.tokenizer.vocab_size, (n_candidates, trigger_ids.shape[1]))
    loss_fn = SimilarityLoss()

    losses = encoder_model.compute_loss_from_tokens(candidate_ids, loss_fn)

    assert isinstance(losses, torch.Tensor)
    assert losses.shape == (n_candidates,)
    assert not torch.isnan(losses).any()

    encoder_model.reset_inputs_from_tokens()

def test_encoder_compute_grad(encoder_model):
    texts = [f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    targets = Targets(target_vectors=torch.randn(1, encoder_model.d_model))
    encoder_model.set_inputs_from_tokens(texts, targets)
    trigger_ids = encoder_model.tokenizer(
        "init", add_special_tokens=False, return_tensors="pt"
    )["input_ids"]

    n_candidates = 2
    candidate_ids = torch.randint(0, encoder_model.tokenizer.vocab_size, (n_candidates, trigger_ids.shape[1]))
    loss_fn = SimilarityLoss()

    grads = encoder_model.compute_grad_from_tokens(loss_func=loss_fn, candidate_trigger_ids=candidate_ids)

    assert isinstance(grads, torch.Tensor)
    assert grads.shape == (n_candidates, trigger_ids.shape[1], encoder_model.tokenizer.vocab_size)
    assert not torch.isnan(grads).any()

    encoder_model.reset_inputs_from_tokens()

# --- _HFTokenInputManager Tests (Encoder Context) ---

def test_token_input_manager_initialization(encoder_model):
    texts = [f"A {OPTIMIZED_TRIGGER_PLACEHOLDER} B", f"C {OPTIMIZED_TRIGGER_PLACEHOLDER} D"]
    targets = Targets(target_vectors=torch.randn(2, encoder_model.d_model))
    encoder_model.set_inputs_from_tokens(texts, targets)
    inputs = encoder_model._token_input_manager

    n_templates = len(texts)
    assert inputs.n_templates == n_templates
    assert len(inputs.before_ids) == n_templates
    assert len(inputs.after_ids) == n_templates
    assert inputs.vocab_size == encoder_model.tokenizer.vocab_size
    # Check target vectors are present
    assert inputs.targets.target_vectors is not None
    assert inputs.targets.target_vectors.shape == (n_templates, encoder_model.d_model)

    encoder_model.reset_inputs_from_tokens()

def test_token_input_manager_get_triggered_inputs(encoder_model):
    texts = [f"A {OPTIMIZED_TRIGGER_PLACEHOLDER} B", f"C {OPTIMIZED_TRIGGER_PLACEHOLDER} D"]
    targets = Targets(target_vectors=torch.randn(2, encoder_model.d_model))
    n_templates = len(texts)

    encoder_model.set_inputs_from_tokens(texts, targets)
    inputs = encoder_model._token_input_manager
    trigger_ids = encoder_model.tokenizer(
        "init", add_special_tokens=False, return_tensors="pt"
    )["input_ids"]

    n_candidates = 2
    candidate_trigger_ids = torch.randint(0, 100, (n_candidates, trigger_ids.shape[1]))

    # Loop over messages and test each one individually
    for template_idx in range(n_templates):
        res = inputs.get_triggered_inputs(trigger_ids=candidate_trigger_ids, chosen_template_idx=template_idx)

        # Check that key attributes are present
        assert res.input_embeds is not None
        assert res.input_attention_mask is not None
        assert res.message_targets is not None

        # inputs_embeds: (n_candidates, seq_len, embd_dim) -- message dim removed
        assert res.input_embeds.dim() == 3
        assert res.input_embeds.shape[0] == n_candidates

        # attention mask: (n_candidates, seq_len)
        assert res.input_attention_mask.dim() == 2
        assert res.input_attention_mask.shape[0] == n_candidates

        # Check targets expansion
        assert res.message_targets.target_vectors is not None

        from tropt.common import SliceKey

        assert res.input_slices is not None
        assert isinstance(res.input_slices, dict)
        assert isinstance(res.input_slices[SliceKey.TRIGGER], slice)
        # Trigger slice length should match the actual trigger length (default is 20 tokens)
        assert res.input_slices[SliceKey.TRIGGER].stop - res.input_slices[SliceKey.TRIGGER].start == trigger_ids.shape[1]

    encoder_model.reset_inputs_from_tokens()

def test_token_input_manager_chosen_message(encoder_model):
    texts = [f"A {OPTIMIZED_TRIGGER_PLACEHOLDER} B", f"C {OPTIMIZED_TRIGGER_PLACEHOLDER} D"]
    targets = Targets(target_vectors=torch.randn(2, encoder_model.d_model))
    encoder_model.set_inputs_from_tokens(texts, targets)
    inputs = encoder_model._token_input_manager
    trigger_ids = encoder_model.tokenizer(
        "init", add_special_tokens=False, return_tensors="pt"
    )["input_ids"]

    n_candidates = 2
    candidate_trigger_ids = torch.randint(0, 100, (n_candidates, trigger_ids.shape[1]))

    # Choose message 1
    res = inputs.get_triggered_inputs(trigger_ids=candidate_trigger_ids, chosen_template_idx=1)

    # inputs_embeds: (n_candidates, seq_len, embd_dim) -- message dim removed
    assert res.input_embeds.dim() == 3
    assert res.input_embeds.shape[0] == n_candidates

    # Targets should be for single message now
    tgt_vecs = res.message_targets.target_vectors
    # Should be (d_model)
    assert isinstance(tgt_vecs, torch.Tensor)
    assert tgt_vecs.shape[0] == encoder_model.d_model

    encoder_model.reset_inputs_from_tokens()
