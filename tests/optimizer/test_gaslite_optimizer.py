import pytest
import torch
from transformers import AutoTokenizer
from unittest.mock import MagicMock
from tropt.optimizer.gaslite_optimizer import GASLITEOptimizer
from tropt.model import BaseModel, LossTokenAccessMixin, GradientTokenAccessMixin, TokenInputManager
from tropt.loss import BaseLoss

# TODO review & consider dropping the mocks


class MockInputsManager(TokenInputManager):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.n_templates = 1
    
    def get_triggered_inputs(self, *args, **kwargs):
        pass

class MockModel(BaseModel, LossTokenAccessMixin, GradientTokenAccessMixin):
    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self):
        return torch.device("cpu")

    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def __call__(self, *args, **kwargs):
        pass

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size

    def set_token_inputs(self, texts, targets=None):
        self._token_input_manager = MockInputsManager(self.tokenizer)

    def reset_token_inputs(self):
        self._token_input_manager = None

    def compute_loss_from_tokens(self, candidate_trigger_ids, loss_func=None, keep_message_dim=False, **kwargs):
        # return random loss
        n_candidates = candidate_trigger_ids.shape[0]
        # shape: (n_templates, n_candidates)
        losses = torch.rand(self._token_input_manager.n_templates, n_candidates, device=self.device)
        if not keep_message_dim:
            losses = losses.mean(dim=0)
        return losses

    def compute_grad_from_tokens(self, candidate_trigger_ids, **kwargs):
        # return random grad
        # Input shape: (n_cands, trigger_seq_len)
        # Output shape: (n_cands, trigger_seq_len, vocab_size)
        n_cands, trigger_seq_len = candidate_trigger_ids.shape
        vocab_size = self._token_input_manager.vocab_size
        return torch.randn(n_cands, trigger_seq_len, vocab_size, device=self.device)

    def compute_logits_from_tokens(self, *args, **kwargs):
        pass

class MockLoss(BaseLoss):
    def __call__(self, *args, **kwargs):
        return torch.tensor([0.0])

def test_gaslite_optimizer_run():
    model = MockModel()
    loss = MockLoss()
    
    optimizer = GASLITEOptimizer(
        model=model,
        loss=loss,
        num_steps=2,
        n_grad=2, # Small number for testing
        n_flip=1,
        n_candidates=10,
        use_retokenize=True # Use real tokenizer logic
    )
    
    texts = ["Test message"]
    initial_trigger = "ABC"
    
    result = optimizer.optimize_trigger(texts, initial_trigger=initial_trigger)
    
    assert result.best_loss is not None
    assert len(result.losses) == 2
    assert len(result.trigger_strs) == 2
    assert result.best_trigger_str is not None

def test_gaslite_requirements():
    class BadModel(BaseModel):
        def __init__(self, model_name=""): pass
        def __call__(self): pass

    with pytest.raises(AssertionError):
        GASLITEOptimizer(BadModel(""), MockLoss())