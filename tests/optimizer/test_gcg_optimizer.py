import pytest
import torch
from transformers import AutoTokenizer
from unittest.mock import MagicMock
from tropt.optimizer.gcg_optimizer import GCGOptimizer
from tropt.model import BaseModel, LossTokenAccessMixin, GradientTokenAccessMixin, TokenInputManager
from tropt.loss import BaseLoss

# TODO review & consider dropping the mocks (or not because it's useful for isolated testing)

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

    def compute_loss_from_tokens(self, candidate_trigger_ids, **kwargs):
        # return random loss
        n_candidates = candidate_trigger_ids.shape[0]
        # shape: (n_templates, n_candidates)
        return torch.rand(self._token_input_manager.n_templates, n_candidates)

    def compute_grad_from_tokens(self, candidate_trigger_ids, **kwargs):
        # return random grad
        # Input shape: (n_candidates, trigger_seq_len)
        n_candidates = candidate_trigger_ids.shape[0]
        trigger_seq_len = candidate_trigger_ids.shape[-1]
        vocab_size = self._token_input_manager.vocab_size
        # Output shape: (n_candidates, trigger_seq_len, vocab_size)
        return torch.randn(n_candidates, trigger_seq_len, vocab_size)

    def compute_logits_from_tokens(self, *args, **kwargs):
        pass

class MockLoss(BaseLoss):
    def __call__(self, *args, **kwargs):
        return torch.tensor([0.0])

def test_gcg_optimizer_run():
    model = MockModel()
    loss = MockLoss()
    
    optimizer = GCGOptimizer(
        model=model,
        loss=loss,
        num_steps=2,
        n_candidates=10,
        sample_topk=5,
        sample_n_replace=1,
        use_retokenize=True # Use real tokenizer logic
    )
    
    texts = ["Test message"]
    initial_trigger = "ABC"
    
    result = optimizer.optimize_trigger(texts, initial_trigger=initial_trigger)
    
    assert result.best_loss is not None
    assert len(result.losses) == 2
    assert len(result.trigger_strs) == 2
    assert result.best_trigger_str is not None

def test_gcg_requirements():
    class BadModel(BaseModel):
        def __init__(self, model_name=""): pass
        def __call__(self): pass

    with pytest.raises(AssertionError):
        GCGOptimizer(BadModel(""), MockLoss())
