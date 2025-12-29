import pytest
import torch
from transformers import AutoTokenizer
from unittest.mock import MagicMock
from tropt.optimizer.gaslite_optimizer import GASLITEOptimizer
from tropt.models.base import BaseModel, LossTokenAccessMixin, GradientTokenAccessMixin, TokenInputsManager
from tropt.loss.base import BaseLoss

# TODO review & consider dropping the mocks


class MockInputsManager(TokenInputsManager):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.n_messages = 1
    
    def toks_to_strs(self, toks, **kwargs):
        return self.tokenizer.decode(toks)
        
    def get_triggered_inputs(self, *args, **kwargs):
        pass

class MockModel(BaseModel, LossTokenAccessMixin, GradientTokenAccessMixin):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cpu")
    
    def __call__(self, *args, **kwargs):
        pass

    def prepare_token_inputs(self, texts, initial_trigger, targets=None):
        inputs = MockInputsManager(self.tokenizer)
        trigger_ids = torch.tensor([self.tokenizer.encode(initial_trigger, add_special_tokens=False)], dtype=torch.long)
        return inputs, trigger_ids

    def compute_loss_from_tokens(self, candidate_trigger_ids, inputs, loss_func, keep_message_dim=False, **kwargs):
        # return random loss
        n_candidates = candidate_trigger_ids.shape[0]
        # shape: (n_messages, n_candidates)
        losses = torch.rand(inputs.n_messages, n_candidates)
        if not keep_message_dim:
            losses = losses.mean(dim=0)
        return losses

    def compute_grad_from_tokens(self, candidate_trigger_ids, inputs, **kwargs):
        # return random grad
        # Input shape: (n_cands, trigger_seq_len)
        # Output shape: (n_cands, trigger_seq_len, vocab_size)
        n_cands, trigger_seq_len = candidate_trigger_ids.shape
        vocab_size = inputs.vocab_size
        return torch.randn(n_cands, trigger_seq_len, vocab_size)

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