import pytest
import torch
from transformers import AutoTokenizer
from unittest.mock import MagicMock
from tropt.optimizer.rasliteplus_optimizer import RASLITEPlusOptimizer
from tropt.models.base import (
    BaseModel, 
    LossTextAccessMixin, 
    LogitsTokenAccessMixin, 
    LMBaseModel,
    TokenInputsManager,
    TextInputsManager
)
from tropt.loss.base import BaseLoss

class MockInputsManager(TokenInputsManager):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.n_messages = 1
    
    def toks_to_strs(self, toks, **kwargs):
        # Handle batched input
        if toks.dim() == 2:
            return self.tokenizer.batch_decode(toks)
        return self.tokenizer.decode(toks)
        
    def get_triggered_inputs(self, *args, **kwargs):
        pass

class MockTextInputsManager(TextInputsManager):
    def __init__(self):
        # Mocking the attributes required by the n_messages property in TextInputsManager
        self.before_texts = [""] 
        self.targets = {}


class MockTargetModel(BaseModel, LossTextAccessMixin):
    def __init__(self):
        self.device = torch.device("cpu")
        
    def __call__(self, *args, **kwargs):
        pass

    def prepare_text_inputs(self, texts, initial_trigger, targets=None):
        return MockTextInputsManager(), initial_trigger

    def compute_loss_from_texts(self, candidate_trigger_strs, inputs, loss_func, keep_message_dim=False, **kwargs):
        # return random loss for each candidate string
        n_candidates = len(candidate_trigger_strs)
        # shape: (n_messages, n_candidates)
        # Assuming 1 message
        losses = torch.rand(1, n_candidates)
        if not keep_message_dim:
            losses = losses.mean(dim=0)
        return losses

class MockUtilModel(LMBaseModel, LogitsTokenAccessMixin):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cpu")
        self.vocab_size = self.tokenizer.vocab_size
    
    def __call__(self, *args, **kwargs):
        pass

    def prepare_token_inputs(self, texts, initial_trigger, targets=None):
        inputs = MockInputsManager(self.tokenizer)
        trigger_ids = torch.tensor([self.tokenizer.encode(initial_trigger, add_special_tokens=False)], dtype=torch.long)
        return inputs, trigger_ids

    def compute_logits_from_tokens(self, candidate_trigger_ids, inputs, return_trigger_logits_only=True, keep_message_dim=False, **kwargs):
        # Return random logits
        # candidate_trigger_ids shape: (n_cands, seq_len)
        n_cands, seq_len = candidate_trigger_ids.shape
        # Logits shape: (n_cands, seq_len, vocab_size) if return_trigger_logits_only=True
        # Actually in RASLITEPlus we handle keep_message_dim=False usually for candidate selection
        # If we use 1 message.
        return torch.randn(n_cands, seq_len, self.vocab_size)

class MockLoss(BaseLoss):
    def __call__(self, *args, **kwargs):
        return torch.tensor([0.0])

def test_rasliteplus_optimizer_run():
    target_model = MockTargetModel()
    util_model = MockUtilModel()
    loss = MockLoss()
    
    optimizer = RASLITEPlusOptimizer(
        model=target_model,
        util_lm=util_model,
        loss=loss,
        num_steps=2,
        n_grad=2, 
        n_flip=1,
        n_candidates=5,
        buffer_size=5,
        use_retokenize=True,
        n_bulk_flips=1 # Simplify for test
    )
    
    texts = ["Test message"]
    initial_trigger = "ABC"
    
    result = optimizer.optimize_trigger(texts, initial_trigger=initial_trigger)
    
    assert result.best_loss is not None
    assert len(result.losses) == 2
    assert len(result.trigger_strs) == 2
    assert result.best_trigger_str is not None

def test_rasliteplus_requirements():
    class BadModel(BaseModel):
        def __init__(self, model_name=""): pass
        def __call__(self): pass

    with pytest.raises(AssertionError):
        # Fails because util_lm is not TokenAccess/LogitsAccess
        RASLITEPlusOptimizer(model=MockTargetModel(), loss=MockLoss(), util_lm=BadModel("bad"))
