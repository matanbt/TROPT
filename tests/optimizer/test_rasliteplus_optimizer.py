import pytest
import torch
from transformers import AutoTokenizer
from unittest.mock import MagicMock
from tropt.optimizer.rasliteplus_optimizer import RASLITEPlusOptimizer
from tropt.model import (
    BaseModel, 
    LossTextAccessMixin, 
    LogitsTokenAccessMixin, 
    LMBaseModel,
    TokenInputManager,
    TextInputManager
)
from tropt.loss import BaseLoss

class MockInputsManager(TokenInputManager):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self.n_templates = 1
    
    def get_triggered_inputs(self, *args, **kwargs):
        pass

class MockTextInputManager(TextInputManager):
    def __init__(self):
        # Mocking the attributes required by the n_templates property in TextInputManager
        self.before_texts = [""] 
        self.targets = {}


class MockTargetModel(BaseModel, LossTextAccessMixin):
    @property
    def tokenizer(self):
        return MagicMock()

    def __init__(self):
        pass
        
    def __call__(self, *args, **kwargs):
        pass

    def set_text_inputs(self, texts, targets=None):
        self._text_input_manager = MockTextInputManager()

    def reset_text_inputs(self):
        self._text_input_manager = None

    def compute_loss_from_texts(self, candidate_trigger_strs, loss_func, keep_message_dim=False, **kwargs):
        # return random loss for each candidate string
        n_candidates = len(candidate_trigger_strs)
        # shape: (n_templates, n_candidates)
        # Assuming 1 message
        losses = torch.rand(1, n_candidates)
        if not keep_message_dim:
            losses = losses.mean(dim=0)
        return losses

class MockUtilModel(LMBaseModel, LogitsTokenAccessMixin):
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

    def invoke_from_texts(self, *args, **kwargs):
        from tropt.common import ModelOutput
        return ModelOutput()

    def invoke_from_tokens(self, *args, **kwargs):
        from tropt.common import ModelOutput
        return ModelOutput()

    def set_inputs_from_tokens(self, templates, targets=None):
        self._token_input_manager = MockInputsManager(self.tokenizer)

    def reset_inputs_from_tokens(self):
        self._token_input_manager = None

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size

    def compute_logits_from_tokens(self, candidate_trigger_ids, return_trigger_logits_only=True, keep_message_dim=False, **kwargs):
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
        util_model=util_model,
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
        # Fails because util_model is not TokenAccess/LogitsAccess
        RASLITEPlusOptimizer(model=MockTargetModel(), loss=MockLoss(), util_model=BadModel("bad"))
