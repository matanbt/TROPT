from unittest.mock import MagicMock

import pytest
import torch
from transformers import AutoTokenizer

from tropt.common import ModelOutput
from tropt.loss import BaseLoss
from tropt.model import (
    BaseModel,
    GradientTokenAccessMixin,
    LossTokenAccessMixin,
    TokenInputManager,
)
from tropt.optimizer.gcg_optimizer import GCGOptimizer

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

    def invoke_from_tokens(self, *args, **kwargs):
        return ModelOutput()

    def set_inputs_from_tokens(self, templates, targets=None):
        self._token_input_manager = MockInputsManager(self.tokenizer)

    def reset_inputs_from_tokens(self):
        self._token_input_manager = None

    def compute_loss_from_tokens(self, candidate_trigger_ids, **kwargs):
        n_candidates = candidate_trigger_ids.shape[0]
        return torch.rand(self._token_input_manager.n_templates, n_candidates)

    def compute_grad_from_tokens(self, candidate_trigger_ids, **kwargs):
        n_candidates = candidate_trigger_ids.shape[0]
        trigger_seq_len = candidate_trigger_ids.shape[-1]
        vocab_size = self._token_input_manager.vocab_size
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
        use_retokenize=True
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
