import pytest
import torch
from transformers import AutoTokenizer
from unittest.mock import MagicMock
from tropt.optimizer.beast_optimizer import BEASTOptimizer
from tropt.models import (
    BaseModel,
    LMBaseModel,
    LossTokenAccessMixin, # Changed from LossTextAccessMixin
    LogitsTokenAccessMixin,
    TextInputManager,
    TokenInputManager
)
from tropt.loss.base import BaseLoss


class MockTextInputManager(TextInputManager):
    """Mock text inputs manager for testing"""
    def __init__(self):
        # Set the attributes that n_templates property depends on
        self.before_texts = ["Test message "]
        self.after_texts = [""]
        self.targets = {}  # TargetsDict is a type alias, use plain dict

    def get_triggered_inputs(self, *args, **kwargs):
        pass


class MockTokenInputManager(TokenInputManager):
    """Mock token inputs manager for testing"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        # Set attributes that n_templates might depend on
        self.before_ids = [torch.tensor([1, 2, 3])]  # Single message
        self.after_ids = [torch.tensor([4, 5])]
        self.targets = {}  # TargetsDict is a type alias, use plain dict

    @property
    def n_templates(self):
        return len(self.before_ids)

    def get_triggered_inputs(self, *args, **kwargs):
        pass


class MockUtilLM(LMBaseModel, LogitsTokenAccessMixin):
    """Mock utility LM for generating candidate tokens"""

    @property
    def tokenizer(self):
        return self._tokenizer

    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def __call__(self, *args, **kwargs):
        pass

    def set_token_inputs(self, texts, targets=None):
        self._token_input_manager = MockTokenInputManager(self.tokenizer)

    def reset_token_inputs(self):
        self._token_input_manager = None

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size

    def compute_logits_from_tokens(self, trigger_ids, return_after_trigger_logits_only=False):
        """Return mock logits for next token prediction"""
        batch_size = trigger_ids.shape[0]
        vocab_size = self.tokenizer.vocab_size

        if return_after_trigger_logits_only:
            # Return logits for next token: (batch, 1, vocab_size)
            return torch.randn(batch_size, 1, vocab_size)
        else:
            seq_len = trigger_ids.shape[-1]
            return torch.randn(batch_size, seq_len, vocab_size)


class MockTargetModel(BaseModel, LossTokenAccessMixin): # Changed from LossTextAccessMixin
    """Mock target model for white-box loss evaluation""" # Changed docstring

    @property
    def tokenizer(self):
        return self._tokenizer

    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def __call__(self, *args, **kwargs):
        pass

    def set_token_inputs(self, texts, targets=None):
        self._token_input_manager = MockTokenInputManager(self.tokenizer)

    def reset_token_inputs(self):
        self._token_input_manager = None

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size

    def compute_loss_from_tokens(self, candidate_trigger_ids, loss_func=None):
        """Return mock losses for candidate triggers"""
        n_candidates = candidate_trigger_ids.shape[0]
        # Return random losses that decrease over time to simulate optimization
        return torch.rand(n_candidates) * 0.5 + 0.5  # losses in [0.5, 1.0]


class MockLoss(BaseLoss):
    """Mock loss function"""
    def __call__(self, *args, **kwargs):
        return torch.tensor([0.0])


def test_beast_optimizer_initialization():
    """Test BEAST optimizer initialization with model requirements"""
    target_model = MockTargetModel()
    util_lm = MockUtilLM()
    loss = MockLoss()

    optimizer = BEASTOptimizer(
        model=target_model,
        loss=loss,
        util_lm=util_lm,
        num_steps=3,
        beam_size=2,
        branching_factor=2
    )

    assert optimizer.num_steps == 3
    assert optimizer.beam_size == 2
    assert optimizer.branching_factor == 2
    assert optimizer.util_lm is util_lm


def test_beast_optimizer_run():
    """Test basic BEAST optimization run"""
    target_model = MockTargetModel()
    util_lm = MockUtilLM()
    loss = MockLoss()

    optimizer = BEASTOptimizer(
        model=target_model,
        loss=loss,
        util_lm=util_lm,
        num_steps=3,  # Minimal steps for testing
        beam_size=2,  # Small beam for testing
        branching_factor=2,  # Small branching for testing
        temperature=1.0
    )

    texts = ["Test message {{OPTIMIZED_TRIGGER}}"]

    result = optimizer.optimize_trigger(texts, initial_trigger="")

    # Check result structure
    assert result.best_loss is not None
    assert result.best_trigger_str is not None
    assert len(result.trigger_strs) == 2  # num_steps - 1 (we start with 1 token)
    assert isinstance(result.best_trigger_str, str)


def test_beast_requirements_validation():
    """Test that BEAST validates model requirements"""

    class BadModel(BaseModel):
        """Model without required LossTextAccessMixin"""
        def __init__(self, model_name=""):
            pass
        def __call__(self):
            pass

    with pytest.raises(AssertionError):
        BEASTOptimizer(BadModel(""), MockLoss())


def test_beast_util_lm_validation():
    """Test that BEAST validates util_lm requirements"""
    target_model = MockTargetModel()
    loss = MockLoss()

    class BadUtilLM(BaseModel):
        """Util LM without required LogitsTokenAccessMixin"""
        def __init__(self):
            pass
        def __call__(self):
            pass

    with pytest.raises(AssertionError, match="BEAST requires util_lm to be LM"):
        BEASTOptimizer(
            model=target_model,
            loss=loss,
            util_lm=BadUtilLM()
        )


def test_beast_multinomial_sampling():
    """Test the multinomial sampling utility function"""
    # Test basic sampling
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    probs = probs / probs.sum()  # Normalize

    samples = BEASTOptimizer._sample_multinomial(probs, return_tokens=2)
    assert samples.shape == (1, 2)
    assert samples.max() < 4  # Valid indices
    assert samples.min() >= 0

    # Test top-k sampling
    samples_topk = BEASTOptimizer._sample_multinomial(probs, return_tokens=2, top_k=3)
    assert samples_topk.shape == (1, 2)
    assert samples_topk.max() < 4
    assert samples_topk.min() >= 0
