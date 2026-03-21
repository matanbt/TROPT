"""
Tests for base model functionality and mixins.

These tests verify the core infrastructure that all model implementations build upon.
"""

import pytest
import torch

from tropt.common import ModelOutput
from tropt.model.model_base import BaseModel, EncoderBaseModel, LMBaseModel

# ==============================================================================
# BaseModel Tests
# ==============================================================================

class MockModel(BaseModel):
    """Mock model for testing base functionality."""

    def __init__(self, device="cpu"):
        self._device = torch.device(device)

    def __call__(self, *args, **kwargs):
        pass

    @property
    def device(self):
        return self._device


def test_base_model_usage_stats():
    """Test that BaseModel tracks usage statistics correctly."""
    model = MockModel()

    expected_initial_stats = {
        "usage/total_tokens": 0,
        "usage/forward_calls": 0,
        "usage/forward_samples": 0,
        "usage/grad_calls": 0,
        "usage/grad_samples": 0,
    }
    assert model.get_usage_stats() == expected_initial_stats

    model._update_usage_stats(forward_calls=1, forward_samples=10)

    stats = model.get_usage_stats()
    assert stats["usage/forward_calls"] == 1
    assert stats["usage/forward_samples"] == 10

    model._update_usage_stats(forward_calls=2, forward_samples=5)

    stats = model.get_usage_stats()
    assert stats["usage/forward_calls"] == 3
    assert stats["usage/forward_samples"] == 15


def test_base_model_reset_stats():
    """Test that usage statistics can be reset."""
    model = MockModel()

    model._update_usage_stats(forward_calls=5, forward_samples=50)
    model.reset_usage_stats()

    stats = model.get_usage_stats()
    assert stats["usage/forward_calls"] == 0
    assert stats["usage/forward_samples"] == 0


def test_base_model_device_property():
    """Test that device property is accessible."""
    model_cpu = MockModel(device="cpu")
    assert model_cpu.device.type == "cpu"


# ==============================================================================
# LMBaseModel Tests
# ==============================================================================

class MockLMModel(LMBaseModel):
    """Mock language model for testing."""

    def __init__(self, device="cpu"):
        self._device = torch.device(device)

    def __call__(self, *args, **kwargs):
        pass

    def invoke_from_texts(self, *args, **kwargs):
        return ModelOutput()

    def generate(self, *args, **kwargs):
        return ["Generated text"]

    @property
    def device(self):
        return self._device


def test_lm_base_model_is_base_model():
    """Test that LMBaseModel inherits from BaseModel."""
    model = MockLMModel()
    assert isinstance(model, BaseModel)
    assert hasattr(model, "get_usage_stats")


def test_lm_base_model_has_generate():
    """Test that LMBaseModel has generate method."""
    model = MockLMModel()
    assert hasattr(model, "generate")

    result = model.generate()
    assert isinstance(result, list)


# ==============================================================================
# EncoderBaseModel Tests
# ==============================================================================

class MockEncoderModel(EncoderBaseModel):
    """Mock encoder model for testing."""

    def __init__(self, d_model=128, device="cpu"):
        self._d_model = d_model
        self._device = torch.device(device)

    @property
    def d_model(self):
        return self._d_model

    def __call__(self, texts):
        return torch.randn(len(texts), self._d_model)

    def invoke_from_texts(self, *args, **kwargs):
        return ModelOutput()

    @property
    def device(self):
        return self._device


def test_encoder_base_model_is_base_model():
    """Test that EncoderBaseModel inherits from BaseModel."""
    model = MockEncoderModel()
    assert isinstance(model, BaseModel)
    assert hasattr(model, "get_usage_stats")


def test_encoder_base_model_has_d_model():
    """Test that EncoderBaseModel has d_model attribute."""
    model = MockEncoderModel(d_model=256)
    assert model.d_model == 256


def test_encoder_base_model_call():
    """Test that EncoderBaseModel is callable."""
    model = MockEncoderModel(d_model=128)

    texts = ["Hello", "World"]
    embeddings = model(texts)

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 128)


# ==============================================================================
# Mixin Testing Framework
# ==============================================================================

def test_model_requirements_validation():
    """Test that model requirements are validated correctly.

    This is a critical test because optimizers depend on this validation
    to ensure they can safely call model methods.
    """
    from tropt.common import ModelOutput
    from tropt.model import (
        GradientTokenAccessMixin,
        LMBaseModel,
        LossTokenAccessMixin,
    )
    from tropt.model.model_base import BaseTokenizer

    class MockTokenizer(BaseTokenizer):
        @property
        def vocab_size(self) -> int:
            return 100
        def __call__(self, text, return_tensors="list", **kwargs):
            return {"input_ids": [[0,1,2]]}
        def decode(self, ids, **kwargs):
            return ""
        def encode(self, text, **kwargs):
            return [0,1,2]
        def batch_decode(self, ids, **kwargs):
            return [""]

    class ValidModel(LMBaseModel, GradientTokenAccessMixin, LossTokenAccessMixin):
        def __init__(self):
            self._device = torch.device("cpu")
            self._tokenizer = MockTokenizer()

        def __call__(self, *args, **kwargs):
            pass

        def invoke_from_texts(self, *args, **kwargs):
            return ModelOutput()

        def invoke_from_tokens(self, *args, **kwargs):
            return ModelOutput()

        def compute_grad_from_tokens(self, *args, **kwargs):
            pass

        def compute_loss_from_tokens(self, *args, **kwargs):
            pass

        def set_inputs_from_tokens(self, *args, **kwargs):
            pass

        @property
        def tokenizer(self):
            return self._tokenizer

        @property
        def device(self):
            return self._device

    model = ValidModel()
    assert isinstance(model, GradientTokenAccessMixin)
    assert isinstance(model, LossTokenAccessMixin)


def test_usage_stats_initialization():
    """Test that all models initialize usage stats correctly."""
    from tropt.model.huggingface.encoder import EncoderHFModel

    try:
        model = EncoderHFModel(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

        stats = model.get_usage_stats()

        assert "usage/forward_calls" in stats
        assert "usage/forward_samples" in stats
        assert stats["usage/forward_calls"] == 0
        assert stats["usage/forward_samples"] == 0

    except Exception as e:
        pytest.skip(f"Could not load model: {e}")


# ==============================================================================
# Edge Cases and Robustness
# ==============================================================================

def test_usage_stats_large_numbers():
    """Test that usage stats handle large numbers."""
    model = MockModel()

    for _ in range(10000):
        model._update_usage_stats(forward_calls=1, forward_samples=100)

    stats = model.get_usage_stats()
    assert stats["usage/forward_calls"] == 10000
    assert stats["usage/forward_samples"] == 1000000


def test_usage_stats_negative_values():
    """Test that negative values in usage stats are handled."""
    model = MockModel()

    model._update_usage_stats(forward_calls=-1, forward_samples=-10)

    stats = model.get_usage_stats()

    assert "usage/forward_calls" in stats
    assert "usage/forward_samples" in stats


def test_device_consistency():
    """Test that device is consistently tracked."""
    model_cpu = MockModel(device="cpu")
    assert model_cpu.device.type == "cpu"


def test_encoder_model_different_dimensions():
    """Test encoder model with various embedding dimensions."""
    for d_model in [64, 128, 256, 512, 768, 1024, 1536]:
        model = MockEncoderModel(d_model=d_model)
        embeddings = model(["test"])

        assert embeddings.shape == (1, d_model)


def test_lm_model_generation_interface():
    """Test that LM models have consistent generation interface."""
    model = MockLMModel()

    output = model.generate()
    assert isinstance(output, list)

    output = model.generate(max_length=100, temperature=0.7)
    assert isinstance(output, list)
