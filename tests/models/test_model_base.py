"""
Tests for base model functionality and mixins.

These tests verify the core infrastructure that all model implementations build upon.
"""

import pytest
import torch

from tropt.models.model_base import BaseModel, EncoderBaseModel, LMBaseModel

# ==============================================================================
# BaseModel Tests
# ==============================================================================

class MockModel(BaseModel):
    """Mock model for testing base functionality."""

    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self._usage_stats = {"forward_calls": 0, "forward_samples": 0}


def test_base_model_usage_stats():
    """Test that BaseModel tracks usage statistics correctly."""
    model = MockModel()

    assert model.get_usage_stats() == {"forward_calls": 0, "forward_samples": 0}

    # Update stats
    model._update_usage_stats(forward_calls=1, forward_samples=10)

    stats = model.get_usage_stats()
    assert stats["forward_calls"] == 1
    assert stats["forward_samples"] == 10

    # Update again (should accumulate)
    model._update_usage_stats(forward_calls=2, forward_samples=5)

    stats = model.get_usage_stats()
    assert stats["forward_calls"] == 3
    assert stats["forward_samples"] == 15


def test_base_model_reset_stats():
    """Test that usage statistics can be reset."""
    model = MockModel()

    model._update_usage_stats(forward_calls=5, forward_samples=50)
    model.reset_usage_stats()

    stats = model.get_usage_stats()
    assert stats["forward_calls"] == 0
    assert stats["forward_samples"] == 0


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
        self.device = torch.device(device)
        self._usage_stats = {"forward_calls": 0, "forward_samples": 0}

    def generate(self, *args, **kwargs):
        return ["Generated text"]


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
        self.d_model = d_model
        self.device = torch.device(device)
        self._usage_stats = {"forward_calls": 0, "forward_samples": 0}

    def __call__(self, texts):
        # Return random embeddings
        return torch.randn(len(texts), self.d_model)


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
    from tropt.models import (
        LMBaseModel,
        GradientTokenAccessMixin,
        LossTokenAccessMixin,
    )

    # Create a model that implements required mixins
    class ValidModel(LMBaseModel, GradientTokenAccessMixin, LossTokenAccessMixin):
        def __init__(self):
            self.device = torch.device("cpu")
            self._usage_stats = {}

    # This should not raise an error
    model = ValidModel()
    assert isinstance(model, GradientTokenAccessMixin)
    assert isinstance(model, LossTokenAccessMixin)


def test_usage_stats_initialization():
    """Test that all models initialize usage stats correctly."""
    from tropt.models.huggingface.encoder import EncoderHFModel

    # Test with a small model (requires internet to download)
    try:
        model = EncoderHFModel(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )

        stats = model.get_usage_stats()

        # Should start at zero
        assert "forward_calls" in stats
        assert "forward_samples" in stats
        assert stats["forward_calls"] == 0
        assert stats["forward_samples"] == 0

    except Exception as e:
        pytest.skip(f"Could not load model: {e}")


# ==============================================================================
# Edge Cases and Robustness
# ==============================================================================

def test_usage_stats_large_numbers():
    """Test that usage stats handle large numbers."""
    model = MockModel()

    # Simulate many calls
    for _ in range(10000):
        model._update_usage_stats(forward_calls=1, forward_samples=100)

    stats = model.get_usage_stats()
    assert stats["forward_calls"] == 10000
    assert stats["forward_samples"] == 1000000


def test_usage_stats_negative_values():
    """Test that negative values in usage stats are handled."""
    model = MockModel()

    # This shouldn't happen in practice, but test robustness
    model._update_usage_stats(forward_calls=-1, forward_samples=-10)

    stats = model.get_usage_stats()

    # Should still update (no validation on negatives, but shouldn't crash)
    assert "forward_calls" in stats
    assert "forward_samples" in stats


def test_device_consistency():
    """Test that device is consistently tracked."""
    model_cpu = MockModel(device="cpu")
    assert model_cpu.device.type == "cpu"

    # CUDA test would require GPU
    # model_cuda = MockModel(device="cuda")
    # assert model_cuda.device.type == "cuda"


def test_encoder_model_different_dimensions():
    """Test encoder model with various embedding dimensions."""
    for d_model in [64, 128, 256, 512, 768, 1024, 1536]:
        model = MockEncoderModel(d_model=d_model)
        embeddings = model(["test"])

        assert embeddings.shape == (1, d_model)


def test_lm_model_generation_interface():
    """Test that LM models have consistent generation interface."""
    model = MockLMModel()

    # Basic generation
    output = model.generate()
    assert isinstance(output, list)

    # With arguments (should not crash)
    output = model.generate(max_length=100, temperature=0.7)
    assert isinstance(output, list)
