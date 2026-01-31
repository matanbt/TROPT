"""
Tests for OpenAI encoder model integration.

These tests use mocked API calls to avoid requiring actual API keys and incurring costs.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER
from tropt.loss.base import SimilarityLoss


@pytest.fixture
def mock_openai_client():
    """Create a mocked OpenAI client."""
    with patch("tropt.models.openai.encoder.OpenAI") as mock_client_class:
        # Mock the client instance
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock embeddings response
        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1] * 1536  # Standard embedding size

        mock_response = Mock()
        mock_response.data = [mock_embedding_data]

        mock_client.embeddings.create.return_value = mock_response

        yield mock_client


@pytest.fixture
def mock_tiktoken():
    """Mock tiktoken for tokenization."""
    with patch("tropt.models.openai.encoder.tiktoken") as mock_tiktoken_module:
        # Mock encoding
        mock_encoding = MagicMock()
        mock_encoding.encode.return_value = [1, 2, 3, 4, 5]
        mock_encoding.decode.return_value = "test"
        mock_encoding.eot_token = 0
        mock_encoding.max_token_value = 100000
        mock_encoding.name = "cl100k_base"
        mock_encoding.special_tokens_set = set()

        mock_tiktoken_module.encoding_for_model.return_value = mock_encoding
        mock_tiktoken_module.get_encoding.return_value = mock_encoding

        yield mock_encoding


def test_openai_encoder_init(mock_openai_client, mock_tiktoken):
    """Test OpenAI encoder initialization."""
    from tropt.models.openai.encoder import EncoderOpenAIModel

    model = EncoderOpenAIModel(
        model_name="text-embedding-3-small",
        d_model=1536,
        api_key="test-key",
    )

    assert model is not None
    assert model.model_name == "text-embedding-3-small"
    assert model.d_model == 1536
    assert model.client is not None


def test_openai_encoder_init_deduces_d_model(mock_openai_client, mock_tiktoken):
    """Test that OpenAI encoder can deduce d_model from API."""
    from tropt.models.openai.encoder import EncoderOpenAIModel

    # Mock should return 1536-dim embedding
    model = EncoderOpenAIModel(
        model_name="text-embedding-3-small",
        api_key="test-key",
    )

    assert model.d_model == 1536  # From mocked response


def test_openai_encoder_call(mock_openai_client, mock_tiktoken):
    """Test OpenAI encoder forward pass (embedding generation)."""
    from tropt.models.openai.encoder import EncoderOpenAIModel

    model = EncoderOpenAIModel(
        model_name="text-embedding-3-small",
        d_model=1536,
        api_key="test-key",
    )

    texts = ["Hello world", "Test embedding"]
    embeddings = model(texts)

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (2, 1536)  # (n_texts, d_model)
    assert not torch.isnan(embeddings).any()


def test_openai_encoder_prepare_token_inputs(mock_openai_client, mock_tiktoken):
    """Test prepare_token_inputs for OpenAI encoder."""
    from tropt.models.openai.encoder import EncoderOpenAIModel

    model = EncoderOpenAIModel(
        model_name="text-embedding-3-small",
        d_model=1536,
        api_key="test-key",
    )

    texts = [f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    targets = {"target_vectors": torch.randn(1, 1536)}

    inputs, trigger_ids = model.prepare_token_inputs(
        texts, targets, initial_trigger="test trigger"
    )

    assert inputs is not None
    assert isinstance(trigger_ids, torch.Tensor)
    assert trigger_ids.ndim == 2  # (n_messages, trigger_length)


def test_openai_encoder_compute_loss_from_tokens(mock_openai_client, mock_tiktoken):
    """Test compute_loss_from_tokens for OpenAI encoder."""
    from tropt.models.openai.encoder import EncoderOpenAIModel

    model = EncoderOpenAIModel(
        model_name="text-embedding-3-small",
        d_model=1536,
        api_key="test-key",
    )

    texts = [f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    targets = {"target_vectors": torch.randn(1, 1536)}

    inputs, trigger_ids = model.prepare_token_inputs(texts, targets)

    # Create candidate trigger IDs
    n_candidates = 3
    candidate_ids = trigger_ids.repeat(n_candidates, 1)  # Same shape as trigger_ids

    loss_fn = SimilarityLoss()

    losses = model.compute_loss_from_tokens(candidate_ids, inputs, loss_fn)

    assert isinstance(losses, torch.Tensor)
    assert losses.shape == (n_candidates,)
    assert not torch.isnan(losses).any()


def test_openai_encoder_multi_message(mock_openai_client, mock_tiktoken):
    """Test OpenAI encoder with multiple messages."""
    from tropt.models.openai.encoder import EncoderOpenAIModel

    model = EncoderOpenAIModel(
        model_name="text-embedding-3-small",
        d_model=1536,
        api_key="test-key",
    )

    texts = [
        f"First query: {OPTIMIZED_TRIGGER_PLACEHOLDER}",
        f"Second query: {OPTIMIZED_TRIGGER_PLACEHOLDER}",
    ]
    targets = {"target_vectors": torch.randn(2, 1536)}

    inputs, trigger_ids = model.prepare_token_inputs(texts, targets)

    assert inputs.n_messages == 2
    assert trigger_ids.shape[0] == 1  # Shared trigger


def test_openai_tokenizer(mock_tiktoken):
    """Test OpenAI tokenizer wrapper."""
    from tropt.models.openai.encoder import OpenAITokenizer

    tokenizer = OpenAITokenizer("gpt-4")

    assert tokenizer is not None
    assert hasattr(tokenizer, "bos_token_id")
    assert hasattr(tokenizer, "eos_token_id")
    assert hasattr(tokenizer, "vocab_size")


def test_openai_encoder_usage_stats(mock_openai_client, mock_tiktoken):
    """Test that usage statistics are tracked."""
    from tropt.models.openai.encoder import EncoderOpenAIModel

    model = EncoderOpenAIModel(
        model_name="text-embedding-3-small",
        d_model=1536,
        api_key="test-key",
    )

    # Reset stats
    model._usage_stats = {"forward_calls": 0, "forward_samples": 0}

    texts = ["Test 1", "Test 2"]
    _ = model(texts)

    stats = model.get_usage_stats()

    assert stats["forward_calls"] >= 1
    assert stats["forward_samples"] >= 2
