"""
Tests for OpenAI encoder model integration.

These tests use mocked API calls to avoid requiring actual API keys and incurring costs.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER, Targets
from tropt.loss import SimilarityLoss
from tropt.model.openai.encoder import OpenAITokenizer # Import OpenAITokenizer directly


@pytest.fixture
def mock_openai_client():
    """Create a mocked OpenAI client."""
    mock_client = MagicMock()

    def mock_create_embeddings(input, model, **kwargs):
        num_inputs = len(input) if isinstance(input, list) else 1
        mock_embedding_data_list = []
        for _ in range(num_inputs):
            mock_embedding_data = Mock()
            mock_embedding_data.embedding = [0.1] * 1536  # Standard embedding size
            mock_embedding_data_list.append(mock_embedding_data)

        mock_response = Mock()
        mock_response.data = mock_embedding_data_list
        mock_response.usage.total_tokens = 10 * num_inputs
        return mock_response

    mock_client.embeddings.create.side_effect = mock_create_embeddings

    with patch("openai.OpenAI", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_tiktoken():
    """Mock tiktoken for tokenization."""
    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = [[1, 2, 3, 4, 5]] # Changed to 2D tensor
    mock_encoding.decode.return_value = f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}" # Example with placeholder
    
    def mock_encode_batch_side_effect(texts, **kwargs):
        # Each text gets a dummy token list
        return [[1, 2, 3, 4, 5] for _ in texts]

    def mock_decode_batch_side_effect(ids, **kwargs):
        # Each list of ids gets a dummy string with placeholder
        return [f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}" for _ in ids]

    mock_encoding.encode_batch.side_effect = mock_encode_batch_side_effect
    mock_encoding.decode_batch.side_effect = mock_decode_batch_side_effect

    mock_encoding.eot_token = 0
    mock_encoding.max_token_value = 100000
    mock_encoding.name = "cl100k_base"
    mock_encoding.special_tokens_set = set()

    with patch("tropt.models.openai.encoder.tiktoken.encoding_for_model", return_value=mock_encoding), \
         patch("tropt.models.openai.encoder.tiktoken.get_encoding", return_value=mock_encoding), \
         patch.object(OpenAITokenizer, "batch_decode", side_effect=lambda ids, **kwargs: [f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}" for _ in ids]):
        yield mock_encoding


def test_openai_encoder_init(mock_openai_client, mock_tiktoken):
    """Test OpenAI encoder initialization."""
    from tropt.model.openai.encoder import EncoderOpenAIModel

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
    from tropt.model.openai.encoder import EncoderOpenAIModel

    # Mock should return 1536-dim embedding
    model = EncoderOpenAIModel(
        model_name="text-embedding-3-small",
        api_key="test-key",
    )

    assert model.d_model == 1536  # From mocked response


def test_openai_encoder_call(mock_openai_client, mock_tiktoken):
    """Test OpenAI encoder forward pass (embedding generation)."""
    from tropt.model.openai.encoder import EncoderOpenAIModel

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


def test_openai_encoder_set_token_inputs(mock_openai_client, mock_tiktoken):
    """Test set_token_inputs for OpenAI encoder."""
    from tropt.model.openai.encoder import EncoderOpenAIModel

    model = EncoderOpenAIModel(
        model_name="text-embedding-3-small",
        d_model=1536,
        api_key="test-key",
    )

    texts = [f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    targets = Targets(target_vectors=torch.randn(1, 1536))

    model.set_token_inputs(texts, targets)
    trigger_ids = model.tokenizer("test trigger", return_tensors="pt")["input_ids"]

    assert model._token_input_manager is not None
    assert isinstance(trigger_ids, torch.Tensor)
    assert trigger_ids.ndim == 2  # (1, trigger_length)

    model.reset_token_inputs()
    assert model._token_input_manager is None


def test_openai_encoder_compute_loss_from_texts(mock_openai_client, mock_tiktoken):
    """Test compute_loss_from_texts for OpenAI encoder."""
    from tropt.model.openai.encoder import EncoderOpenAIModel

    model = EncoderOpenAIModel(
        model_name="text-embedding-3-small",
        d_model=1536,
        api_key="test-key",
    )

    texts = [f"Query: {OPTIMIZED_TRIGGER_PLACEHOLDER}"]
    targets = Targets(target_vectors=torch.randn(1, 1536))

    # Use set_text_inputs to configure the model
    model.set_text_inputs(texts, targets)

    # Create candidate trigger strings
    n_candidates = 3
    candidate_strs = ["trigger one", "trigger two", "trigger three"]

    loss_fn = SimilarityLoss()

    losses = model.compute_loss_from_texts(candidate_strs, loss_fn)

    assert isinstance(losses, torch.Tensor)
    assert losses.shape == (n_candidates,)
    assert not torch.isnan(losses).any()

    model.reset_text_inputs()


def test_openai_encoder_multi_message(mock_openai_client, mock_tiktoken):
    """Test OpenAI encoder with multiple messages."""
    from tropt.model.openai.encoder import EncoderOpenAIModel

    model = EncoderOpenAIModel(
        model_name="text-embedding-3-small",
        d_model=1536,
        api_key="test-key",
    )

    texts = [
        f"First query: {OPTIMIZED_TRIGGER_PLACEHOLDER}",
        f"Second query: {OPTIMIZED_TRIGGER_PLACEHOLDER}",
    ]
    targets = Targets(target_vectors=torch.randn(2, 1536))

    model.set_token_inputs(texts, targets)

    assert model._token_input_manager.n_templates == 2

    model.reset_token_inputs()


def test_openai_tokenizer(mock_tiktoken):
    """Test OpenAI tokenizer wrapper."""
    from tropt.model.openai.encoder import OpenAITokenizer

    tokenizer = OpenAITokenizer("gpt-4")

    assert tokenizer is not None
    assert hasattr(tokenizer, "bos_token_id")
    assert hasattr(tokenizer, "eos_token_id")
    assert hasattr(tokenizer, "vocab_size")


def test_openai_encoder_usage_stats(mock_openai_client, mock_tiktoken):
    """Test that usage statistics are tracked."""
    from tropt.model.openai.encoder import EncoderOpenAIModel

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
