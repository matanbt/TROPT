"""
Tests for the LiteLLM model wrapper.
Important note: These tests require a running LiteLLM proxy server and Ollama with the specified model pulled.
"""
from tropt.models.litellm_proxy.lm import LiteLLMModel

import pytest
from tropt.models.litellm_proxy.lm import LiteLLMModel

MODEL_NAME = "ollama/gemma3:270m"
LITELLM_URL = "http://localhost:4000"
OLLAMA_URL = "http://localhost:11434"

# Define the two configurations here
CONFIGURATIONS = [
    # Case 1: Connect via LiteLLM Proxy (OpenAI Protocol)
    pytest.param(
        {
            "model_name": MODEL_NAME, # Proxy handles the 'ollama/' prefix usually
            "base_url": LITELLM_URL,
        },
        id="litellm_proxy"
    ),
    # Case 2: Connect directly to Ollama (Ollama Protocol)
    pytest.param(
        {
            "model_name": MODEL_NAME,
            "base_url": OLLAMA_URL,
            # "api_key": None,
            "using_litellm_proxy": False,
        },
        id="ollama_direct"
    ),
]

@pytest.mark.parametrize("config", CONFIGURATIONS)
def test_litellm_model_integration(config):
    """
    Tests the LiteLLMModel wrapper against both Proxy and Direct connections.
    """
    
    # Initialize model by unpacking the dictionary (**config)
    model = LiteLLMModel(**config)
    
    # Test single input
    text = "Hello, reply with 'World'"
    responses = model([text], max_new_tokens=8)
    
    assert len(responses) == 1
    if responses[0]:
        assert len(responses[0]) > 0
    assert isinstance(responses[0], str)

    # Test multiple inputs
    texts = [f"Input {i}: Say 'Test {i}'" for i in range(2)]
    responses = model(texts, max_new_tokens=8)
    assert len(responses) == 2
    if responses[0]:
        assert len(responses[0]) > 0
    if responses[1]:
        assert len(responses[1]) > 0
    assert isinstance(responses[0], str) and isinstance(responses[1], str)
