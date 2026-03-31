import logging
from typing import Dict, List, Optional
import torch
import litellm

from tropt.common import ModelOutput
from tropt.model import BaseTokenizer, LMBaseModel, LossTextAccessMixin
from tropt.model.openai.encoder import OpenAITokenizer

logger = logging.getLogger(__name__)

_LITELLM_PROXY_URL = "http://localhost:4000"

# Maximum top_logprobs supported by OpenAI (and most providers).
_MAX_TOP_LOGPROBS = 20

class LiteLLMModel(LMBaseModel, LossTextAccessMixin):
    """Model wrapper using LiteLLM as a unified interface to LLM providers.

    By default operates in **library mode**: LiteLLM routes requests directly
    to the provider based on the model name prefix (e.g. ``"openai/gpt-4o"``
    -> OpenAI, ``"anthropic/claude-3.5-sonnet"`` -> Anthropic).  API keys are
    read from environment variables (``OPENAI_API_KEY``, etc.).

    Alternatively, set ``using_litellm_proxy=True`` and ``base_url`` to
    connect via a running LiteLLM proxy server (``litellm --port 4000``),
    which is mostly used for centralized key management or shared server setups.
    """
    _N_RETRIES: int = 5
    _RETRY_STRATEGY: str = "exponential_backoff_retry"

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_concurrent_requests: int = 20,
        using_litellm_proxy: bool = False,
        **client_kwargs,
    ):
        """
        Args:
            model_name: LiteLLM model identifier, e.g. ``"openai/gpt-4o-mini"``.
            base_url: Override the provider URL.  Only needed when using a
                LiteLLM proxy or a custom endpoint.
            api_key: API key override (by default read from env vars).
            system_prompt: Optional system prompt to prepend.
            max_concurrent_requests: Maximum number of parallel requests when batching is not supported.
            using_litellm_proxy: Set True when connecting to a LiteLLM proxy
                server.  This sets ``base_url`` to ``localhost:4000`` (if not
                already provided) and tells LiteLLM the endpoint speaks the
                OpenAI protocol.
            **client_kwargs: Additional arguments for the litellm completion call.
        """
        self.model_name = model_name
        self._system_prompt = system_prompt
        self._max_concurrent_requests = max_concurrent_requests

        # Expose an OpenAI-compatible tokenizer (via tiktoken) for optimizers
        # that need token-level mutations (RS, GCGPlus, etc.).
        # OpenAI models get their exact tokenizer; others fall back to cl100k_base.
        bare_name = model_name.split("/", 1)[-1]  # e.g., "openai/gpt-4o-mini" -> "gpt-4o-mini"
        self._tokenizer = OpenAITokenizer(bare_name)

        self._client_kwargs = client_kwargs
        if using_litellm_proxy:
            self._client_kwargs.setdefault("base_url", base_url or _LITELLM_PROXY_URL)
            self._client_kwargs["custom_llm_provider"] = "openai"
        elif base_url:
            self._client_kwargs["base_url"] = base_url
        if api_key:
            self._client_kwargs["api_key"] = api_key

    @property
    def tokenizer(self) -> BaseTokenizer:
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def invoke_from_texts(
        self,
        input_texts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0,

        require_generation: bool = False,
        require_target_prefill: bool = False,
        require_first_token_logprobs: bool = False,
        **kwargs,
    ) -> ModelOutput:
        """
        Generates text completions for the given input texts using parallel execution.

        Args:
            input_texts: List of input strings.
            require_generation: Whether to perform generation.
            max_new_tokens: Maximum number of tokens to generate. Relevant when require_generation=True.
            temperature: Sampling temperature. Relevant when require_generation=True.

            require_first_token_logprobs: Whether to return log-probabilities for the first generated token. Default is False.

            require_target_prefill: Whether to prefill the target response. Currently *not supported* in this class and will raise an error.

            **kwargs: Additional arguments to pass to the litellm completion call.

        Returns:
            ModelOutput containing the generated response strings and optionally the first-token logprobs.
        """
        if require_target_prefill:
            raise ValueError("Prefill target response is not supported in LiteLLMModel.")
        assert require_generation or require_first_token_logprobs, "At least one of require_generation or require_first_token_logprobs must be True."

        # Build prompts
        prompts = [ [{"role": "user", "content": text}] for text in input_texts]
        if self._system_prompt:
            # prepend system prompt, if provided
            for prompt in prompts:
                prompt.insert(0, {"role": "system", "content": self._system_prompt})

        # Generation params:
        generation_kwargs = {
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }

        if not require_generation:
            # if generation is not needed, let's save the tokens
            generation_kwargs["max_tokens"] = 1

        if require_first_token_logprobs:
            generation_kwargs["logprobs"] = True
            generation_kwargs["top_logprobs"] = _MAX_TOP_LOGPROBS

        try:
            outputs = litellm.batch_completion(
                model=self.model_name,
                messages=prompts,
                num_retries=self._N_RETRIES,
                retry_strategy=self._RETRY_STRATEGY,
                **self._client_kwargs,
                **generation_kwargs,
                **kwargs,
            )
            responses: list[str] = _parse_responses_from_outputs(outputs)
        except Exception as e:
            logger.warning(f"LiteLLM batch completion failed: {e}")
            responses = ["" for _ in input_texts]
            outputs = []

        # Parse first-token logprobs when requested
        first_token_logprobs: Optional[List[Dict[str, float]]] = None
        if require_first_token_logprobs:
            first_token_logprobs = _parse_first_token_logprobs_from_outputs(outputs)

        # Track usage
        try:
            total_tokens = sum(
                output.usage.total_tokens
                for output in outputs
            )
        except Exception as e:
            logger.warning(f"Failed to compute token usage stats: {e}")
            total_tokens = 0
        self._update_invoke_stats(
            n_tokens=total_tokens,
            n_samples=len(input_texts),
        )

        return ModelOutput(
            generated_response_strs=responses,
            response_first_token_logprobs=first_token_logprobs,
        )


## Parsing helpers: ##

def _parse_responses_from_outputs(outputs) -> List[str]:
    """Extract generated response strings from LiteLLM/OpenAI response objects."""
    return [
        output.choices[0].message.content.strip() for output in outputs
    ]


def _parse_first_token_logprobs_from_outputs(outputs) -> List[Dict[str, float]]:
    """Extract first-token logprobs from LiteLLM/OpenAI response objects.

    Returns a list of dicts (one per sample) mapping token strings to their
    log-probabilities.  Falls back to an empty dict when logprobs are
    unavailable for a given sample.
    """
    result: List[Dict[str, float]] = []
    for output in outputs:
        logprobs_dict: Dict[str, float] = {}
        logprobs_obj = output.choices[0].logprobs
        if logprobs_obj is not None and logprobs_obj.content:
            first_token_info = logprobs_obj.content[0]
            for top_lp in first_token_info.top_logprobs:
                logprobs_dict[top_lp.token] = top_lp.logprob

        result.append(logprobs_dict)
    return result
