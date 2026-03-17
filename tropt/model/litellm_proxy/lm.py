import logging
from typing import List, Optional

import litellm

from tropt.common import ModelOutput
from tropt.model import LMBaseModel, LossTextAccessMixin

logger = logging.getLogger(__name__)

DEFAULT_LITELLM_URL = "http://localhost:4000"


class LiteLLMModel(LMBaseModel, LossTextAccessMixin):
    """
    A model wrapper for interacting with models via the LiteLLM proxy server.
    This wrapper by default reaches out to a LiteLLM proxy server running at localhost:4000 (which in turn queries the requested model),
    but can be configured to use any LiteLLM-compatible provider by specifying the `base_url` and `api_key` parameters.

    The default setup requires a running LiteLLM proxy server:
        litellm serve --host localhost --port 4000
    """
    _N_RETRIES: int = 5
    _RETRY_STRATEGY: str = "exponential_backoff_retry"

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = DEFAULT_LITELLM_URL,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_concurrent_requests: int = 20,
        using_litellm_proxy: bool = True,
        **client_kwargs,
    ):
        """
        Args:
            model_name: The name of the model to query.
            base_url: The base URL of the LiteLLM proxy server (e.g. "http://localhost:4000") or provider.
            api_key: The API key (if required).
            system_prompt: Optional system prompt to prepend.
            max_concurrent_requests: Maximum number of parallel requests when batching is not supported.
            using_litellm_proxy: Whether to use LiteLLM proxy specific settings; should be True when connecting to a LiteLLM proxy server (default).
            **client_kwargs: Additional arguments for the litellm completion call.
        """
        self.model_name = model_name
        self._system_prompt = system_prompt
        self._max_concurrent_requests = max_concurrent_requests

        self._client_kwargs = client_kwargs
        if base_url:
            self._client_kwargs["base_url"] = base_url
        if api_key:
            self._client_kwargs["api_key"] = api_key
        if using_litellm_proxy:
            # LiteLLM proxy uses OpenAI-compatible API
            self._client_kwargs['custom_llm_provider'] = 'openai'

    def invoke_from_texts(
        self,
        input_texts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        **kwargs,
    ) -> ModelOutput:
        """
        Generates text completions for the given input texts using parallel execution.
        """

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
            responses: list[str] = [
                output.choices[0].message.content.strip() if hasattr(output, 'choices') else ""
                for output in outputs
            ]
        except Exception as e:
            logger.warning(f"LiteLLM batch completion failed: {e}")
            responses = ["" for _ in input_texts]

        # Track usage
        total_tokens = sum(
            (output.usage.total_tokens if hasattr(output, 'usage') and hasattr(output.usage, 'total_tokens') else 0)
            for output in outputs
        )
        self._update_usage_stats(
            tokens=total_tokens,
            forward_calls=1, # One batch call
            forward_samples=len(input_texts)
        )

        return ModelOutput(
            generated_response_strs=responses,
        )


