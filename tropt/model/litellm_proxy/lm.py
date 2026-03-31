import logging
from typing import Dict, List, Optional

import litellm

from tropt.common import ModelOutput
from tropt.model import LMBaseModel, LossTextAccessMixin

logger = logging.getLogger(__name__)

DEFAULT_LITELLM_URL = "http://localhost:4000"

# Maximum top_logprobs supported by OpenAI (and most providers).
_MAX_TOP_LOGPROBS = 20

# TODO if we load openai model, we can also add tokenizer; otherwise let's add here a default HF tokenizer (what's the most common tokenizer you can think of in HF? should we just default to OpenAI tokenizer? that's also possible by me!)

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

        do_generate: bool = False,  # TODO change to require_generation
        do_prefill_target_response: bool = False,
        do_first_token_logprobs: bool = False,
        **kwargs,
    ) -> ModelOutput:
        """
        Generates text completions for the given input texts using parallel execution.

        Args:
            input_texts: List of input strings.
            do_generate: Whether to perform generation. Currently we default to performing genertion. By default we perform generation.
            max_new_tokens: Maximum number of tokens to generate for each input. Relevnt for generation.
            temperature: Sampling temperature for generation. Relevnt for generation.

            do_first_token_logprobs: Whether to return log-probabilities for the first generated token. Default is False.

            do_prefill_target_response: Whether to prefill the target response. Currently *not supported* in this class and will raise an error.
            
            **kwargs: Additional arguments to pass to the litellm completion call. 
    
        Returns:
            ModelOutput containing the generated response strings and optionally the first-token logprobs.
        """
        if do_prefill_target_response:
            raise ValueError("Prefill target response is not supported in LiteLLMModel.")
        assert do_generate or do_first_token_logprobs, "At least one of do_generate or do_first_token_logprobs must be True."

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

        if not do_generate:
            # if generation is not needed, let's save the tokens
            generation_kwargs["max_tokens"] = 1

        if do_first_token_logprobs:
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
        if do_first_token_logprobs:
            first_token_logprobs = _parse_first_token_logprobs_from_outputs(outputs)

        # Track usage
        # TODO this can silently fail - not good. we should wrap with `try` and have a warning
        total_tokens = sum(
            (output.usage.total_tokens if hasattr(output, 'usage') and hasattr(output.usage, 'total_tokens') else 0)
            for output in outputs
        )
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
