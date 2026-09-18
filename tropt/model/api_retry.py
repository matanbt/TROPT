"""Shared transient-error retry for the API-backed model wrappers.

Each SDK (openai / google-genai / voyageai) defines its own exception classes but
they all fail the same two ways: a network-level error, or an HTTP 429/5xx. We
match on exception *class name* (so no SDK needs importing here) plus the status
code, and leave 4xx alone — retrying an auth failure just wastes the backoff.

``tenacity`` is imported lazily so it stays an optional dependency of the API
extras rather than a core one.
"""

import logging

logger = logging.getLogger(__name__)

# Union of the network-error class names raised by httpx, requests/urllib3, and
# the three SDKs' own wrappers.
_TRANSIENT_EXC_NAMES = frozenset({
    "APIConnectionError",
    "ChunkedEncodingError",
    "ConnectError",
    "ConnectionAbortedError",
    "ConnectionError",
    "ConnectionResetError",
    "ConnectTimeout",
    "NetworkError",
    "PoolTimeout",
    "ProtocolError",
    "RateLimitError",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "ServerError",
    "ServiceUnavailableError",
    "Timeout",
    "TimeoutException",
    "WriteError",
    "WriteTimeout",
})


def is_transient_api_error(e: BaseException) -> bool:
    """Whether `e` is worth retrying: network blip, rate limit, or 5xx."""
    if type(e).__name__ in _TRANSIENT_EXC_NAMES:
        return True
    code = (
        getattr(e, "http_status", None)
        or getattr(e, "status_code", None)
        or getattr(e, "code", None)
    )
    return isinstance(code, int) and (code == 429 or 500 <= code < 600)


def retry_transient(fn, label: str, attempts: int = 6):
    """Wrap `fn` with exponential backoff on transient API errors.

    Args:
        label: Short backend name used in the retry log line (e.g. "gemini").
    """
    from tenacity import (
        retry,
        retry_if_exception,
        stop_after_attempt,
        wait_random_exponential,
    )

    def _log(retry_state):
        e = retry_state.outcome.exception()
        logger.warning(
            "[%s retry] %s: %s -> sleeping %.1fs (attempt %d)",
            label, type(e).__name__, str(e)[:80],
            retry_state.next_action.sleep, retry_state.attempt_number,
        )

    return retry(
        retry=retry_if_exception(is_transient_api_error),
        wait=wait_random_exponential(multiplier=1.5, max=60),
        stop=stop_after_attempt(attempts),
        before_sleep=_log,
        reraise=True,
    )(fn)
