import random as _random
import time
from typing import List, Optional

import torch

from tropt.common import ModelOutput
from tropt.model import EncoderBaseModel, LossTextAccessMixin

_RETRY_MAX_ATTEMPTS = 6
_RETRY_BASE_DELAY = 1.5
_RETRY_CAP_DELAY = 60.0


def _is_transient_voyage_error(e: BaseException) -> bool:
    # Transient: voyageai connection / timeout / 5xx / 429, plus the underlying
    # requests/urllib3/socket errors that surface through them.
    # Non-transient: 4xx (auth, bad request, etc.).
    name = type(e).__name__
    if name in {
        "APIConnectionError",
        "Timeout",
        "ServiceUnavailableError",
        "RateLimitError",
        "ConnectionError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "ProtocolError",
        "ReadTimeout",
        "WriteTimeout",
        "ConnectTimeout",
        "ChunkedEncodingError",
    }:
        return True
    code = (
        getattr(e, "http_status", None)
        or getattr(e, "status_code", None)
        or getattr(e, "code", None)
    )
    if isinstance(code, int) and (code == 429 or 500 <= code < 600):
        return True
    return False


class EncoderVoyageModel(EncoderBaseModel, LossTextAccessMixin):
    """
    Voyage AI Encoder model wrapper, with text-query access.
    https://docs.voyageai.com/docs/embeddings
    """

    def __init__(
        self,
        model_name: str = "voyage-4",
        d_model: int = 1024,
        **kwargs,
    ):
        """
        Initializes the Voyage Encoder Model wrapper.

        Args:
            model_name: The name of the Voyage embedding model to use.
            d_model: The dimensionality of the embeddings. For voyage-4
                / voyage-3-large: supports 256, 512, 1024 (default), 2048.

        Note:
        Requires `os.environ["VOYAGE_API_KEY"]` to be set externally.
        """
        # Import voyageai only when instantiating (optional dependency)
        import voyageai

        self._client = voyageai.Client()
        self.model_name = model_name
        self._d_model = d_model
        self._text_to_input_type = {
            "document": "document",
            "query": "query",
        }

    @property
    def d_model(self) -> int:
        return self._d_model

    def invoke_from_texts(
        self,
        input_texts: List[str],
        text_type: Optional[str] = None,
        **kwargs,
    ) -> ModelOutput:
        """
        Generates embeddings for the given texts using the Voyage API.

        Args:
            input_texts: A list of strings to embed.
            text_type: The type of text (e.g., "document" or "query") to guide
                the embedding generation.

        Returns:
            A ModelOutput containing the generated embeddings.
        """
        assert text_type in (
            None,
            "document",
            "query",
        ), f"Unsupported text_type {text_type}"
        input_type = self._text_to_input_type.get(text_type) if text_type else None

        # Voyage's /embeddings caps at 128 texts per call on the newer models; chunk.
        MAX_BATCH = 128
        all_embeddings: list = []
        total_tokens = 0
        for start in range(0, len(input_texts), MAX_BATCH):
            chunk = input_texts[start : start + MAX_BATCH]
            response = None
            for attempt in range(_RETRY_MAX_ATTEMPTS):
                try:
                    response = self._client.embed(
                        texts=chunk,
                        model=self.model_name,
                        input_type=input_type,
                        output_dimension=self._d_model,
                    )
                    break
                except Exception as e:
                    if (
                        attempt == _RETRY_MAX_ATTEMPTS - 1
                        or not _is_transient_voyage_error(e)
                    ):
                        raise
                    delay = min(
                        _RETRY_CAP_DELAY,
                        _RETRY_BASE_DELAY * (2**attempt) + _random.random(),
                    )
                    print(
                        f"[voyage retry] {type(e).__name__}: {str(e)[:80]} "
                        f"-> sleeping {delay:.1f}s "
                        f"(attempt {attempt + 1}/{_RETRY_MAX_ATTEMPTS})",
                        flush=True,
                    )
                    time.sleep(delay)
            assert response is not None and response.embeddings is not None, (
                "voyage embed returned no embeddings"
            )
            all_embeddings.extend(response.embeddings)
            total_tokens += getattr(response, "total_tokens", 0)

        result = torch.tensor(all_embeddings, dtype=torch.float32)

        self._update_invoke_stats(
            n_tokens=total_tokens,
            n_samples=len(input_texts),
        )

        return ModelOutput(
            output_embeddings=result,
        )
