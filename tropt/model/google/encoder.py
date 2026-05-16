import random as _random
import time
from typing import List, Optional

import torch

from tropt.common import ModelOutput
from tropt.model import EncoderBaseModel, LossTextAccessMixin

_RETRY_MAX_ATTEMPTS = 6
_RETRY_BASE_DELAY = 1.5
_RETRY_CAP_DELAY = 60.0
# Per-request HTTP timeout (ms). Without this, a half-open connection or
# hung server can deadlock the run indefinitely.
_REQUEST_TIMEOUT_MS = 120_000


def _is_transient_gemini_error(e: BaseException) -> bool:
    # Transient: any httpx network error, plus google.genai errors with
    # 429/5xx status. Non-transient: 4xx (bad request, auth, etc.).
    name = type(e).__name__
    if name in {
        "ReadError",
        "WriteError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "TimeoutException",
        "NetworkError",
        "ProtocolError",
    }:
        return True
    code = getattr(e, "code", None) or getattr(e, "status_code", None)
    if isinstance(code, int) and (code == 429 or 500 <= code < 600):
        return True
    if name == "ServerError":
        return True
    return False


class EncoderGeminiModel(EncoderBaseModel, LossTextAccessMixin):
    """
    Google Gemini Encoder model wrapper, with text-query access.
    https://ai.google.dev/gemini-api/docs/embeddings
    """

    def __init__(
        self, model_name="gemini-embedding-001", d_model: int = 3072, **kwargs
    ):
        """
        Initializes the Gemini Encoder Model wrapper.

        Args:
            model_name: The name of the Gemini embedding model to use.
            d_model: The dimensionality of the embeddings (e.g., 768, 3072).

        Note:
        Requires `os.environ["GOOGLE_API_KEY"]` to be set externally.

        """
        # Import google.genai only when instantiating (optional dependency)
        from google import genai

        self._client = genai.Client(
            http_options=genai.types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
        )
        self.model_name = model_name
        self._d_model = d_model  # for gemini-embedding-001: could be 768, 1536, or 3072
        self._text_to_task_type = {
            "document": "RETRIEVAL_DOCUMENT",
            "query": "RETRIEVAL_QUERY",
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
        Generates embeddings for the given texts using the Gemini API.

        Args:
            input_texts: A list of strings to embed.
            text_type: The type of text (e.g., "document" or "query") to guide the embedding generation.

        Returns:
            A ModelOutput containing the generated embeddings.
        """
        assert text_type in (
            None,
            "document",
            "query",
        ), f"Unsupported text_type {text_type}"
        task_type = self._text_to_task_type.get(text_type) if text_type else None

        import google.genai as genai

        # Gemini's BatchEmbedContents caps at 100 requests per call; chunk.
        MAX_BATCH = 100
        all_embeddings = []
        total_tokens = 0
        for start in range(0, len(input_texts), MAX_BATCH):
            chunk = input_texts[start : start + MAX_BATCH]
            cfg = genai.types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._d_model,
            )
            response = None
            for attempt in range(_RETRY_MAX_ATTEMPTS):
                try:
                    response = self._client.models.embed_content(
                        contents=chunk, model=self.model_name, config=cfg,
                    )
                    break
                except Exception as e:
                    if (
                        attempt == _RETRY_MAX_ATTEMPTS - 1
                        or not _is_transient_gemini_error(e)
                    ):
                        raise
                    delay = min(
                        _RETRY_CAP_DELAY,
                        _RETRY_BASE_DELAY * (2**attempt) + _random.random(),
                    )
                    print(
                        f"[gemini retry] {type(e).__name__}: {str(e)[:80]} "
                        f"-> sleeping {delay:.1f}s "
                        f"(attempt {attempt + 1}/{_RETRY_MAX_ATTEMPTS})",
                        flush=True,
                    )
                    time.sleep(delay)
            assert response is not None and response.embeddings is not None, (
                "embed_content returned no embeddings"
            )
            all_embeddings.extend(response.embeddings)
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                total_tokens += getattr(
                    response.usage_metadata, "total_token_count", 0
                )

        result = torch.stack(
            [torch.tensor(emb.values) for emb in all_embeddings], dim=0
        )  # shape: (n_texts, d_model)

        self._update_invoke_stats(
            n_tokens=total_tokens,
            n_samples=len(input_texts),
        )

        return ModelOutput(
            output_embeddings=result,
        )
