from typing import List, Optional

import torch

from tropt.common import ModelOutput
from tropt.model import EncoderBaseModel, LossTextAccessMixin
from tropt.model.api_retry import retry_transient


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

        # Voyage's /embeddings caps at 128 texts per call on the newer models; chunk.
        MAX_BATCH = 128
        all_embeddings: list = []
        total_tokens = 0

        def _embed_chunk(chunk):
            return self._client.embed(
                texts=chunk,
                model=self.model_name,
                input_type=text_type,  # voyage's input_type values match ours ("document"/"query")
                output_dimension=self._d_model,
            )
        _embed_chunk = retry_transient(_embed_chunk, label="voyage")

        for start in range(0, len(input_texts), MAX_BATCH):
            chunk = input_texts[start : start + MAX_BATCH]
            response = _embed_chunk(chunk)
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
