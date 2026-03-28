from typing import List, Optional

import torch

from tropt.common import ModelOutput
from tropt.model import EncoderBaseModel, LossTextAccessMixin


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

        self._client = genai.Client()
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
        task_type = self._text_to_task_type.get(text_type, None)

        import google.genai as genai  # optional dependency

        response = self._client.models.embed_content(
            contents=input_texts,
            model=self.model_name,
            config=genai.types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._d_model,
            ),
        )

        result = torch.stack(
            [torch.tensor(emb.values) for emb in response.embeddings], dim=0
        )  # shape: (n_texts, d_model)

        # Extract token count from usage metadata if available
        total_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            total_tokens = getattr(response.usage_metadata, 'total_token_count', 0)

        self._update_invoke_stats(
            n_tokens=total_tokens,
            n_samples=len(input_texts),
        )

        return ModelOutput(
            output_embeddings=result,
        )
