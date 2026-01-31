from typing import List

import torch
from jaxtyping import Float
from torch import Tensor

from tropt.models import EncoderBaseModel, LossTextAccessMixin


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
        """
        # Import google.genai only when instantiating (optional dependency)
        from google import genai

        # os.environ["GOOGLE_API_KEY"] = ...  # required to be set externally

        self.client = genai.Client()
        self.model_name = model_name
        self.d_model = d_model  # for gemini-embedding-001: could be 768, 1536, or 3072
        self.text_to_task_type = {
            "document": "RETRIEVAL_DOCUMENT",
            "query": "RETRIEVAL_QUERY",
        }

    def __call__(
        self, texts: List[str], text_type: str = None
    ) -> Float[Tensor, "n_texts d_model"]:
        """
        Generates embeddings for the given texts using the Gemini API.

        Args:
            texts: A list of strings to embed.
            text_type: The type of text (e.g., "document" or "query") to guide the embedding generation.

        Returns:
            A tensor containing the generated embeddings.
        """
        assert text_type in (
            None,
            "document",
            "query",
        ), f"Unsupported text_type {text_type}"
        task_type = self.text_to_task_type.get(text_type, None)

        import google.genai as genai  # optional dependency

        response = self.client.models.embed_content(
            contents=texts,
            model=self.model_name,
            config=genai.types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.d_model,
            ),
        )

        result = torch.stack(
            [torch.tensor(emb.values) for emb in response.embeddings], dim=0
        )  # shape: (n_texts, d_model)

        # Extract token count from usage metadata if available
        total_tokens = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            total_tokens = getattr(response.usage_metadata, 'total_token_count', 0)

        self._update_usage_stats(
            tokens=total_tokens,
            forward_calls=1,
            forward_samples=len(texts)
        )

        return result
