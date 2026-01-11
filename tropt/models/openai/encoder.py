from typing import List, Optional

import torch
from jaxtyping import Float
from openai import OpenAI
from torch import Tensor
from tenacity import retry, stop_after_attempt, wait_exponential

from tropt.models.base import EncoderBaseModel, LossTextAccessMixin


class OpenAIEncoderModel(EncoderBaseModel, LossTextAccessMixin):
    """
    OpenAI Encoder model wrapper for embedding generation via the OpenAI API.
    https://platform.openai.com/docs/guides/embeddings
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        d_model: Optional[int] = None,
        api_key: str = None,
        base_url: str = None,
        **kwargs,
    ):
        """
        Initializes the OpenAI Encoder Model wrapper.

        Args:
            model_name: The name of the OpenAI embedding model to use.
            d_model: The dimensionality of the embeddings. If None, it is deduced via a dummy API call.
            api_key: The OpenAI API key. If None, it will be read from the OPENAI_API_KEY environment variable.
            base_url: Optional base URL for the OpenAI client. If None, the default OpenAI API URL is used.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

        if d_model is None:
            # Deduce d_model via a dummy request
            try:
                # We use a single token to check the dimensionality
                response = self.client.embeddings.create(
                    input="test",
                    model=self.model_name,
                )
                d_model = len(response.data[0].embedding)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to deduce d_model for {model_name}. "
                    f"Please specify d_model explicitly or check your API connection. Error: {e}"
                ) from e

        self.d_model = d_model

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5)
    )
    def __call__(
        self, texts: List[str], **kwargs
    ) -> Float[Tensor, "n_texts d_model"]:
        """
        Generates embeddings for the given texts using the OpenAI API.

        Args:
            texts: A list of strings to embed.

        Returns:
            A tensor containing the generated embeddings.
        """
        # Note: OpenAI's API handles batches of texts
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
            **kwargs
        )

        embeddings = [data.embedding for data in response.data]
        result = torch.tensor(embeddings, dtype=torch.float32)

        # Update token usage
        self._update_usage_stats(
            tokens=response.usage.total_tokens,
            forward_calls=1,
            forward_samples=len(texts)
        )

        return result

