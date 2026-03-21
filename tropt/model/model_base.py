"""
Base definitions, classes, and mixins for targeted text models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizer

from tropt.common import DEFAULT_INIT_TRIGGER, MessageTargets, ModelOutput, Targets
from tropt.model.inputs_manager import (
    TextInputManager,
)

# ====================== Model Base Classes =======================


class BaseModel(ABC):
    def __init__(self, model_name: str):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Forward pass through the model, returns model default output (e.g., text response for LMs).
        - This method sould also update the usage stats (e.g., token counts, forward call counts, etc.)
        """
        raise NotImplementedError

    # ... prepare inputs methods will be added upon expansion ...

    # ... compute loss/grad/... methods will be added upon expansion ...

    # ... usage stats methods will be added upon expansion ...
    def get_model_name(self) -> str:
        """Returns the model identifier string."""
        return getattr(self, "_model_name", getattr(self, "model_name", type(self).__name__))

    def get_usage_stats(self) -> Dict[str, int]:
        """Returns summary of model usage statistics, namespaced under 'usage/' for W&B logging."""
        return {
            "usage/total_tokens": getattr(self, "_token_used", 0),
            "usage/forward_calls": getattr(self, "_forward_call_count", 0),
            "usage/forward_samples": getattr(self, "_forward_sample_count", 0),
            "usage/grad_calls": getattr(self, "_grad_call_count", 0),
            "usage/grad_samples": getattr(self, "_grad_sample_count", 0),
        }

    def _update_usage_stats(
        self,
        tokens: int = 0,
        forward_calls: int = 0,
        forward_samples: int = 0,
        grad_calls: int = 0,
        grad_samples: int = 0,
    ):
        """Updates the usage statistics.

        Call this immediately after any model call (e.g. self._model(...)),
        at the same call site. It is best to AVOID calling from higher-level wrappers (compute_loss_from_tokens,
        compute_grad_from_tokens, etc.) to avoid double-counting.
        """
        if not hasattr(self, "_token_used"):
            # Initialize stats if not present
            self._token_used = 0
            self._forward_call_count = 0
            self._forward_sample_count = 0
            self._grad_call_count = 0
            self._grad_sample_count = 0

        self._token_used += tokens
        self._forward_call_count += forward_calls
        self._forward_sample_count += forward_samples
        self._grad_call_count += grad_calls
        self._grad_sample_count += grad_samples

    def reset_usage_stats(self):
        """Resets the usage statistics."""
        self._token_used = 0
        self._forward_call_count = 0
        self._forward_sample_count = 0
        self._grad_call_count = 0
        self._grad_sample_count = 0

    @property
    def device(self) -> torch.device:
        """
        Returns the default device.
        Should be overriden by the device in which the model is loaded.
        """
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

## -------- Base models by model type ------- ##
class LMBaseModel(BaseModel):
    """Language model base class."""

    def __call__(
        self,
        input_texts: List[str],
        return_full_output: bool = False,
        **kwargs
    ) -> List[str] | ModelOutput:
        """
        Generate text completions for the given input texts.

        Args:
            input_texts (List[str]): List of input prompt strings to generate completions for.
            return_full_output (bool): If True, returns the full ModelOutput. If False, returns just the generated response strings.
        """
        result = self.invoke_from_texts(input_texts=input_texts, do_generate=True, **kwargs)
        if return_full_output:
            return result
        return result.generated_response_strs

    @abstractmethod
    def invoke_from_texts(
        self,
        input_texts: List[str],

        message_targets: Optional[MessageTargets] = None,
        do_prefill_target_response: bool = False,
        do_generate: bool = False,
        **kwargs
    ) -> ModelOutput:
        """
        Generates text completions for the given input texts.

        Args:
            input_texts (List[str]): List of input strings.
            message_targets (Optional[MessageTargets]): Targets for the messages.
            do_prefill_target_response (bool): Whether to prefill the target response from `message_targets`, and return the corresponding logits (e.g., for LMs).
            do_generate (bool): Whether to perform autoregressive generation after the forward pass (for LMs).

        Always returns ModelOutput with at least `generated_response_strs` populated.
        This method also updates the usage stats (e.g., token counts, forward call counts, etc.).
        """
        raise NotImplementedError


class EncoderBaseModel(BaseModel):
    """Encoder model base class."""

    def __call__(
        self,
        input_texts: List[str],
        return_full_output: bool = False,
        **kwargs
    ) -> Union[Float[Tensor, "n_texts d_model"], ModelOutput]:
        """
        Computes encoder embeddings for the given input texts.

        Args:
            input_texts (List[str]): List of input strings to compute embeddings for.
            return_full_output (bool): If True, returns the full ModelOutput. If False, returns just the output embeddings.
        """
        result = self.invoke_from_texts(input_texts=input_texts, **kwargs)
        if return_full_output:
            return result
        return result.output_embeddings

    @abstractmethod
    def invoke_from_texts(
        self,
        input_texts: List[str],
        **kwargs
    ) -> ModelOutput:
        """
        Computes encoder embeddings for the given input texts.
        Always returns ModelOutput with at least `output_embeddings` populated.
        This method also updates the usage stats (e.g., token counts, forward call counts, etc.).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def d_model(self) -> int:
        """Returns the dimensionality of the output embeddings."""
        raise NotImplementedError


# ====================== Tokenzier base classes ===================

class BaseTokenizer(ABC):
    """
    Abstract base class for tokenizers to ensure a unified interface 
    compatible with Hugging Face-style usage.
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        pass

    @abstractmethod
    def __call__(
        self,
        text: str | List[str],
        return_tensors: Literal["list", "pt", "np"] = "list",
        **kwargs,
    ) -> BatchEncoding:
        """
        Main entry point for tokenization.
        Should return a BatchEncoding containing 'input_ids'.
        """
        pass

    @abstractmethod
    def decode(self, ids: int | List[int] | torch.Tensor, **kwargs) -> str:
        """Converts token IDs back to a string."""
        pass

    @abstractmethod
    def encode(self, text: str | List[str], **kwargs) -> List[int]:
        """Converts a string to token IDs."""
        pass

    @abstractmethod
    def batch_decode(self, ids: List[int] | List[List[int]] | torch.Tensor, **kwargs) -> List[str]:
        """Converts a batch of token IDs back to a list of strings."""
        pass

    @property
    def name_or_path(self) -> str:
        return "unknown"
