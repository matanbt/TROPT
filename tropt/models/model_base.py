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

from tropt.common import DEFAULT_INIT_TRIGGER, ModelOutput, Targets
from tropt.models.inputs import (
    TextInputsManager,
    TokenInputsManager,
)

# ====================== Model Base Classes =======================


class BaseModel(ABC):
    def __init__(self, model_name: str):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Forward pass through the model, returns model default output (e.g., text response for LMs)."""
        raise NotImplementedError

    # ... prepare inputs methods will be added upon expansion ...

    # ... compute loss/grad/... methods will be added upon expansion ...

    # ... usage stats methods will be added upon expansion ...
    def get_usage_stats(self) -> Dict[str, int]:
        """Returns summary of model usage statistics."""
        return dict(
            total_tokens=getattr(self, "_token_used", 0),
            forward_calls=getattr(self, "_forward_call_count", 0),
            forward_samples=getattr(self, "_forward_sample_count", 0),
            grad_calls=getattr(self, "_grad_call_count", 0),
            grad_samples=getattr(self, "_grad_sample_count", 0),
        )

    def _update_usage_stats(
        self,
        tokens: int = 0,
        forward_calls: int = 0,
        forward_samples: int = 0,
        grad_calls: int = 0,
        grad_samples: int = 0,
    ):
        """Updates the usage statistics."""
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
            texts: List[str],
            return_full_output: bool = False,
            *args, **kwargs) -> List[str] | ModelOutput:
        """Generates text completions for the given input texts."""
        raise NotImplementedError


class EncoderBaseModel(BaseModel):
    """Encoder model base class."""

    def __call__(
        self, texts: List[str],
        return_full_output: bool = False,
        *args, **kwargs
    ) -> Union[Float[Tensor, "n_texts d_model"], ModelOutput]:
        """Generates encoder embeddings for the given input texts."""
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