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
from tropt.model.flop_counter import ManualFlopCounter, FlopCounterBase
from tropt.model.inputs_manager import (
    TextInputManager,
)

# ====================== Model Base Classes =======================


class BaseModel(ABC):
    def __init__(self, model_name: str):
        pass

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

    _model: Optional[Any] = None
    """The underlying model object. Set by subclass ``__init__``.

    For HuggingFace models this is a ``PreTrainedModel``; for black-box
    models it may be ``None``.
    """

    _flop_counter: Optional[FlopCounterBase] = None
    """Active counter object (set by :meth:`set_flop_counting`).

    Must implement ``count_forward(n_tokens) -> int`` and
    ``count_forward_backward(n_tokens) -> int``.
    """

    def set_flop_counting(self, mode: Literal["manual", "none"] = "manual"):
        """Enable or disable FLOP counting.

        Args:
            mode: The method to use for counting FLOPs. Options:
                -> "manual": Uses a `ManualFlopCounter` that estimates FLOPs based on token counts and model architecture (follows Kaplan et al. 2020). Requires ``_model`` to be a HuggingFace ``PreTrainedModel``. This is the default.
                -> "none": Disables FLOP counting.
        
        - FLOP counting will appear in :meth:`get_usage_stats` under ``"usage/total_flops"``.
        """
        if mode == "manual":
            self._flop_counter = ManualFlopCounter(self._model)
        else:
            self._flop_counter = None

    def get_usage_stats(self) -> Dict[str, int]:
        """Returns summary of model usage statistics for logging."""
        stats: dict[str, Any | int] = {
            "total_tokens": getattr(self, "_token_used", 0),
            "forward_calls": getattr(self, "_forward_call_count", 0),
            "forward_samples": getattr(self, "_forward_sample_count", 0),
            "grad_calls": getattr(self, "_grad_call_count", 0),
            "grad_samples": getattr(self, "_grad_sample_count", 0),
        }
        if self._flop_counter is not None:
            stats["total_flops"] = getattr(self, "_total_flops", 0)
        return stats

    def _update_invoke_stats(
        self,
        *,
        n_tokens: int,
        n_samples: int,
        count_backward: bool = False,
    ):
        """Record usage stats and FLOPs for a single model invocation.

        Call this inside ``invoke_from_tokens`` / ``invoke_from_texts`` after
        each *raw* model call (!), to avoid double-counting.  It handles forward/backward counters **and** FLOP
        computation.

        Args:
            n_tokens: Total tokens processed in this invocation.
            n_samples: Batch size (number of sequences).
            count_backward: Whether this forward pass will also be
                back-propagated through (set by gradient methods).
        """
        flops = 0
        if self._flop_counter is not None:
            if count_backward:
                flops = self._flop_counter.count_forward_backward(n_tokens)
            else:
                flops = self._flop_counter.count_forward(n_tokens)

        self._update_usage_stats(
            tokens=n_tokens,
            forward_calls=1,
            forward_samples=n_samples,
            flops=flops,
            grad_calls=1 if count_backward else 0,
            grad_samples=n_samples if count_backward else 0,
        )

    def _update_usage_stats(
        self,
        tokens: int = 0,
        forward_calls: int = 0,
        forward_samples: int = 0,
        grad_calls: int = 0,
        grad_samples: int = 0,
        flops: int = 0,
    ):
        """Low-level accumulator. Prefer :meth:`_update_invoke_stats`."""
        if not hasattr(self, "_token_used"):
            self._token_used = 0
            self._forward_call_count = 0
            self._forward_sample_count = 0
            self._grad_call_count = 0
            self._grad_sample_count = 0
            self._total_flops = 0

        self._token_used += tokens
        self._forward_call_count += forward_calls
        self._forward_sample_count += forward_samples
        self._grad_call_count += grad_calls
        self._grad_sample_count += grad_samples
        self._total_flops += flops

    def reset_usage_stats(self):
        """Resets the usage statistics."""
        self._token_used = 0
        self._forward_call_count = 0
        self._forward_sample_count = 0
        self._grad_call_count = 0
        self._grad_sample_count = 0
        self._total_flops = 0


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
        result: ModelOutput = self.invoke_from_texts(input_texts=input_texts, do_generate=True, **kwargs)
        if return_full_output:
            return result
        assert result.generated_response_strs is not None, "Model did not generate response strings"
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
        This method also updates the usage stats (e.g., token counts, forward call counts, etc.). It must call `_update_invoke_stats` after each raw model call.
        """
        raise NotImplementedError


class EncoderBaseModel(BaseModel):
    """Encoder model base class."""

    def __call__(
        self,
        input_texts: List[str],
        return_full_output: bool = False,
        **kwargs
    ) -> Float[Tensor, "n_texts d_model"] | ModelOutput:
        """
        Computes encoder embeddings for the given input texts.

        Args:
            input_texts (List[str]): List of input strings to compute embeddings for.
            return_full_output (bool): If True, returns the full ModelOutput. If False, returns just the output embeddings.
        """
        result: ModelOutput = self.invoke_from_texts(input_texts=input_texts, **kwargs)
        if return_full_output:
            return result
        assert result.output_embeddings is not None, "Model did not return embeddings"
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
        It must call `_update_invoke_stats` after each raw model call.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def d_model(self) -> int:
        """Returns the dimensionality of the output embeddings."""
        raise NotImplementedError


class ClassifierBaseModel(BaseModel):
    """Classifier model base class."""

    def __call__(
        self,
        input_texts: List[str],
        return_full_output: bool = False,
        **kwargs
    ) -> Float[Tensor, "n_texts n_classes"] | ModelOutput:
        """
        Compute classification logits for the given input texts.

        Args:
            input_texts (List[str]): List of input strings to classify.
            return_full_output (bool): If True, returns the full ModelOutput. If False, returns just the class logits.
        """
        result: ModelOutput = self.invoke_from_texts(input_texts=input_texts, **kwargs)
        if return_full_output:
            return result
        assert result.output_class_logits is not None, "Model did not return class logits"
        return result.output_class_logits

    @abstractmethod
    def invoke_from_texts(
        self,
        input_texts: List[str],
        **kwargs
    ) -> ModelOutput:
        """
        Compute classification logits for the given input texts.
        Always returns ModelOutput with at least `output_class_logits` populated.
        This method also updates the usage stats.
        It must call `_update_invoke_stats` after each raw model call.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def n_classes(self) -> int:
        """Returns the number of output classes."""
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

    # --- Specific helpers (concrete, build on abstract primitives above) ---

    def encode_trigger(self, trigger_str: str) -> Int[Tensor, "trigger_seq_len"]:
        """Encode a trigger string -> 1-D tensor (no special tokens)."""
        ids = self.encode(trigger_str, add_special_tokens=False)
        return torch.tensor(ids, dtype=torch.int64)

    def decode_trigger(self, trigger_ids: Int[Tensor, "trigger_seq_len"]) -> str:
        """Decode a 1-D trigger ids tensor -> string (special tokens skipped)."""
        return self.decode(trigger_ids, skip_special_tokens=True)

    def decode_triggers(self, trigger_ids: Int[Tensor, "bsz trigger_seq_len"]) -> List[str]:
        """Batch-decode a 2-D trigger ids tensor -> list of strings (special tokens skipped)."""
        return self.batch_decode(trigger_ids, skip_special_tokens=True)


class HFTokenizerWrapper(BaseTokenizer):
    """Wraps a HuggingFace PreTrainedTokenizerBase, exposing it as a BaseTokenizer.

    All attributes not defined here are transparently forwarded to the underlying
    HF tokenizer, so existing code that accesses tokenizer internals (padding_side,
    apply_chat_template, etc.) continues to work unchanged.
    """

    def __init__(self, tokenizer):
        # Store without triggering __getattr__
        object.__setattr__(self, "_hf_tokenizer", tokenizer)

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_hf_tokenizer"), name)

    # --- BaseTokenizer abstract method implementations ---

    @property
    def vocab_size(self) -> int:
        return self._hf_tokenizer.vocab_size

    def __call__(self, text, return_tensors=None, **kwargs):
        # "list" in our abstraction maps to None (Python objects) in HF
        if return_tensors == "list":
            return_tensors = None
        return self._hf_tokenizer(text, return_tensors=return_tensors, **kwargs)

    def decode(self, ids, **kwargs) -> str:
        return self._hf_tokenizer.decode(ids, **kwargs)

    def encode(self, text, **kwargs) -> List[int]:
        return self._hf_tokenizer.encode(text, **kwargs)

    def batch_decode(self, ids, **kwargs) -> List[str]:
        return self._hf_tokenizer.batch_decode(ids, **kwargs)

    @property
    def name_or_path(self) -> str:
        return self._hf_tokenizer.name_or_path
