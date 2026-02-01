from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Union

import torch
from jaxtyping import Float, Int
from torch import Tensor

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER

# ======================= Slice Keys Enum =======================

class SliceKey(str, Enum):
    """
    Enum for standardized slice keys used in input embeddings.

    These slices mark different regions in the tokenized input sequence:
    - INPUT_BEFORE: Tokens before the trigger (formerly 'chat_template_before')
    - TRIGGER: The optimized trigger tokens (formerly 'adv')
    - INPUT_AFTER: Tokens after the trigger in the template (formerly 'chat_template_after')
    - INPUT_LAST_TOKEN: The last token of the input sequence (formerly 'last_input_token')
    - APPENDED: Optional tokens appended at the end (e.g., target outputs for LMs)
    """
    TRIGGER = "trigger"  # The optimized trigger tokens
    INPUT_BEFORE = "input_before"  # Tokens before the trigger
    INPUT_AFTER = "input_after"  # Tokens after the trigger
    INPUT_LAST_TOKEN = "input_last_token"  # Last input token
    APPENDED = "appended"  # Appended tokens (if any); a.k.a. prefilled tokens


class TargetKey(str, Enum):
    """
    Enum for standardized target entry keys used in TargetsDict.

    These keys identify different types of target data used by loss functions:

    - TARGET_OUTPUTS: Raw text target outputs (List[str])
      Format: List of strings, one per message
      Shape: n_messages strings
      Used by: Language models for target matching

    - TARGET_OUTPUTS_TOKS: Tokenized target outputs (List[Tensor] or Tensor)
      Format: List of token ID tensors or batched tensor
      Shape: List of (target_seq_len,) or (n_messages, target_seq_len)
      Used by: Language models for computing cross-entropy loss

    - TARGET_VECTORS: Target embedding vectors (Tensor)
      Format: Dense embedding vectors
      Shape: (n_messages, d_model)
      Used by: Encoder models for similarity-based losses

    - TARGET_DIRECTIONS: Target directions in activation space (Tensor)
      Format: Direction vectors for activation steering
      Shape: (n_messages, d_model) or (n_messages, n_layers, d_model)
      Used by: Steering losses (e.g., representation engineering)

    - SLICES: Slice information for input regions (List[List[Dict]])
      Format: Nested list of slice dictionaries
      Shape: n_messages lists dicts
      Used by: Internal bookkeeping for token position tracking
    """
    # TODO make everyone use this enum instead of hardcoding strings everywhere!
    TARGET_RESPONSE_STRS = "target_outputs"  # target_outputs

    TARGET_RESPONSE_TOKS = "target_outputs_toks" # target_response_toks
    TARGET_VECTORS = "target_vectors"
    TARGET_DIRECTIONS = "target_directions"
    SLICES = "slices"

# ======================= Common input types =======================
TokenTrigger = Float[Tensor, "1 trigger_seq_len"]
TokenTriggerCandidates = Float[Tensor, "n_candidates trigger_seq_len"]

# ======================= Target Types and Utils =======================

## A dict of each target; where each message is mapped to its target tensor/string/etc
TargetsDict = Dict[str,
    List[str]   # list of length n_messages
      | Float[Tensor, "n_messages target_seq_len"]
      | Float[Tensor, "n_messages d_model"],
]  # each entry has n_messages elements, each elenents has the target data.

## A dict for each target; where each message is mapped
# to a repeated _batch_ of its target tensor/string/etc
BatchedTargetsDict = Dict[str,
    List[List[str]]   # list of n_messages batches of strings
      | Float[Tensor, "n_messages bsz target_seq_len"]
      | List[Float[Tensor, "bsz target_seq_len"]]
      | Float[Tensor, "n_messages bsz d_model"]
]  # each entry has n_messages, each a batch of identical targets

## A dict for each target; where _a single pre-selected message_ is mapped
#  to a batch of its target tensor/string/etc
MessageBatchedTargetsDict = Dict[str,
    List[str]   # of length bsz
      | Float[Tensor, "bsz target_seq_len"]
      | List[Float[Tensor, "target_seq_len"]]  # of length bsz
      | Float[Tensor, "bsz d_model"]
]  # each entry has a batch of targets (for the selected message)


class TargetsDictPlus(dict):
    """
    Class for extending the TargetsDict with useful utilities; this dict maps target keys to
    their corresponding target values (used for different losses) for each input message.
    While for most common logic it's sufficient to use a plain dict (TargetsDict), this class
    provides some useful utils and validations, making it a good practice to use it as the
    targets container.
    """

    def __init__(self, targets: TargetsDict = None, n_messages: int = None):
        """
        Initializes the TargetsManager with the given targets dictionary.
        Optionally provide `n_messages` to validate the targets.

        Each entry can be a tensor (shape: (n_messages, *)) or a list (e.g., of string, of tensors of varying lengths) of size n_messages.
        """
        if targets is None:
            targets = {}
        super().__init__(targets)

        if n_messages is None:
            # if not provided, infer from the an entry
            n_messages = len(next(iter(self.values())))
        self.n_messages = n_messages

        assert isinstance(self, dict)
        assert all(isinstance(k, str) for k in self.keys())
        assert all(len(val) == n_messages for val in self.values())

    def to_device(self, device: torch.device) -> "TargetsDictPlus":
        """
        Moves all the tensor targets to the specified device; inplace.
        """
        for k in self.keys():
            if isinstance(self[k], torch.Tensor):
                self[k] = self[k].to(device)
            elif isinstance(self[k], list) and isinstance(self[k][0], torch.Tensor):
                self[k] = [t.to(device) for t in self[k]]
        return self

    def __setitem__(self, key, value):
        assert len(value) == self.n_messages, f"Length of target entry for key {key} must be {self.n_messages}, but got {len(value)}."
        return super().__setitem__(key, value)

    #----------------------------------------------------------------------------#
    ## Utils for obtaining and manipulating different views of the TargetsDict: ##
    @staticmethod
    def get_expanded_with_candidates(targets: TargetsDict | "TargetsDictPlus", n_candidates: int) -> BatchedTargetsDict:
        """
        Repeats each target entry `n_repeats` times along the message dimension.
        Useful when expanding targets to match multiple candidate triggers per message.
        Returns result in a new dict (BatchedTargetsDict).
        """
        targets = targets.copy()

        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                # (n_messages, ...) -> (n_messages, n_candidates, ...)
                targets[k] = v.unsqueeze(1).expand(v.shape[0], n_candidates, *v.shape[1:])
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                # for each element: (...,) -> (n_candidates, ...)
                targets[k] = [t.unsqueeze(0).expand(n_candidates, *t.shape) for t in v]
            elif isinstance(v, list):
                targets[k] = [[elem] * n_candidates for elem in v]
            else:
                raise ValueError(f"Unsupported target type for key {k}: {type(targets[k])}")

        return targets

    @staticmethod
    def get_message_from_batched_targets(
        targets: BatchedTargetsDict,
        chosen_message_idx: int,
    ) -> MessageBatchedTargetsDict:
        """
        Selects the targets for a specific message index from the BatchedTargetsDict.
        Returns result in a new dict (MessageBatchedTargetsDict).
        """
        targets = {k: v[chosen_message_idx] for k, v in targets.items()}
        return targets

    #----------------------------------------------------------------------------


# ======================= Model Input Wrapper =======================

from pydantic import BaseModel, ConfigDict, field_validator


class ModelInput(BaseModel):
    """Standardized input container returned by InputsManager.get_triggered_inputs().

    This renders a uniform interface for model outputs, that can then be used
    to compute different losses agnostic of the underlying model type/implementation.

    Shape Notation:
        - bsz: batch size (typically n_candidates for a single message)
        - seq_len: total sequence length
        - trigger_seq_len: number of trigger tokens
        - d_model: embedding dimension

    Examples:
        >>> # Token-level input
        >>> token_input = ModelInput(
        ...     input_trigger_ids=torch.randint(0, 1000, (4, 20)),
        ...     input_embeds=torch.randn(4, 100, 768),
        ...     input_attention_mask=torch.ones(4, 100),
        ...     targets={"target_outputs_toks": target_ids}
        ... )

        >>> # Text-level input
        >>> text_input = ModelInput(
        ...     input_texts=["Text with trigger 1", "Text with trigger 2"],
        ...     targets={"target_outputs": ["Response 1", "Response 2"]}
        ... )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # === Text-level inputs (TextInputsManager) ===
    input_texts: Optional[Annotated[List[str], "bsz"]] = None
    """List of complete text strings with triggers inserted, of length batch_size.
    """

    # === Token-level inputs (TokenInputsManager) ===
    input_trigger_ids: Optional[Int[Tensor, "bsz trigger_seq_len"]] = None
    """Token IDs of the trigger candidates. Shape: (batch_size, trigger_sequence_length).

    Used by some losses to compute trigger-specific metrics (e.g., perplexity of trigger).
    """

    input_embeds: Optional[Float[Tensor, "bsz seq_len d_model"]] = None
    """Full input embeddings with trigger embeddings inserted, and potentially prefilled target tokens.
    Could be passed to model as inputs.
    Shape: (batch_size, total_sequence_length, embedding_dimension).
    """

    input_attention_mask: Optional[Float[Tensor, "bsz seq_len"]] = None
    """Binary attention mask for the input sequence.
        Passed to HuggingFace models to indicate valid token positions.
      Shape: (batch_size, total_sequence_length).
    """

    input_prefix_cache_kwargs: Optional[Dict[str, Any]] = None
    """Keyword arguments for HuggingFace's prefix caching (KV cache optimization).
    """

    # === Position information (slicing) ===
    input_slices: Optional[List[Dict[str, slice]]] = None
    """Position slices marking different regions in the input sequence.

    List of length batch_size, where each element is a dictionary mapping SliceKey
    to slice objects. Used to extract specific regions (trigger, input_before,
    input_after, appended) from model outputs like logits or hidden states.

    Example:
        [{
            SliceKey.TRIGGER: slice(10, 30),
            SliceKey.INPUT_BEFORE: slice(0, 10),
            SliceKey.INPUT_AFTER: slice(30, 50),
            SliceKey.APPENDED: slice(50, 60)
        }, ...]

    Critical for loss functions that need to identify specific token positions
    in the output (e.g., target output region for cross-entropy loss).
    """

    # === Targets (used by loss functions) ===
    targets: Optional[TargetsDict | TargetsDictPlus] = None
    """Target data required by loss functions.

    Dictionary mapping `TargetKey`s to their corresponding target values. The specific
    keys and values depend on which loss function is being used.
    """

    # === Validators ===

    @field_validator('input_texts', mode='before')
    @classmethod
    def validate_input_texts(cls, v):
        """Validate that input_texts is a list of strings."""
        if v is not None:
            if not isinstance(v, list):
                raise TypeError(f"input_texts must be a list, got {type(v)}")
            for i, text in enumerate(v):
                if not isinstance(text, str):
                    raise TypeError(f"input_texts[{i}] must be a string, got {type(text)}")
        return v

    @field_validator('input_trigger_ids', mode='before')
    @classmethod
    def validate_trigger_ids_shape(cls, v):
        """Validate that trigger IDs are 2D tensors."""
        if v is not None:
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"input_trigger_ids must be a Tensor, got {type(v)}")
            if v.ndim != 2:
                raise ValueError(
                    f"input_trigger_ids must be 2D (bsz, trigger_seq_len), got shape {v.shape}"
                )
        return v

    @field_validator('input_embeds', mode='before')
    @classmethod
    def validate_embeds_shape(cls, v):
        """Validate that input embeddings are 3D tensors."""
        if v is not None:
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"input_embeds must be a Tensor, got {type(v)}")
            if v.ndim != 3:
                raise ValueError(
                    f"input_embeds must be 3D (bsz, seq_len, d_model), got shape {v.shape}"
                )
        return v

    @field_validator('input_attention_mask', mode='before')
    @classmethod
    def validate_attention_mask_shape(cls, v):
        """Validate that attention mask is 2D tensor."""
        if v is not None:
            if not isinstance(v, torch.Tensor):
                raise TypeError(f"input_attention_mask must be a Tensor, got {type(v)}")
            if v.ndim != 2:
                raise ValueError(
                    f"input_attention_mask must be 2D (bsz, seq_len), got shape {v.shape}"
                )
        return v

    @field_validator('input_slices', mode='before')
    @classmethod
    def validate_input_slices(cls, v):
        """Validate that input_slices is a list of dicts."""
        if v is not None:
            if not isinstance(v, list):
                raise TypeError(f"input_slices must be a list, got {type(v)}")
            for i, slices_dict in enumerate(v):
                if not isinstance(slices_dict, dict):
                    raise TypeError(
                        f"input_slices[{i}] must be a dict, got {type(slices_dict)}"
                    )
        return v


# ======================= Triggered Input Managers =======================


class InputsManager(ABC):
    """
    Base class for maintaining the input template, corresponding targets, and the method for injecting triggers into the inputs.
    This class wraps `n_messages` texts and targets, and provides a unified interface for different types of inputs (e.g., text-based, token-based) used in adversarial trigger optimization.
    """

    optimized_trigger_placeholder: str = OPTIMIZED_TRIGGER_PLACEHOLDER

    def __init__(
        self,
        text_templates: List[str],  # n_messages texts
        targets: TargetsDict | TargetsDictPlus,  # n_messages elements per target entry
    ):
        raise NotImplementedError

    @abstractmethod
    def get_triggered_inputs(self, *args, **kwargs) -> ModelInput:
        raise NotImplementedError

## Text inputs manager ##
class TextInputsManager(InputsManager):
    """
    Class for maintaining text-based trigger-combined inputs (fits black-box text-level query access).
    """

    before_texts: List[str]
    after_texts: List[str]  # of length n_messages
    targets: TargetsDict | TargetsDictPlus

    def __init__(
        self,
        texts: List[str],  # n_messages texts
        targets: TargetsDict = {},  # n_messages elements per target entry
        optimized_trigger_placeholder: str = OPTIMIZED_TRIGGER_PLACEHOLDER,
    ):
        assert isinstance(texts, list), "texts must be a string or a list of strings."
        n_messages = len(texts)
        targets = TargetsDictPlus(targets, n_messages=n_messages)
        targets = targets.to_device("cuda" if torch.cuda.is_available() else "cpu")

        before_texts, after_texts = [], []
        for text in texts:
            bef, aft = text.split(optimized_trigger_placeholder)
            before_texts.append(bef)
            after_texts.append(aft)

        self.before_texts = before_texts
        self.after_texts = after_texts
        self.targets = targets

    @property
    def n_messages(self):
        return len(self.before_texts)

    def get_triggered_inputs(
        self,
        trigger_strs: List[str],
        chosen_message_idx: Optional[int] = None,
    ) -> ModelInput:
        """
        Returns a list of inputs with the given trigger strings merged in.
        The list is two-dimensional: outer list over messages, inner list over trigger variations; also, returns the corresponding targets.

        Given `chosen_message_idx`, returns only the inputs for that message (1D list), and the corresponding targets.
        """
        assert isinstance(trigger_strs, list) and all(
            isinstance(s, str) for s in trigger_strs
        ), "trigger_strs must be a list of strings."
        n_candidates = len(trigger_strs)
        inputs = []

        for message_idx in range(self.n_messages):
            inputs.append([])
            for trigger_str in trigger_strs:
                curr_text = (
                    self.before_texts[message_idx]
                    + trigger_str
                    + self.after_texts[message_idx]
                )
                inputs[-1].append(curr_text)

        # Expand the target entries accordingly
        targets: TargetsDictPlus = self.targets.copy()
        targets: BatchedTargetsDict = TargetsDictPlus.get_expanded_with_candidates(targets, n_candidates)

        # If specified, select only the chosen message's inputs and targets
        if chosen_message_idx is not None:
            inputs = inputs[chosen_message_idx]
            targets: MessageBatchedTargetsDict = TargetsDictPlus.get_message_from_batched_targets(
                targets, chosen_message_idx
            )

        return ModelInput(
            input_texts=inputs,
            targets=targets
        )

## Token inputs manager ##
class TokenInputsManager(InputsManager):
    """
    Base class for maintaining token-level trigger-combined inputs (fits models with token-level access).
    """

    before_ids: List[Float[Tensor, "bef_len"]]
    after_ids: List[Float[Tensor, "aft_len"]]  # of length n_messages
    targets: TargetsDict | TargetsDictPlus
    tokenizer: Any

    # Properties:
    vocab_size: int
    n_messages: int

