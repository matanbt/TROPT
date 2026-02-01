from abc import ABC, abstractmethod


from enum import Enum
from typing import Annotated, Any, Dict, List, Optional, Union

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict, field_validator
from torch import Tensor


# TODO arrange the comments and structure here

# =========== Common constants and utilities for TTOP ==========
# Defines a placeholder string for optimized triggers
OPTIMIZED_TRIGGER_PLACEHOLDER = "{{OPTIMIZED_TRIGGER}}"

# Default initial trigger
DEFAULT_INIT_TRIGGER = ("! " * 20).strip()

# ======================= Common input types =======================
TokenTrigger = Float[Tensor, "1 trigger_seq_len"]
TokenTriggerCandidates = Float[Tensor, "n_candidates trigger_seq_len"]


## A dict of each target; where each message is mapped to its target tensor/string/etc

## TODO deprecate for Targets class
TargetsDict = Dict["TargetKey",
    List[str]   # list of length n_messages
      | Int[Tensor, "n_messages target_seq_len"]
      | Float[Tensor, "n_messages d_model"],
]  # each entry has n_messages elements, each elenents has the target data.

#### TODO remove
## A dict for each target; where each message is mapped
# to a repeated _batch_ of its target tensor/string/etc
# BatchedTargetsDict = Dict[str,
#     List[List[str]]   # list of n_messages batches of strings
#       | Float[Tensor, "n_messages bsz target_seq_len"]
#       | List[Float[Tensor, "bsz target_seq_len"]]
#       | Float[Tensor, "n_messages bsz d_model"]
# ]  # each entry has n_messages, each a batch of identical targets

## A dict for each target; where _a single pre-selected message_ is mapped
#  to a batch of its target tensor/string/etc
# MessageBatchedTargetsDict = Dict[str,
#     List[str]   # of length bsz
#       | Float[Tensor, "bsz target_seq_len"]
#       | List[Float[Tensor, "target_seq_len"]]  # of length bsz
#       | Float[Tensor, "bsz d_model"]
# ]  # each entry has a batch of targets (for the selected message)

## A dict for each target; where _a single pre-selected message_ is mapped
#  to its target tensor/string/etc
MessageTargetsDict = Dict["TargetKey",
    str
      | Int[Tensor, "target_seq_len"]
      | Float[Tensor, "d_model"],
]  # each entry has a batch of targets (for the selected message)


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


# TODO deprecate for targets pydantic class!
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
    TARGET_RESPONSE_STRS = "target_response_strs"
    """Raw text target outputs (List[str])
      Format: List of strings, one per message
      Shape: n_messages strings
      Used by: Language models for target matching
    """

    TARGET_RESPONSE_TOKS = "target_response_toks"
    """Tokenized target outputs (List[Tensor] or Tensor)
      Format: List of token ID tensors or batched tensor
      Shape: (n_messages, target_seq_len) or, when message is selected, (target_seq_len,)
      Used by: Language models for computing cross-entropy loss
    """

    TARGET_VECTORS = "target_vectors"
    """Target embedding vectors (Tensor)
        Format: Dense embedding vectors
        Shape: (n_messages, d_model)
        Used by: Encoder models for similarity-based losses
    """

    TARGET_DIRECTIONS = "target_directions"
    """Target directions in activation space (Tensor)
        Format: Direction vectors for activation steering
        Shape: (n_messages, d_model) or (n_messages, n_layers, d_model)
        Used by: Steering losses (e.g., representation engineering)
    """


class MessageTargets(BaseModel):
    """Targets for a single selected message."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_response_strs: Optional[str] = None
    """Raw text target response for this message.
    """

    target_response_toks: Optional[Int[Tensor, "target_seq_len"]] = None
    """Tokenized target response for this message.
    """

    target_vectors: Optional[Float[Tensor, "d_model"]] = None
    """Target embedding vector for this message.
    """

    target_directions: Optional[Float[Tensor, "d_model"]] = None
    """Target direction in activation space for this message.
    Used by steering losses (e.g., representation engineering).
    """


class Targets(BaseModel):
    """Targets for all messages. Each field has an an initial n_messages dimension.

    Typically only one or two of these fields need to be provided depending
    on the loss function being used.
    For example, a standard LM jailbreak only needs `target_response_strs` (which will be
    tokenized internally).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_response_strs: Optional[Annotated[List[str], "n_messages"]] = None
    """Raw text target outputs, one per message.
    
    List is of length n_messages.
    Used by: Language models for target matching. Will be tokenized
    internally to produce `target_response_toks` if not provided directly.
    """

    target_response_toks: Optional[Int[Tensor, "n_messages target_seq_len"] | Annotated[List[Int[Tensor, "target_seq_len"]], "n_messages"]] = None
    """Tokenized target outputs, one per message.

    Shape: (n_messages, target_seq_len) OR List of length n_messages, 
    each of (potentially different) shape (target_seq_len,)
    Used by: Language models for computing cross-entropy loss.
    """

    target_vectors: Optional[Float[Tensor, "n_messages d_model"]] = None
    """Target embedding vectors, one per message.

    Shape: (n_messages, d_model)
    Used by: Encoder models for similarity-based losses.
    """

    target_directions: Optional[Float[Tensor, "n_messages d_model"]] = None
    """Target directions in activation space, one per message.

    Shape: (n_messages, d_model)
    Used by: Steering losses (e.g., refusal suppression).
    Note: if you need per-layer directions, store as (n_messages, n_layers, d_model)
    and update this annotation accordingly.
    """

    @property
    def n_messages(self) -> int:
        for field_name in self.model_fields_set:
            val = getattr(self, field_name)
            if val is not None:
                return len(val)
        raise ValueError("No targets set")

    def select_message(self, idx: int) -> "MessageTargets":
        return MessageTargets(
            **{k: v[idx] for k, v in self if v is not None}
        )

    def to_device(self, device: torch.device) -> "Targets":
        updates = {}
        for k, v in self:
            if isinstance(v, Tensor):
                updates[k] = v.to(device)
            if isinstance(v, list) and isinstance(v[0], Tensor):
                updates[k] = [t.to(device) for t in v]
        return self.model_copy(update=updates)


# ======================= Model Input Wrapper =======================

class ModelInput(BaseModel):
    """Standardized input container returned by InputsManager.get_triggered_inputs().

    This renders a uniform interface for model outputs, that can then be used
    to compute different losses agnostic of the underlying model type/implementation.

    The convention is that such object conveys the data of a single message, without
    mixing multiple messages.

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
        ...     targets={TargetKey.TARGET_RESPONSE_TOKS: target_ids}
        ... )

        >>> # Text-level input
        >>> text_input = ModelInput(
        ...     input_texts=["Text with trigger 1", "Text with trigger 2"],
        ...     targets={TargetKey.TARGET_RESPONSE_STRS: ["Response 1", "Response 2"]}
        ... )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # === Text-level inputs (TextInputsManager) ===
    input_texts: Optional[Annotated[List[str], "bsz"]] = None
    """List of complete text strings with triggers inserted, of length batch_size.
    """

    input_trigger_strs: Optional[Annotated[List[str], "bsz"]] = None
    """List of trigger strings used in the inputs, of length batch_size.
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

    input_attention_mask: Optional[Int[Tensor, "bsz seq_len"]] = None
    """Binary attention mask for the input sequence.
        Passed to HuggingFace models to indicate valid token positions.
      Shape: (batch_size, total_sequence_length).
    """

    input_prefix_cache_kwargs: Optional[Dict[str, Any]] = None
    """Keyword arguments for HuggingFace's prefix caching (KV cache optimization).
    """

    # === Position information (slicing) ===
    input_slices: Optional[Dict[SliceKey, Optional[slice]]] = None
    """Position slices marking different regions in the input sequence.

    List of length batch_size, where each element is a dictionary mapping SliceKey
    to slice objects. Used to extract specific regions (trigger, input_before,
    input_after, appended) from model outputs like logits or hidden states.

    Example:
        >>> input_slices = {
                SliceKey.TRIGGER: slice(10, 30),
                SliceKey.INPUT_BEFORE: slice(0, 10),
                SliceKey.INPUT_AFTER: slice(30, 50),
                SliceKey.APPENDED: slice(50, 60)
            }

    Critical for loss functions that need to identify specific token positions
    in the output (e.g., target output region for cross-entropy loss).
    """

    # === Targets (used by loss functions) ===
    # targets: Optional[MessageTargetsDict] = None   # <-- deprecated 
    targets: Optional[Targets] = None  
    """Target data required by loss functions.

    Dictionary mapping `TargetKey`s to their corresponding target values. The specific
    keys and values depend on which loss function is being used.

    Expects a single message's targets.
    """

    # === Validators ===

    # @field_validator('input_texts', mode='before')
    # @classmethod
    # def validate_input_texts(cls, v):
    #     """Validate that input_texts is a list of strings."""
    #     if v is not None:
    #         if not isinstance(v, list):
    #             raise TypeError(f"input_texts must be a list, got {type(v)}")
    #         for i, text in enumerate(v):
    #             if not isinstance(text, str):
    #                 raise TypeError(f"input_texts[{i}] must be a string, got {type(text)}")
    #     return v

    # @field_validator('input_trigger_ids', mode='before')
    # @classmethod
    # def validate_trigger_ids_shape(cls, v):
    #     """Validate that trigger IDs are 2D tensors."""
    #     if v is not None:
    #         if not isinstance(v, torch.Tensor):
    #             raise TypeError(f"input_trigger_ids must be a Tensor, got {type(v)}")
    #         if v.ndim != 2:
    #             raise ValueError(
    #                 f"input_trigger_ids must be 2D (bsz, trigger_seq_len), got shape {v.shape}"
    #             )
    #     return v

    # @field_validator('input_embeds', mode='before')
    # @classmethod
    # def validate_embeds_shape(cls, v):
    #     """Validate that input embeddings are 3D tensors."""
    #     if v is not None:
    #         if not isinstance(v, torch.Tensor):
    #             raise TypeError(f"input_embeds must be a Tensor, got {type(v)}")
    #         if v.ndim != 3:
    #             raise ValueError(
    #                 f"input_embeds must be 3D (bsz, seq_len, d_model), got shape {v.shape}"
    #             )
    #     return v

    # @field_validator('input_attention_mask', mode='before')
    # @classmethod
    # def validate_attention_mask_shape(cls, v):
    #     """Validate that attention mask is 2D tensor."""
    #     if v is not None:
    #         if not isinstance(v, torch.Tensor):
    #             raise TypeError(f"input_attention_mask must be a Tensor, got {type(v)}")
    #         if v.ndim != 2:
    #             raise ValueError(
    #                 f"input_attention_mask must be 2D (bsz, seq_len), got shape {v.shape}"
    #             )
    #     return v

    # @field_validator('input_slices', mode='before')
    # @classmethod
    # def validate_input_slices(cls, v):
    #     """Validate that input_slices is a list of dicts."""
    #     if v is not None:
    #         if not isinstance(v, dict):
    #             raise TypeError(f"input_slices must be a dict, got {type(v)}")
    #         for k, v in v.items():
    #             if not isinstance(k, str) and k not in SliceKey:
    #                 raise TypeError(
    #                     f"input_slices keys must be a SliceKey, got {k}"
    #                 )
    #             if not isinstance(v, slice):
    #                 raise TypeError(
    #                     f"input_slices[{k}] must be a slice, got {type(v)}"
    #                 )
    #     return v


# ======================= Model Output Wrapper =======================

class ModelOutput(BaseModel):
    """Standardized output container for all model types in TROPT.

    This renders a uniform interface for model outputs, that can then be used
    to compute different losses agnostic of the underlying model type/implementation.

    Shape Notation:
        - bsz: batch size (number of candidates or samples in a batch)
        - n_layers: number of model layers
        - n_heads: number of attention heads per layer
        - seq_len: total sequence length
        - response_len: length of generated response (variable per sample)
        - full_seq_len: full sequence length including prompt and generation
        - vocab_size: vocabulary size
        - d_model: model embedding dimension

    Examples:
        >>> # Encoder model output
        >>> encoder_output = ModelOutput(output_embeddings=torch.randn(4, 768))

        >>> # Language model output with logits
        >>> lm_output = ModelOutput(
        ...     output_logits=torch.randn(2, 50, 32000),
        ...     generated_response_strs=["Response 1", "Response 2"]
        ... )

        >>> # Full output with hidden states and attentions
        >>> full_output = ModelOutput(
        ...     output_logits=logits,
        ...     output_hidden_states=hidden_states,
        ...     output_attentions=attentions,
        ...     generated_response_strs=responses
        ... )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # === Embedding outputs (Encoder models) ===
    output_embeddings: Optional[Float[Tensor, "bsz d_model"]] = None
    """Pooled output embeddings from encoder models.
    Shape: (batch_size, d_model)
    """

    # === Logits (Language models) ===
    output_logits: Optional[Float[Tensor, "bsz seq_len vocab_size"]] = None
    """Full sequence logits from language models (including both inputs and outputs).
    """

    response_logits: Optional[Float[Tensor, "bsz response_seq_len vocab_size"]] = None
    """Logits corresponding to the prefilled response portion of the sequence.
    """

    # === Hidden states (Transformer models with output_hidden_states=True) ===
    output_hidden_states: Optional[Float[Tensor, "bsz n_layers seq_len d_model"]] = None
    """Hidden states from all layers.

    Note: Typically requires stacking tuple outputs from HuggingFace models:
        `torch.stack(outputs.hidden_states, dim=1)`
    """

    # === Attention weights (Transformer models with output_attentions=True) ===
    output_attentions: Optional[Float[Tensor, "bsz n_layers n_heads seq_len seq_len"]] = None
    """Attention weights from all layers.

    Note: Typically requires stacking tuple outputs from HuggingFace models:
        `torch.stack(outputs.attentions, dim=1)`
    """

    # === Generated responses (Language models with generation) ===
    generated_response_ids: Optional[List[Int[Tensor, "response_len"]]] = None
    """Generated token IDs from language model generation. 
    Response lengths may vary across samples.
    """

    generated_response_strs: Optional[List[str]] = None
    """Generated text strings from language model generation.
    """

    generated_response_logits: Optional[List[Float[Tensor, "response_len vocab_size"]]] = None
    """Logits for generated tokens from language model generation. Notably, this differs from `response_logits` which take the logits w.r.t. a prefilled (mostly target) response.
    Response lengths may vary across samples.
    """

    # === Full template ===
    full_template_ids: Optional[Int[Tensor, "bsz full_seq_len"]] = None
    """Full template token IDs (prompt + generation; includes optional padding).
    """

    full_template_strs: Optional[List[str]] = None
    """Full template strings (prompt + generation).
    """

    # === Validators ===

    # @field_validator('output_embeddings', mode='before')
    # @classmethod
    # def validate_embeddings_shape(cls, v):
    #     """Validate that embeddings are 2D tensors."""
    #     if v is not None:
    #         if not isinstance(v, Tensor):
    #             raise TypeError(f"output_embeddings must be a Tensor, got {type(v)}")
    #         if v.ndim != 2:
    #             raise ValueError(
    #                 f"output_embeddings must be 2D (bsz, d_model), got shape {v.shape}"
    #             )
    #     return v

    # @field_validator('output_logits', mode='before')
    # @classmethod
    # def validate_logits_shape(cls, v):
    #     """Validate that logits are 3D tensors."""
    #     if v is not None:
    #         if not isinstance(v, Tensor):
    #             raise TypeError(f"output_logits must be a Tensor, got {type(v)}")
    #         if v.ndim != 3:
    #             raise ValueError(
    #                 f"output_logits must be 3D (bsz, seq_len, vocab_size), got shape {v.shape}"
    #             )
    #     return v

    # @field_validator('response_logits', mode='before')
    # @classmethod
    # def validate_response_logits_shape(cls, v):
    #     """Validate that response logits are 3D tensors."""
    #     if v is not None:
    #         if not isinstance(v, Tensor):
    #             raise TypeError(f"response_logits must be a Tensor, got {type(v)}")
    #         if v.ndim != 3:
    #             raise ValueError(
    #                 f"response_logits must be 3D (bsz, response_seq_len, vocab_size), "
    #                 f"got shape {v.shape}"
    #             )
    #     return v

    # @field_validator('output_hidden_states', mode='before')
    # @classmethod
    # def validate_hidden_states_shape(cls, v):
    #     """Validate that hidden states are 4D tensors."""
    #     if v is not None:
    #         if not isinstance(v, Tensor):
    #             raise TypeError(f"output_hidden_states must be a Tensor, got {type(v)}")
    #         if v.ndim != 4:
    #             raise ValueError(
    #                 f"output_hidden_states must be 4D (bsz, n_layers, seq_len, d_model), "
    #                 f"got shape {v.shape}"
    #             )
    #     return v

    # @field_validator('output_attentions', mode='before')
    # @classmethod
    # def validate_attentions_shape(cls, v):
    #     """Validate that attentions are 5D tensors."""
    #     if v is not None:
    #         if not isinstance(v, Tensor):
    #             raise TypeError(f"output_attentions must be a Tensor, got {type(v)}")
    #         if v.ndim != 5:
    #             raise ValueError(
    #                 f"output_attentions must be 5D (bsz, n_layers, n_heads, seq_len, seq_len), "
    #                 f"got shape {v.shape}"
    #             )
    #     return v

    # @field_validator('full_template_ids', mode='before')
    # @classmethod
    # def validate_template_ids_shape(cls, v):
    #     """Validate that template IDs are 2D tensors."""
    #     if v is not None:
    #         if not isinstance(v, Tensor):
    #             raise TypeError(f"full_template_ids must be a Tensor, got {type(v)}")
    #         if v.ndim != 2:
    #             raise ValueError(
    #                 f"full_template_ids must be 2D (bsz, full_seq_len), got shape {v.shape}"
    #             )
    #     return v

    # @field_validator('generated_response_ids', mode='before')
    # @classmethod
    # def validate_response_ids(cls, v):
    #     """Validate that response IDs are a list of 1D tensors."""
    #     if v is not None:
    #         if not isinstance(v, list):
    #             raise TypeError(f"generated_response_ids must be a list, got {type(v)}")
    #         for i, tensor in enumerate(v):
    #             if not isinstance(tensor, Tensor):
    #                 raise TypeError(
    #                     f"generated_response_ids[{i}] must be a Tensor, got {type(tensor)}"
    #                 )
    #             if tensor.ndim != 1:
    #                 raise ValueError(
    #                     f"generated_response_ids[{i}] must be 1D, got shape {tensor.shape}"
    #                 )
    #     return v

    # @field_validator('generated_response_logits', mode='before')
    # @classmethod
    # def validate_response_logits(cls, v):
    #     """Validate that response logits are a list of 2D tensors."""
    #     if v is not None:
    #         if not isinstance(v, list):
    #             raise TypeError(f"generated_response_logits must be a list, got {type(v)}")
    #         for i, tensor in enumerate(v):
    #             if not isinstance(tensor, Tensor):
    #                 raise TypeError(
    #                     f"generated_response_logits[{i}] must be a Tensor, got {type(tensor)}"
    #                 )
    #             if tensor.ndim != 2:
    #                 raise ValueError(
    #                     f"generated_response_logits[{i}] must be 2D (response_len, vocab_size), "
    #                     f"got shape {tensor.shape}"
    #                 )
    #     return v

    # @field_validator('generated_response_strs', mode='before')
    # @classmethod
    # def validate_response_strs(cls, v):
    #     """Validate that response strings are a list of strings."""
    #     if v is not None:
    #         if not isinstance(v, list):
    #             raise TypeError(f"generated_response_strs must be a list, got {type(v)}")
    #         for i, s in enumerate(v):
    #             if not isinstance(s, str):
    #                 raise TypeError(
    #                     f"generated_response_strs[{i}] must be a string, got {type(s)}"
    #                 )
    #     return v

    # @field_validator('full_template_strs', mode='before')
    # @classmethod
    # def validate_template_strs(cls, v):
    #     """Validate that template strings are a list of strings."""
    #     if v is not None:
    #         if not isinstance(v, list):
    #             raise TypeError(f"full_template_strs must be a list, got {type(v)}")
    #         for i, s in enumerate(v):
    #             if not isinstance(s, str):
    #                 raise TypeError(
    #                     f"full_template_strs[{i}] must be a string, got {type(s)}"
    #                 )
    #     return v



# ======================= Target Types and Utils =======================


class TargetsDictPlus(dict):
    pass

    #----------------------------------------------------------------------------#
    # ## TODO remove
    #     ## Utils for obtaining and manipulating different views of the TargetsDict: ##
    # @staticmethod
    # def get_expanded_with_candidates(targets: TargetsDict | "TargetsDictPlus", n_candidates: int) -> BatchedTargetsDict:
    #     """
    #     Repeats each target entry `n_repeats` times along the message dimension.
    #     Useful when expanding targets to match multiple candidate triggers per message.
    #     Returns result in a new dict (BatchedTargetsDict).
    #     """
    #     targets = targets.copy()

    #     for k, v in targets.items():
    #         if isinstance(v, torch.Tensor):
    #             # (n_messages, ...) -> (n_messages, n_candidates, ...)
    #             targets[k] = v.unsqueeze(1).expand(v.shape[0], n_candidates, *v.shape[1:])
    #         elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
    #             # for each element: (...,) -> (n_candidates, ...)
    #             targets[k] = [t.unsqueeze(0).expand(n_candidates, *t.shape) for t in v]
    #         elif isinstance(v, list):
    #             targets[k] = [[elem] * n_candidates for elem in v]
    #         else:
    #             raise ValueError(f"Unsupported target type for key {k}: {type(targets[k])}")

    #     return targets

    # @staticmethod
    # def get_message_from_batched_targets(
    #     targets: BatchedTargetsDict,
    #     chosen_message_idx: int,
    # ) -> MessageBatchedTargetsDict:
    #     """
    #     Selects the targets for a specific message index from the BatchedTargetsDict.
    #     Returns result in a new dict (MessageBatchedTargetsDict).
    #     """
    #     targets = {k: v[chosen_message_idx] for k, v in targets.items()}
    #     return targets

    @staticmethod
    def select_message(
        targets: TargetsDict | "TargetsDictPlus",
        chosen_message_idx: int,
    ) -> MessageTargetsDict:
        """
        Selects the targets for a specific message index from the TargetsDict.
        Returns result in a new dict (MessageTargetsDict).
        """
        targets = {k: v[chosen_message_idx] for k, v in targets.items()}
        return targets

    #----------------------------------------------------------------------------
