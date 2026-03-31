from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

import pydantic
import torch
from jaxtyping import Float, Int
from torch import Tensor

# =========== Common constants and utilities for TTOP ==========
# Defines a placeholder string for optimized triggers
OPTIMIZED_TRIGGER_PLACEHOLDER: str = "{{OPTIMIZED_TRIGGER}}"

# Default initial trigger
DEFAULT_INIT_TRIGGER = ("! " * 20).strip()



# ======================= Common input types =======================
TextTemplates = Annotated[
    List[str],
    pydantic.Field(min_length=1),
    "n_templates"
]
"""List of text templates, one per optimization target. Each must contain the trigger placeholder (`{{OPTIMIZED_TRIGGER}}`).
Length: n_templates.
"""
TokenTrigger = Float[Tensor, "1 trigger_seq_len"]
TextTrigger = str
TokenTriggerCandidates = Float[Tensor, "n_candidates trigger_seq_len"]

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

class MessageTargets(pydantic.BaseModel):
    """Targets for a single selected message."""
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

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


class Targets(pydantic.BaseModel):
    """Targets for all templates. Each field has an initial n_templates dimension.

    Typically only one or two of these fields need to be provided depending
    on the loss function being used.
    For example, a standard LM jailbreak only needs `target_response_strs` to provide the target outputs.
    """
    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    target_response_strs: Optional[Annotated[List[str], "n_templates"]] = None
    """Raw text target outputs, one per template.

    List is of length n_templates.
    - Used by: Language models for target matching. Will be tokenized
               internally to produce `target_response_toks` if not provided directly.
    - If used with prefill-based losses, it will automatically run model computations with a the prefilled target response
    """

    target_response_toks: Optional[Int[Tensor, "n_templates target_seq_len"] | Annotated[List[Int[Tensor, "target_seq_len"]], "n_templates"]] = None
    """Tokenized target outputs, one per template.

    Shape: (n_templates, target_seq_len) OR List of length n_templates,
    each of (potentially different) shape (target_seq_len,)
    Used by: Language models for computing cross-entropy loss.
    """

    target_vectors: Optional[Float[Tensor, "n_templates d_model"]] = None
    """Target embedding vectors, one per template.

    Shape: (n_templates, d_model)
    Used by: Encoder models for similarity-based losses.
    """

    target_directions: Optional[Float[Tensor, "n_templates d_model"]] = None
    """Target directions in activation space, one per template.

    Shape: (n_templates, d_model)
    Used by: Steering losses (e.g., refusal suppression).
    Note: if you need per-layer directions, store as (n_templates, n_layers, d_model)
    and update this annotation accordingly.
    """

    @property
    def n_templates(self) -> int:
        for field_name in self.model_fields_set:
            val = getattr(self, field_name)
            if val is not None:
                return len(val)
        return 0

    @pydantic.model_validator(mode="after")
    def check_field_lengths(self) -> "Targets":
        lengths = {len(v) for k, v in self if v is not None}
        if len(lengths) > 1:
            raise ValueError("All target fields must have the same length")
        return self

    def select_message(self, idx: int) -> "MessageTargets":
        return MessageTargets(
            **{k: v[idx] for k, v in self if v is not None}
        )

    def to_device(self, device: torch.device | str) -> "Targets":
        updates = {}
        for k, v in self:
            if isinstance(v, Tensor):
                updates[k] = v.to(device)
            if isinstance(v, list) and isinstance(v[0], Tensor):
                updates[k] = [t.to(device) for t in v]
        return self.model_copy(update=updates)



# ======================= Model Input Wrapper =======================

class ModelInput(pydantic.BaseModel):
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
        ...     message_targets=MessageTargets(target_response_toks=target_ids)
        ... )

        >>> # Text-level input
        >>> text_input = ModelInput(
        ...     input_texts=["Text with trigger 1", "Text with trigger 2"],
        ...     message_targets=MessageTargets(target_response_strs="Response 1")
        ... )
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    # === Text-level inputs (TextInputManager) ===
    input_texts: Optional[Annotated[List[str], "bsz"]] = None
    """List of complete text strings with triggers inserted, of length batch_size.
    """

    input_trigger_strs: Optional[Annotated[List[str], "bsz"]] = None
    """List of trigger strings used in the inputs, of length batch_size.
    """

    # === Token-level inputs (TokenInputManager) ===
    input_ids: Optional[Int[Tensor, "bsz seq_len"]] = None
    """Token IDs of the full input sequence (prompt + trigger), plus optionally target tokens.
    """


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
    message_targets: Optional[MessageTargets] = None
    """Target data required by loss functions.

    A `MessageTargets` instance containing the target data for a single message.
    The specific fields used depend on the loss function (e.g., `target_response_strs`
    for text-based losses, `target_directions` for steering losses).
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary, excluding None values."""
        return self.model_dump(exclude_none=True)

    # TODO optional additional validators to check shapes of inputs?


# ======================= Model Output Wrapper =======================

class ModelOutput(pydantic.BaseModel):
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
        ...     full_logits=torch.randn(2, 50, 32000),
        ...     generated_response_strs=["Response 1", "Response 2"]
        ... )

        >>> # Full output with hidden states and attentions
        >>> full_output = ModelOutput(
        ...     full_logits=logits,
        ...     full_hidden_states=hidden_states,
        ...     full_attentions=attentions,
        ...     generated_response_strs=responses
        ... )
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    # === Embedding outputs (Encoder models) ===
    output_embeddings: Optional[Float[Tensor, "bsz d_model"]] = None
    """Pooled output embeddings from encoder models.
    Shape: (batch_size, d_model)
    """

    # === Logits (Language models) ===
    # LM legend: all = input (incl trigger) + prefilled response + generated response tokens
    full_logits: Optional[Float[Tensor, "bsz seq_len vocab_size"]] = None
    """Full sequence logits from language models; including both inputs and outputs (prefilled and generated).
    """

    prefill_response_logits: Optional[Float[Tensor, "bsz response_seq_len vocab_size"]] = None
    """Logits corresponding to the *prefilled* response portion of the sequence.
    """

    # === Hidden states (Transformer models with output_hidden_states=True) ===
    full_hidden_states: Optional[Float[Tensor, "bsz n_layers seq_len d_model"]] = None
    """Hidden states from all layers.

    Note: Typically requires stacking tuple outputs from HuggingFace models:
        `torch.stack(outputs.hidden_states, dim=1)`
    """

    # === Attention weights (Transformer models with output_attentions=True) ===
    full_attentions: Optional[Float[Tensor, "bsz n_layers n_heads seq_len seq_len"]] = None
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

    generated_response_logits: Optional[List[Float[Tensor, "response_len vocab_size"]] | Float[Tensor, "bsz response_len vocab_size"]] = None
    """Logits for generated tokens from language model generation.
    Notably, this differs from `response_logits` which take the logits w.r.t. a prefilled (mostly target) response. In particular, this excludes any prefilled tokens.
    Response lengths may vary across samples.
    """

    # === First-token logprobs (API and HF models with generation) ===
    response_first_token_logprobs: Optional[List[Dict[str, float]]] = None
    """Sparse log-probabilities for the first generated token.

    List of length bsz, where each element is a dict mapping token strings to
    their log-probability.  For API models this is typically the top-k returned
    by the provider (e.g. top-20 from OpenAI); for HF models it can cover the
    full vocabulary.
    """

    # === Classification outputs (Classifier models) ===
    output_class_logits: Optional[Float[Tensor, "bsz n_classes"]] = None
    """Classification logits from classifier models (pre-softmax).
    Shape: (batch_size, num_classes)
    """

    # === Full template ===
    full_ids: Optional[Int[Tensor, "bsz full_seq_len"]] = None
    """Full template token IDs (prompt + generation; includes optional padding).
    """

    full_strs: Optional[List[str]] = None
    """Full template strings (prompt + generation).
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary, excluding None values."""
        return self.model_dump(exclude_none=True)

    # TODO add validators to check shapes?


    #----------------------------------------------------------------------------
