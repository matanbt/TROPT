"""Standardized output containers for model forward passes.

This module defines dataclass containers that provide a uniform interface for
accessing model outputs across different model types and implementations.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor


@dataclass
class ModelOutput:
    """Standardized output container for all model types in TROPT.

    Models populate only the fields they can provide. Loss resolution logic
    uses this standardized interface to extract required data without knowing
    the specific model type.

    Field Naming Convention:
        All fields are prefixed with `output_` for clarity and to distinguish
        them from input fields in ModelInput.

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

    # === Embedding outputs (Encoder models) ===
    output_embeddings: Optional[Float[Tensor, "bsz d_model"]] = None
    """Sentence/sequence embeddings from encoder models.

    Typically produced by models like SentenceTransformers, OpenAI embeddings,
    or Gemini embeddings. Shape: (batch_size, embedding_dimension).
    """

    # === Logits (Language models) ===
    output_logits: Optional[Float[Tensor, "bsz seq_len vocab_size"]] = None
    """Full sequence logits from language models.

    Logits for each position in the input sequence. Used by logit-based losses
    to compute cross-entropy, mellowmax, or other objectives. Shape: (batch_size,
    sequence_length, vocabulary_size).

    Note: For generation-only models without logit access, this will be None.
    """

    # === Hidden states (Transformer models with output_hidden_states=True) ===
    output_hidden_states: Optional[Float[Tensor, "bsz n_layers seq_len d_model"]] = None
    """Hidden states from all layers.

    Requires `output_hidden_states=True` in the model forward call. Used by
    activation-based losses like SteeringActivationLoss. Shape: (batch_size,
    num_layers, sequence_length, embedding_dimension).

    Note: Typically requires stacking tuple outputs from HuggingFace models:
        `torch.stack(outputs.hidden_states, dim=1)`
    """

    # === Attention weights (Transformer models with output_attentions=True) ===
    output_attentions: Optional[Float[Tensor, "bsz n_layers n_heads seq_len seq_len"]] = None
    """Attention weights from all layers.

    Requires `output_attentions=True` in the model forward call. Used by
    attention-based losses. Shape: (batch_size, num_layers, num_heads,
    sequence_length, sequence_length) where the last two dimensions represent
    (destination_position, source_position).

    Note: Typically requires stacking tuple outputs from HuggingFace models:
        `torch.stack(outputs.attentions, dim=1)`
    """

    # === Generated responses (Language models with generation) ===
    generated_response_ids: Optional[List[Int[Tensor, "response_len"]]] = None
    """Generated token IDs from language model generation.

    List of tensors, one per sample. Each tensor contains the generated token
    IDs for that sample. Response lengths may vary across samples.

    Example: [tensor([123, 456, 789]), tensor([111, 222])]
    """

    generated_response_strs: Optional[List[str]] = None
    """Generated text strings from language model generation.

    List of strings, one per sample. The decoded text corresponding to
    generated_response_ids.

    Example: ["Sure, here is a response.", "Another generated text."]
    """

    generated_response_logits: Optional[List[Float[Tensor, "response_len vocab_size"]]] = None
    """Logits for generated tokens from language model generation.

    List of tensors, one per sample. Each tensor contains logits for the
    generated tokens in that sample. Used for analyzing generation probabilities.
    Response lengths may vary across samples.

    Note: Only available for white-box models that expose logits during generation.
    """

    # === Full template (for reference/debugging) ===
    full_template_ids: Optional[Int[Tensor, "bsz full_seq_len"]] = None
    """Full template token IDs (prompt + generation).

    Complete sequence of token IDs including both the input prompt and the
    generated response. Shape: (batch_size, full_sequence_length).

    Useful for debugging and verifying the complete prompt-response sequence.
    """

    full_template_strs: Optional[List[str]] = None
    """Full template strings (prompt + generation).

    Complete text strings including both the input prompt and the generated
    response. One string per sample.

    Useful for debugging and logging the complete prompt-response sequence.
    """
