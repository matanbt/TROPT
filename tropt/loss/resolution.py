"""Unified loss resolution and computation.

This module provides a centralized function for computing losses from model
inputs and outputs. It eliminates the need for each model to implement
loss-specific logic, allowing models to simply provide data while this module
handles loss resolution.

The key function `compute_loss_from_model_data()` accepts standardized
ModelOutput and ModelInput containers and dispatches to the appropriate
loss-specific computation logic.
"""

from typing import Union

import torch
from jaxtyping import Float
from torch import Tensor

from tropt.loss.base import (
    AttentionBasedLoss,
    BaseLoss,
    CombinedLoss,
    EmbeddingBasedLoss,
    HiddenStateBased,
    LogitBasedLoss,
    SteeringActivationLoss,
    TextBasedLoss,
    TriggerLogitBasedLoss,
)
from tropt.models.inputs import ModelInput, SliceKey
from tropt.models.outputs import ModelOutput


class LossResolutionError(Exception):
    """Raised when required data is missing for loss computation.

    This exception indicates that a loss function requires specific model
    outputs or inputs that were not provided. The error message should
    clearly state what data is missing and which loss type requires it.

    Examples:
        >>> raise LossResolutionError(
        ...     "LogitBasedLoss requires output_logits, but model did not provide it."
        ... )
    """
    pass


def compute_loss_from_model_data(
    model_output: ModelOutput,
    model_input: ModelInput,
    loss_func: BaseLoss,
) -> Float[Tensor, "bsz"]:
    """Universal loss computation function that resolves loss from model I/O.

    This function centralizes all loss resolution logic, removing duplication
    across model implementations. Models only need to populate ModelOutput with
    available data; this function handles extracting and providing the correct
    data to each loss.

    The function performs type-based dispatch to loss-specific helper functions,
    each of which validates required data, extracts it from the model I/O
    containers, and calls the loss function with the correct arguments.

    Args:
        model_output: Standardized model output containing available data
            (logits, embeddings, hidden_states, attentions, etc.)
        model_input: Standardized model input containing triggers, slices,
            and targets
        loss_func: The loss function to compute

    Returns:
        Loss tensor of shape (bsz,) containing per-sample losses

    Raises:
        LossResolutionError: If required data is missing for the specified loss
        NotImplementedError: If the loss type is not yet supported

    Examples:
        >>> # Encoder model with embedding-based loss
        >>> output = ModelOutput(output_embeddings=torch.randn(4, 768))
        >>> input_data = ModelInput(targets={"target_vectors": target_vecs})
        >>> loss = compute_loss_from_model_data(output, input_data, SimilarityLoss())
        >>> loss.shape
        torch.Size([4])

        >>> # Language model with logit-based loss
        >>> output = ModelOutput(output_logits=torch.randn(2, 50, 32000))
        >>> input_data = ModelInput(
        ...     input_slices=[{SliceKey.APPENDED: slice(40, 50)}] * 2,
        ...     targets={"target_outputs_toks": target_ids}
        ... )
        >>> loss = compute_loss_from_model_data(output, input_data, PrefillCELoss())

    Design Notes:
        - Type-based dispatch using isinstance() checks
        - Each loss category has a dedicated helper function
        - CombinedLoss is handled recursively
        - Clear error messages indicate missing data
    """

    if isinstance(loss_func, LogitBasedLoss):
        return _compute_logit_based_loss(model_output, model_input, loss_func)

    elif isinstance(loss_func, TriggerLogitBasedLoss):
        return _compute_trigger_logit_based_loss(model_output, model_input, loss_func)

    elif isinstance(loss_func, AttentionBasedLoss):
        return _compute_attention_based_loss(model_output, model_input, loss_func)

    elif isinstance(loss_func, SteeringActivationLoss):
        return _compute_steering_loss(model_output, model_input, loss_func)

    elif isinstance(loss_func, EmbeddingBasedLoss):
        return _compute_embedding_based_loss(model_output, model_input, loss_func)

    elif isinstance(loss_func, TextBasedLoss):
        return _compute_text_based_loss(model_output, model_input, loss_func)

    elif isinstance(loss_func, CombinedLoss):
        return _compute_combined_loss(model_output, model_input, loss_func)

    else:
        raise NotImplementedError(
            f"Loss type {type(loss_func).__name__} is not yet supported in unified loss resolution. "
            f"Supported types: LogitBasedLoss, TriggerLogitBasedLoss, AttentionBasedLoss, "
            f"SteeringActivationLoss, EmbeddingBasedLoss, TextBasedLoss, CombinedLoss."
        )


# ============================================================================
# Helper functions for each loss type
# ============================================================================


def _compute_logit_based_loss(
    model_output: ModelOutput,
    model_input: ModelInput,
    loss_func: LogitBasedLoss,
) -> Float[Tensor, "bsz"]:
    """Compute losses that operate on target output logits.

    These losses compute objectives over the logits corresponding to target
    output tokens (e.g., cross-entropy, mellowmax, Carlini-Wagner).

    Args:
        model_output: Must contain output_logits
        model_input: Must contain input_slices and targets with target_outputs_toks
        loss_func: The logit-based loss to compute

    Returns:
        Per-sample losses of shape (bsz,)

    Raises:
        LossResolutionError: If required data is missing or malformed
    """

    # Validate required data
    if model_output.output_logits is None:
        raise LossResolutionError(
            f"{type(loss_func).__name__} requires output_logits, but model did not provide it. "
            f"Ensure the model forward pass returns logits."
        )

    logits = model_output.output_logits  # (bsz, seq_len, vocab_size)
    targets = model_input.targets
    slices = model_input.input_slices

    if targets is None or "target_outputs_toks" not in targets:
        raise LossResolutionError(
            f"{type(loss_func).__name__} requires 'target_outputs_toks' in targets dict. "
            f"Available keys: {list(targets.keys()) if targets else 'None'}"
        )

    if slices is None:
        raise LossResolutionError(
            f"{type(loss_func).__name__} requires input_slices to locate target positions in the sequence."
        )

    # Extract target IDs and slices
    response_target_ids = targets["target_outputs_toks"]  # (bsz, target_seq_len)
    response_slcs = [s[SliceKey.APPENDED] for s in slices]  # bsz of slice objects

    # Validate alignment across batch
    first_slc = response_slcs[0]
    are_slcs_aligned = all(
        s.start == first_slc.start and s.stop == first_slc.stop
        for s in response_slcs
    )

    if not are_slcs_aligned:
        raise LossResolutionError(
            f"Response slices are not aligned across batch. Variable-length target sequences "
            f"are not supported yet. Slices: {response_slcs}"
        )

    # Validate response_target_ids shape
    if not isinstance(response_target_ids, torch.Tensor):
        raise LossResolutionError(
            f"response_target_ids must be a tensor, got {type(response_target_ids)}"
        )

    if response_target_ids.dim() != 2 or response_target_ids.shape[0] != logits.shape[0]:
        raise LossResolutionError(
            f"response_target_ids must have shape (bsz, target_seq_len) matching batch size. "
            f"Expected batch size {logits.shape[0]}, got shape {response_target_ids.shape}"
        )

    # Extract response logits (with -1 offset for logit shift)
    start_idx = first_slc.start - 1
    end_idx = first_slc.stop - 1
    response_logits = logits[:, start_idx:end_idx, :]

    # Validate length match
    if response_logits.shape[1] != response_target_ids.shape[1]:
        raise LossResolutionError(
            f"Length mismatch: response_logits has {response_logits.shape[1]} positions "
            f"but target_ids has {response_target_ids.shape[1]} tokens. "
            f"Slice: {first_slc}, start_idx={start_idx}, end_idx={end_idx}"
        )

    # Compute loss
    return loss_func(response_logits, response_target_ids)


def _compute_trigger_logit_based_loss(
    model_output: ModelOutput,
    model_input: ModelInput,
    loss_func: TriggerLogitBasedLoss,
) -> Float[Tensor, "bsz"]:
    """Compute losses that operate on trigger token logits.

    These losses compute objectives over the logits corresponding to trigger
    tokens (used for trigger-specific regularization or objectives).

    Args:
        model_output: Must contain output_logits
        model_input: Must contain input_trigger_ids and input_slices
        loss_func: The trigger logit-based loss to compute

    Returns:
        Per-sample losses of shape (bsz,)

    Raises:
        LossResolutionError: If required data is missing or malformed
    """

    if model_output.output_logits is None:
        raise LossResolutionError(
            f"{type(loss_func).__name__} requires output_logits, but model did not provide it."
        )

    if model_input.input_trigger_ids is None:
        raise LossResolutionError(
            f"{type(loss_func).__name__} requires input_trigger_ids."
        )

    if model_input.input_slices is None:
        raise LossResolutionError(
            f"{type(loss_func).__name__} requires input_slices to locate trigger positions."
        )

    logits = model_output.output_logits
    trigger_ids = model_input.input_trigger_ids
    trigger_slcs = [s[SliceKey.TRIGGER] for s in model_input.input_slices]

    # Validate alignment
    first_slc = trigger_slcs[0]
    are_slcs_aligned = all(
        s.start == first_slc.start and s.stop == first_slc.stop
        for s in trigger_slcs
    )

    if not are_slcs_aligned:
        raise LossResolutionError(
            f"Trigger slices are not aligned across batch. Slices: {trigger_slcs}"
        )

    if first_slc.start < 1:
        raise LossResolutionError(
            f"Trigger slices should start at position >= 1 for feasible logits (got start={first_slc.start}). "
            f"This may be caused by prefix caching; try disabling it."
        )

    # Extract trigger logits (with -1 offset for logit shift)
    start_idx = first_slc.start - 1
    end_idx = first_slc.stop - 1
    trigger_logits = logits[:, start_idx:end_idx, :]

    # Validate shapes
    if trigger_logits.shape[1] != trigger_ids.shape[1]:
        raise LossResolutionError(
            f"Trigger length mismatch: logits has {trigger_logits.shape[1]} positions "
            f"but trigger_ids has {trigger_ids.shape[1]} tokens."
        )

    return loss_func(trigger_logits, trigger_ids)


def _compute_attention_based_loss(
    model_output: ModelOutput,
    model_input: ModelInput,
    loss_func: AttentionBasedLoss,
) -> Float[Tensor, "bsz"]:
    """Compute losses based on attention weights.

    These losses analyze or optimize attention patterns (e.g., maximizing
    attention from trigger to specific regions).

    Args:
        model_output: Must contain output_attentions
        model_input: Should contain input_slices (optional for some losses)
        loss_func: The attention-based loss to compute

    Returns:
        Per-sample losses of shape (bsz,)

    Raises:
        LossResolutionError: If output_attentions is missing
    """

    if model_output.output_attentions is None:
        raise LossResolutionError(
            f"{type(loss_func).__name__} requires output_attentions, but model did not provide it. "
            f"Ensure model forward pass has output_attentions=True."
        )

    attentions = model_output.output_attentions
    # Provide slices if available, otherwise create empty dicts
    slices = model_input.input_slices or [{} for _ in range(attentions.shape[0])]

    return loss_func(attentions, slices=slices)


def _compute_steering_loss(
    model_output: ModelOutput,
    model_input: ModelInput,
    loss_func: SteeringActivationLoss,
) -> Float[Tensor, "bsz"]:
    """Compute steering/activation-based losses.

    These losses optimize activations to align with or avoid specific
    directions in representation space (e.g., refusal suppression via
    representation engineering).

    Args:
        model_output: Must contain output_hidden_states
        model_input: Must contain targets with target_directions
        loss_func: The steering loss to compute

    Returns:
        Per-sample losses of shape (bsz,)

    Raises:
        LossResolutionError: If required data is missing
    """

    if model_output.output_hidden_states is None:
        raise LossResolutionError(
            f"{type(loss_func).__name__} requires output_hidden_states, but model did not provide it. "
            f"Ensure model forward pass has output_hidden_states=True."
        )

    if model_input.targets is None or "target_directions" not in model_input.targets:
        raise LossResolutionError(
            f"{type(loss_func).__name__} requires 'target_directions' in targets dict. "
            f"Available keys: {list(model_input.targets.keys()) if model_input.targets else 'None'}"
        )

    hidden_states = model_output.output_hidden_states
    target_directions = model_input.targets["target_directions"]
    # Provide slices if available
    slices = model_input.input_slices or [{} for _ in range(hidden_states.shape[0])]

    return loss_func(
        hidden_states,
        target_directions=target_directions,
        slices=slices
    )


def _compute_embedding_based_loss(
    model_output: ModelOutput,
    model_input: ModelInput,
    loss_func: EmbeddingBasedLoss,
) -> Float[Tensor, "bsz"]:
    """Compute losses based on embeddings.

    These losses optimize similarity or distance between output embeddings
    and target vectors (used primarily with encoder models).

    Args:
        model_output: Must contain output_embeddings
        model_input: Must contain targets with the appropriate target key
        loss_func: The embedding-based loss to compute

    Returns:
        Per-sample losses of shape (bsz,)

    Raises:
        LossResolutionError: If required data is missing
    """

    if model_output.output_embeddings is None:
        raise LossResolutionError(
            f"{type(loss_func).__name__} requires output_embeddings, but model did not provide it."
        )

    if model_input.targets is None or loss_func.TARGET_KEY not in model_input.targets:
        raise LossResolutionError(
            f"{type(loss_func).__name__} requires '{loss_func.TARGET_KEY}' in targets dict. "
            f"Available keys: {list(model_input.targets.keys()) if model_input.targets else 'None'}"
        )

    embeddings = model_output.output_embeddings
    target_vectors = model_input.targets[loss_func.TARGET_KEY]

    return loss_func(embeddings, target_vectors)


def _compute_text_based_loss(
    model_output: ModelOutput,
    model_input: ModelInput,
    loss_func: TextBasedLoss,
) -> Float[Tensor, "bsz"]:
    """Compute losses based on generated text.

    These losses use text-level analysis (e.g., LM-as-judge, keyword matching)
    to compute objectives over generated strings.

    Args:
        model_output: Must contain generated_response_strs
        model_input: Not used (text losses operate only on outputs)
        loss_func: The text-based loss to compute

    Returns:
        Per-sample losses of shape (bsz,)

    Raises:
        LossResolutionError: If generated_response_strs is missing
    """

    if model_output.generated_response_strs is None:
        raise LossResolutionError(
            f"{type(loss_func).__name__} requires generated_response_strs, but model did not provide it."
        )

    return loss_func(model_output.generated_response_strs)


def _compute_combined_loss(
    model_output: ModelOutput,
    model_input: ModelInput,
    loss_func: CombinedLoss,
) -> Float[Tensor, "bsz"]:
    """Compute combined (weighted) losses recursively.

    Handles CombinedLoss which aggregates multiple child losses with weights.
    Each child loss is computed recursively using the main resolution function.

    Args:
        model_output: Model outputs (passed to all child losses)
        model_input: Model inputs (passed to all child losses)
        loss_func: The combined loss with child loss functions

    Returns:
        Combined loss of shape (bsz,)

    Raises:
        LossResolutionError: If any child loss fails (propagated from recursive calls)
    """

    child_losses = []

    for nested_loss_func in loss_func.loss_funcs:
        # Recursive call for each child loss
        child_loss = compute_loss_from_model_data(
            model_output,
            model_input,
            nested_loss_func
        )
        child_losses.append(child_loss)

    # Stack: (n_losses, bsz)
    losses = torch.stack(child_losses, dim=0)

    # Apply combination (e.g., weighted sum): (bsz,)
    return loss_func(losses)
