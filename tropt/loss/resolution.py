"""Unified loss resolution and computation via introspection.
"""

import inspect
import logging
from typing import Any, Dict

import torch
from jaxtyping import Float
from torch import Tensor

from tropt.common import ModelInput, ModelOutput
from tropt.loss.base import BaseLoss, CombinedLoss

logger = logging.getLogger(__name__)

class LossResolutionError(Exception):
    """Raised when required data is missing for loss computation.

    This exception indicates that a loss function requires specific model
    outputs or inputs that were not provided. The error message should
    clearly state what parameter is missing and where it should come from.

    Examples:
        >>> raise LossResolutionError(
        ...     "Loss function requires parameter 'full_logits' but it was not "
        ...     "found in model_output or model_input."
        ... )
    """
    pass

def resolve_and_compute_loss(
    model_output: ModelOutput,
    model_input: ModelInput,
    loss_func: BaseLoss,
) -> Float[Tensor, "bsz"]:
    """Universal loss computation via automatic argument matching.
      
    This function invokes `loss_func` with the model input (which includes the targets and slices), 
    and output; it returns the computed loss tensor (size `bsz`).

    Notes:
        - Since loss functions in TROPT declare their required arguments following the naming 
        convention of ModelOutput and ModelInput fields, this function can automatically
        resolve which data to provide to the loss function by inspecting its __call__
        signature.

        - In case insufficient arguments are available (e.g., because the model does not provide the required access),
        a LossResolutionError is raised with details on what is missing.

        - **Parameter Naming Convention:**
            Loss functions should name their parameters exactly as they appear in
            ModelOutput and ModelInput:
            - full_logits: From ModelOutput
            - output_embeddings: From ModelOutput
            - full_attentions: From ModelOutput
            - input_trigger_ids: From ModelInput
            - input_slices: From ModelInput
            - message_targets: From ModelInput
            - etc.

    Args:
        model_output: Standardized model output containing available data
        model_input: Standardized model input containing triggers, slices, message targets
        loss_func: The loss function to compute (must have proper __call__ signature)

    Returns:
        Loss tensor of shape (bsz,) containing per-sample losses

    Raises:
        LossResolutionError: If required parameter is not found in model data
        TypeError: If loss function signature is invalid

    Examples:
        >>> from tropt.common import MessageTargets
        >>> # Encoder model with SimilarityLoss(output_embeddings, target_embeddings)
        >>> output = ModelOutput(output_embeddings=torch.randn(4, 768))
        >>> input_data = ModelInput(message_targets=MessageTargets(target_vectors=target_vecs))
        >>> loss = resolve_and_compute_loss(output, input_data, SimilarityLoss())

        >>> # Language model with PrefillCELoss(prefill_response_logits, message_targets)
        >>> output = ModelOutput(prefill_response_logits=torch.randn(2, 50, 32000))
        >>> input_data = ModelInput(
        ...     input_slices=[{SliceKey.APPENDED: slice(40, 50)}] * 2,
        ...     message_targets=MessageTargets(target_response_toks=target_ids)
        ... )
        >>> loss = resolve_and_compute_loss(output, input_data, PrefillCELoss())

    """

    # Special handling for CombinedLoss (recursive)
    if isinstance(loss_func, CombinedLoss):
        return _compute_combined_loss(model_output, model_input, loss_func)

    # Get the loss function's __call__ signature
    try:
        sig = inspect.signature(loss_func.__call__)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Failed to inspect signature of {type(loss_func).__name__}.__call__: {e}"
        ) from e

    # Build arguments by matching parameter names to model data fields
    kwargs = {}
    for param_name, param in sig.parameters.items():
        # Skip 'self' parameter
        if param_name == 'self':
            continue

        # Try to find this parameter in model_output first
        if hasattr(model_output, param_name):
            value = getattr(model_output, param_name)
            if value is not None:
                kwargs[param_name] = value
                continue
            # If value is None and parameter is optional, that's fine
            if param.default != inspect.Parameter.empty:
                continue
            # If value is None but parameter is required, fall through to error

        # Try to find in model_input second
        if hasattr(model_input, param_name):
            value = getattr(model_input, param_name)
            if value is not None:
                kwargs[param_name] = value
                continue
            # If value is None and parameter is optional, that's fine
            if param.default != inspect.Parameter.empty:
                continue
            # If value is None but parameter is required, fall through to error

        # Special handling for target fields from model_input.message_targets
        if model_input.message_targets is not None and hasattr(model_input.message_targets, param_name):
            value = getattr(model_input.message_targets, param_name)
            if value is not None:
                kwargs[param_name] = value
                continue

        # Parameter not found - raise error if it's required
        if param.default == inspect.Parameter.empty:
            raise LossResolutionError(
                f"Loss function {type(loss_func).__name__} requires parameter "
                f"'{param_name}' but it was not found in model_output or model_input.\n"
                f"It is probably because the model you try to run does not provide this access.\n\n"
                f"Available in model_output: {_get_non_none_fields(model_output)}\n"
                f"Available in model_input: {_get_non_none_fields(model_input)}\n"
                f"Available in message_targets: {_get_non_none_fields(model_input.message_targets)}"
            )

    # Call the loss function with matched arguments
    try:
        return loss_func(**kwargs)
    except Exception as e:
        logger.error(f"Error while calling loss {type(loss_func).__name__} with arguments "
            f"{list(kwargs.keys())}")
        raise e


def _get_non_none_fields(obj: ModelOutput | ModelInput) -> list[str]:
    """Helper to get list of non-None field names from a Pydantic model."""
    return [name for name in type(obj).model_fields if getattr(obj, name) is not None]


def _compute_combined_loss(
    model_output: ModelOutput,
    model_input: ModelInput,
    loss_func: CombinedLoss,
) -> Float[Tensor, "bsz"]:
    """Compute combined loss by recursively calling resolution on each component.

    CombinedLoss is a weighted combination of multiple loss functions. This
    function computes each component loss separately (using recursive calls to
    resolve_and_compute_loss) and combines them with their weights.

    Args:
        model_output: Model outputs available for all component losses
        model_input: Model inputs available for all component losses
        loss_func: CombinedLoss instance with losses and weights

    Returns:
        Weighted combination of component losses
    """
    component_losses = []
    for component_loss in loss_func.loss_funcs:
        # Recursive call - each component loss gets resolved independently
        component_loss_value = resolve_and_compute_loss(
            model_output, model_input, component_loss
        )
        component_losses.append(component_loss_value)

    # Stack into (n_losses, bsz) and delegate weighting to the combine loss function
    stacked = torch.stack(component_losses, dim=0)
    return loss_func(stacked)
