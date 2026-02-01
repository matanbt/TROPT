"""Unified loss resolution and computation via introspection.
"""

import inspect
from typing import Any, Dict

import torch
from jaxtyping import Float
from torch import Tensor

from tropt.common import ModelInput, ModelOutput, SliceKey, TargetKey
from tropt.loss.base import BaseLoss, CombinedLoss


class LossResolutionError(Exception):
    """Raised when required data is missing for loss computation.

    This exception indicates that a loss function requires specific model
    outputs or inputs that were not provided. The error message should
    clearly state what parameter is missing and where it should come from.

    Examples:
        >>> raise LossResolutionError(
        ...     "Loss function requires parameter 'output_logits' but it was not "
        ...     "found in model_output or model_input."
        ... )
    """
    pass


def compute_loss_from_model_data(
    model_output: ModelOutput,
    model_input: ModelInput,
    loss_func: BaseLoss,
) -> Float[Tensor, "bsz"]:
    """Universal loss computation via automatic argument matching.

    Since loss functions in TROPT declare their required arguments following the naming 
    convention of ModelOutput and ModelInput fields, this function can automatically
    resolve which data to provide to the loss function by inspecting its __call__
    signature.

    In case insufficient arguments are avaialble (e.g., becasue the model does not provide the required access),
    a LossResolutionError is raised with details on what is missing.

    **Parameter Naming Convention:**
    Loss functions should name their parameters exactly as they appear in
    ModelOutput and ModelInput:
    - output_logits: From ModelOutput
    - output_embeddings: From ModelOutput
    - output_attentions: From ModelOutput
    - input_trigger_ids: From ModelInput
    - input_slices: From ModelInput
    - targets: From ModelInput
    - etc.

    Args:
        model_output: Standardized model output containing available data
        model_input: Standardized model input containing triggers, slices, targets
        loss_func: The loss function to compute (must have proper __call__ signature)

    Returns:
        Loss tensor of shape (bsz,) containing per-sample losses

    Raises:
        LossResolutionError: If required parameter is not found in model data
        TypeError: If loss function signature is invalid

    Examples:
        >>> # Encoder model with SimilarityLoss(output_embeddings, target_embeddings)
        >>> output = ModelOutput(output_embeddings=torch.randn(4, 768))
        >>> input_data = ModelInput(targets={"target_embeddings": target_vecs})
        >>> loss = compute_loss_from_model_data(output, input_data, SimilarityLoss())

        >>> # Language model with PrefillCELoss(response_logits, input_slices, targets)
        >>> output = ModelOutput(response_logits=torch.randn(2, 50, 32000))
        >>> input_data = ModelInput(
        ...     input_slices=[{SliceKey.APPENDED: slice(40, 50)}] * 2,
        ...     targets={TargetKey.TARGET_RESPONSE_TOKS: target_ids}
        ... )
        >>> loss = compute_loss_from_model_data(output, input_data, PrefillCELoss())

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

        # Special handling for target fields from model_input.targets dict
        if model_input.targets is not None and param_name in model_input.targets:
            kwargs[param_name] = model_input.targets[param_name]
            continue

        # Parameter not found - raise error if it's required
        if param.default == inspect.Parameter.empty:
            raise LossResolutionError(
                f"Loss function {type(loss_func).__name__} requires parameter "
                f"'{param_name}' but it was not found in model_output or model_input.\n"
                f"It is probably because the model you try to run does not provide this access.\n\n"
                f"Available in model_output: {_get_non_none_fields(model_output)}\n"
                f"Available in model_input: {_get_non_none_fields(model_input)}\n"
                f"Available in targets: {list(model_input.targets.keys()) if model_input.targets else []}"
            )

    # Call the loss function with matched arguments
    try:
        return loss_func(**kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Error calling {type(loss_func).__name__}.__call__ with arguments "
            f"{list(kwargs.keys())}: {e}"
        ) from e


def _get_non_none_fields(obj: ModelOutput | ModelInput) -> list[str]:
    """Helper to get list of non-None field names from a Pydantic model."""
    if hasattr(obj, 'model_fields'):
        # Pydantic v2
        return [name for name in obj.model_fields.keys() if getattr(obj, name) is not None]
    else:
        # Fallback for dataclass or Pydantic v1
        return [name for name, value in obj.__dict__.items() if value is not None]


def _compute_combined_loss(
    model_output: ModelOutput,
    model_input: ModelInput,
    loss_func: CombinedLoss,
) -> Float[Tensor, "bsz"]:
    """Compute combined loss by recursively calling resolution on each component.

    CombinedLoss is a weighted combination of multiple loss functions. This
    function computes each component loss separately (using recursive calls to
    compute_loss_from_model_data) and combines them with their weights.

    Args:
        model_output: Model outputs available for all component losses
        model_input: Model inputs available for all component losses
        loss_func: CombinedLoss instance with losses and weights

    Returns:
        Weighted combination of component losses
    """
    component_losses = []
    for component_loss in loss_func.losses:
        # Recursive call - each component loss gets resolved independently
        component_loss_value = compute_loss_from_model_data(
            model_output, model_input, component_loss
        )
        component_losses.append(component_loss_value)

    # Combine with weights
    combined = sum(
        weight * loss_value
        for weight, loss_value in zip(loss_func.weights, component_losses)
    )

    return combined
