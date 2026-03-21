
from jaxtyping import Float
from torch import Tensor

IGNORE_INDEX: int = -100
"""Token index to ignore in loss computation (masked out in cross-entropy).
-100 is also the default in torch's CE."""


def masked_mean(
    values: Float[Tensor, "bsz seq_len"],
    mask: Float[Tensor, "bsz seq_len"]
) -> Float[Tensor, "bsz"]:
    """Compute mean of values over `seq_len` dimension, considering only positions where `mask == 1`."""
    masked_values = values * mask
    sum_values = masked_values.sum(dim=-1)  # (bsz,)
    count_values = mask.sum(dim=-1).clamp(min=1.0)  # (bsz,)
    mean_values = sum_values / count_values  # (bsz,)
    return mean_values
