"""Custom Triton kernels to speed up inference-heavy (no autograd) forward passes."""
import functools
import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
except ImportError:  # e.g., non-Linux platforms
    triton = None  # ty: ignore[invalid-assignment]

if triton is not None:

    @triton.jit
    def _layer_norm_kernel(X, W, B, Y, N, eps, HAS_W: tl.constexpr, HAS_B: tl.constexpr, BLOCK: tl.constexpr):
        # One program per row; the row fits in a single block, so a single read of X.
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / N
        xc = tl.where(mask, x - mean, 0.0)
        y = xc / tl.sqrt(tl.sum(xc * xc, axis=0) / N + eps)
        if HAS_W:
            y *= tl.load(W + cols, mask=mask).to(tl.float32)
        if HAS_B:
            y += tl.load(B + cols, mask=mask).to(tl.float32)
        tl.store(Y + row * N + cols, y.to(Y.dtype.element_ty), mask=mask)


def _layer_norm_forward(ln: torch.nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    needs_autograd = torch.is_grad_enabled() and (x.requires_grad or (ln.weight is not None and ln.weight.requires_grad))
    if needs_autograd or not x.is_cuda or len(ln.normalized_shape) != 1:
        return F.layer_norm(x, ln.normalized_shape, ln.weight, ln.bias, ln.eps)
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.shape[-1]
    block = triton.next_power_of_2(n)
    _layer_norm_kernel[(x.numel() // n,)](
        x, ln.weight, ln.bias, y, n, ln.eps,
        HAS_W=ln.weight is not None, HAS_B=ln.bias is not None,
        BLOCK=block, num_warps=min(max(block // 256, 1), 16),  # ty: ignore[unknown-argument]
    )
    return y


def use_triton_layer_norm(module: torch.nn.Module) -> None:
    """Swap each `nn.LayerNorm` forward in `module` for a Triton kernel (~4x faster on no-grad passes)."""
    if triton is None or not torch.cuda.is_available():
        logger.info("Triton unavailable; keeping PyTorch LayerNorm.")
        return
    for m in module.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.forward = functools.partial(_layer_norm_forward, m)  # ty: ignore[invalid-assignment]
