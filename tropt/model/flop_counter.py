"""FLOP counting utilities for model invoke methods.

FLOP counting is handled at the ``invoke_from_tokens`` / ``invoke_from_texts``
level — these are the model-call bottleneck, and the only place where usage
statistics and FLOPs are tracked. Non-invoke computation in ``compute_*``
methods (tensor stacking, loss aggregation, etc.) is negligible compared to
the model forward/backward passes.

Uses the Kaplan et al. (2020) approximation (https://arxiv.org/abs/2001.08361):
``FLOPs_fwd ≈ 2·N·T``, ``FLOPs_bwd ≈ 4·N·T``.
Cheap and deterministic. Requires ``_model`` to be a HuggingFace
``PreTrainedModel``.
Adapted from https://github.com/romovpa/claudini.

**For model implementers**: invoke methods must call
:meth:`BaseModel._update_invoke_stats` *after* each model call. This handles
both usage statistics and FLOP counting in one place. For backward passes
(gradient methods), pass ``count_backward=True`` to the invoke method, which
propagates to ``_update_invoke_stats``.

Usage::

    model = LMHFModel("meta-llama/Llama-3.2-1B", ...)
    model.set_flop_counting("manual")   # False to disable
    model.reset_usage_stats()
    # ... run optimization ...
    stats = model.get_usage_stats()
    print(stats["usage/total_flops"])
"""

import logging

logger = logging.getLogger(__name__)

class FlopCounterBase:
    """Base class for FLOP counting. Subclasses implement specific counting methods."""

    def count_forward(self, n_tokens: int) -> int:
        """Count forward-pass FLOPs for a given number of tokens."""
        raise NotImplementedError

    def count_backward(self, n_tokens: int) -> int:
        """Count backward-pass FLOPs for a given number of tokens."""
        raise NotImplementedError

    def count_forward_backward(self, n_tokens: int) -> int:
        """Count combined forward+backward FLOPs for a given number of tokens."""
        raise NotImplementedError


class ManualFlopCounter(FlopCounterBase):
    """Track FLOPs using Kaplan et al. (2020) approximation.

    FLOPs_fwd  ≈ 2 · N_params · n_tokens
    FLOPs_bwd  ≈ 4 · N_params · n_tokens

    For MoE models, N_params is the *active* parameter count (shared params +
    expert params scaled by top-k / num_experts).
    """

    def __init__(self, model: "PreTrainedModel"):
        from transformers import PreTrainedModel

        if not isinstance(model, PreTrainedModel):
            raise TypeError(
                f"ManualFlopCounter requires a HuggingFace PreTrainedModel, "
                f"got {type(model).__name__}. "
                f"Manual FLOP counting is only supported for HF models."
            )
        self.total_params: int = model.num_parameters(exclude_embeddings=True)
        self.n_params: int = self._compute_active_params(model)
        logger.info(
            "ManualFlopCounter: %dM active params (%dM total non-embedding)",
            self.n_params // 10**6,
            self.total_params // 10**6,
        )

    # ---- active-param computation (handles dense & MoE) ----

    @staticmethod
    def _compute_active_params(model: "PreTrainedModel") -> int:
        """Active (per-token) parameter count, accounting for MoE sparsity.

        For dense models returns total non-embedding params.
        For MoE models expert parameters are scaled by (num_active / num_experts).
        Falls back to config-based counting when quantization is detected.
        """
        config = model.config
        num_experts = getattr(config, "num_local_experts", None) or getattr(
            config, "num_experts", None
        )
        num_active = (
            getattr(config, "num_experts_per_tok", None)
            or getattr(config, "num_selected_experts", None)
            or getattr(config, "top_k", None)
        )

        if not num_experts or not num_active or num_experts <= 1:
            # Dense model
            reported = model.num_parameters(exclude_embeddings=True)
            config_estimate = ManualFlopCounter._params_from_config(config)
            if config_estimate and reported < config_estimate * 0.5:
                logger.info(
                    "Quantized model detected: reported %dM params but config says %dM. "
                    "Using config estimate.",
                    reported // 10**6,
                    config_estimate // 10**6,
                )
                return config_estimate
            return reported

        # MoE model
        expert_params = 0
        shared_params = 0
        for name, param in model.named_parameters():
            n = param.numel()
            if "embed" in name or "lm_head" in name:
                continue
            elif "expert" in name:
                expert_params += n
            else:
                shared_params += n

        config_expert_params = ManualFlopCounter._expert_params_from_config(config)
        if config_expert_params and expert_params < config_expert_params * 0.1:
            logger.info(
                "Quantized MoE detected: named_parameters reports %dM expert params "
                "but config says %dM. Using config-based counting.",
                expert_params // 10**6,
                config_expert_params // 10**6,
            )
            expert_params = config_expert_params
            config_shared = ManualFlopCounter._shared_params_from_config(config)
            if config_shared:
                shared_params = config_shared

        active_expert_params = int(expert_params * num_active / num_experts)
        active_params = shared_params + active_expert_params
        total_non_emb = shared_params + expert_params

        logger.info(
            "MoE: %d experts, top-%d active. "
            "Params: %dM shared + %dM expert (%.0f%% active) = %dM active / %dM total",
            num_experts,
            num_active,
            shared_params // 10**6,
            expert_params // 10**6,
            100 * num_active / num_experts,
            active_params // 10**6,
            total_non_emb // 10**6,
        )
        return active_params

    # ---- config-based fallbacks (for quantized models) ----

    @staticmethod
    def _expert_params_from_config(config) -> int | None:
        h = getattr(config, "hidden_size", None)
        intermediate = getattr(config, "intermediate_size", None)
        n_layers = getattr(config, "num_hidden_layers", None)
        num_experts = getattr(config, "num_local_experts", None) or getattr(
            config, "num_experts", None
        )
        if not all([h, intermediate, n_layers, num_experts]):
            return None
        return 3 * h * intermediate * num_experts * n_layers

    @staticmethod
    def _shared_params_from_config(config) -> int | None:
        h = getattr(config, "hidden_size", None)
        n_layers = getattr(config, "num_hidden_layers", None)
        n_heads = getattr(config, "num_attention_heads", None)
        n_kv_heads = getattr(config, "num_key_value_heads", None)
        head_dim = getattr(config, "head_dim", None)
        num_experts = getattr(config, "num_local_experts", None) or getattr(
            config, "num_experts", None
        )
        if not all([h, n_layers, n_heads]):
            return None
        if head_dim is None:
            head_dim = h // n_heads
        if n_kv_heads is None:
            n_kv_heads = n_heads
        attn = (
            h * (n_heads * head_dim)
            + 2 * h * (n_kv_heads * head_dim)
            + (n_heads * head_dim) * h
        )
        ln = 2 * h
        router = h * num_experts if num_experts else 0
        return (attn + ln + router) * n_layers

    @staticmethod
    def _params_from_config(config) -> int | None:
        h = getattr(config, "hidden_size", None)
        intermediate = getattr(config, "intermediate_size", None)
        n_layers = getattr(config, "num_hidden_layers", None)
        n_heads = getattr(config, "num_attention_heads", None)
        if not all([h, intermediate, n_layers, n_heads]):
            return None
        head_dim = getattr(config, "head_dim", h // n_heads)
        n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
        attn = (
            h * (n_heads * head_dim)
            + 2 * h * (n_kv_heads * head_dim)
            + (n_heads * head_dim) * h
        )
        mlp = 3 * h * intermediate
        ln = 2 * h
        return (attn + mlp + ln) * n_layers

    # ---- counting helpers ----

    def count_forward(self, n_tokens: int) -> int:
        return 2 * self.n_params * n_tokens

    def count_backward(self, n_tokens: int) -> int:
        return 4 * self.n_params * n_tokens

    def count_forward_backward(self, n_tokens: int) -> int:
        return 6 * self.n_params * n_tokens


# ---------------------------------------------------------------------------
# [DISABLED] Torch automatic FLOP-counting. Disabled due to potential instability and limitations.
# ---------------------------------------------------------------------------


# def track_flops(includes_backward: bool = False):
#     """Decorator for ``compute_*`` methods — dispatches FLOP counting by mode.

#     ``"torch"`` — wraps the method with ``FlopCounterMode``.

#     ``"manual"`` — measures the delta of ``_token_used`` before/after the
#     method, then applies the Kaplan approximation. Requires a
#     ``_kaplan_flop_counter`` (:class:`KaplanFlopCounter`) on the model —
#     provided automatically by :class:`HuggingFaceBackendModel`.

#     Args:
#         includes_backward: The wrapped method includes a backward pass
#             (``"manual"`` uses ``6·N·T`` instead of ``2·N·T``).
#     """

#     def decorator(method):
#         @wraps(method)
#         def wrapper(self, *args, **kwargs):
#             mode = getattr(self, "count_flops", False)
#             if not mode:
#                 return method(self, *args, **kwargs)

#             # --- torch mode ---
#             if mode == "torch":
#                 from torch.utils.flop_counter import FlopCounterMode as _FlopCounterMode

#                 with _FlopCounterMode(display=False) as flop_counter:
#                     result = method(self, *args, **kwargs)
#                 self._update_usage_stats(flops=flop_counter.get_total_flops())
#                 return result

#             # --- manual mode ---
#             if mode == "manual":
#                 counter = getattr(self, "_kaplan_flop_counter", None)
#                 if counter is None:
#                     logger.warning(
#                         "count_flops='manual' but no KaplanFlopCounter on model; "
#                         "skipping FLOP counting."
#                     )
#                     return method(self, *args, **kwargs)

#                 tokens_before = getattr(self, "_token_used", 0)
#                 result = method(self, *args, **kwargs)
#                 tokens_after = getattr(self, "_token_used", 0)
#                 delta_tokens = tokens_after - tokens_before

#                 if includes_backward:
#                     flops = counter.count_forward_backward(delta_tokens)
#                 else:
#                     flops = counter.count_forward(delta_tokens)
#                 self._update_usage_stats(flops=flops)
#                 return result

#             raise ValueError(
#                 f"Unknown count_flops mode: {mode!r}. Use False, 'torch', or 'manual'."
#             )

#         return wrapper

#     return decorator
