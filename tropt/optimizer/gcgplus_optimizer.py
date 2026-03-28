import logging
import math
from typing import List, Literal, Optional, Set

import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from tropt.common import (
    DEFAULT_INIT_TRIGGER,
    OPTIMIZED_TRIGGER_PLACEHOLDER,
    Targets,
    TextTemplates,
)
from tropt.loss import BaseLoss
from tropt.model import (
    BaseModel,
    GradientTokenAccessMixin,
    LossTextAccessMixin,
    LossTokenAccessMixin,
    TokenAccessMixin,
)
from tropt.optimizer import BaseOptimizer, OptimizerResult
from tropt.optimizer.utils.retokenization import retokenize_filtering
from tropt.optimizer.utils.running_best import RunningBest
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)


class GCGPlusOptimizer(BaseOptimizer):
    """Unified optimizer supporting GCG++, RAL, and PAL attacks.

    Two-stage design:
      1. Candidate selection (on proxy model) — gradient-based or random.
      2. Candidate evaluation (on target model) — via text or token access.

    References:
      - PAL: https://arxiv.org/abs/2402.09674
      - GCG: https://arxiv.org/abs/2307.15043

    Note: the official PAL implementation accumulates proxy-filtered candidates
    over multiple proxy-only steps (proxy_tune_period) before querying the
    target, reducing API costs. This is not yet implemented here — every step
    queries the target. See https://github.com/chawins/pal for details.
    """

    model_requirements = (LossTextAccessMixin,)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # Proxy model for candidate selection (defaults to model):
        proxy_model: Optional[BaseModel] = None,
        # Candidate selection strategy:
        candidate_selection: Literal["gradient", "random"] = "gradient",
        # Core GCG params:
        num_steps: int = 500,
        n_candidates: int = 512,
        sample_topk: int = 256,
        sample_n_replace: int = 1,
        token_constraints: TokenConstraints = TokenConstraints(),
        use_retokenize: bool = True,
        # Later tricks:
        oversample_factor: float = 1.5,
        momentum: float = 0.0,
        skip_visited: bool = False,
        # Evaluation mode:
        use_token_eval: bool = False,
        # PAL proxy filtering:
        proxy_filter_k: Optional[int] = None,
    ):
        """
        Args:
            proxy_model: Model used for candidate selection (gradients/tokenizer).
                If None, defaults to `model` (self-proxy / white-box).
            candidate_selection: "gradient" uses gradient-ranked top-k sampling,
                "random" uses uniform random token sampling.
            oversample_factor: Generate n_candidates * factor candidates, then
                truncate after retokenization filtering. Only effective when > 1.0.
            momentum: Gradient momentum coefficient (mu). When > 0, uses
                m = mu*m + grad for candidate ranking instead of raw gradient.
            skip_visited: Skip candidate suffixes that have been evaluated before.
            use_token_eval: If True, evaluate candidates via compute_loss_from_tokens
                on the target (requires LossTokenAccessMixin). Otherwise use text access.
            proxy_filter_k: If set, filter candidates down to top-K by proxy loss
                before evaluating on the target (PAL's proxy filtering step).
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        # Proxy model setup
        self.proxy_model = model if proxy_model is None else proxy_model
        assert isinstance(self.proxy_model, TokenAccessMixin), (
            "proxy_model must support TokenAccessMixin (tokenizer access)"
        )
        if candidate_selection == "gradient":
            assert isinstance(self.proxy_model, GradientTokenAccessMixin), (
                "candidate_selection='gradient' requires proxy_model with GradientTokenAccessMixin"
            )
        if proxy_filter_k is not None:
            assert isinstance(self.proxy_model, LossTokenAccessMixin), (
                "proxy_filter_k requires proxy_model with LossTokenAccessMixin"
            )

        # Token eval validation
        if use_token_eval:
            assert isinstance(model, LossTokenAccessMixin), (
                "use_token_eval=True requires target model with LossTokenAccessMixin"
            )

        # Save params
        self.candidate_selection = candidate_selection
        self.num_steps = num_steps
        self.n_candidates = n_candidates
        self.sample_topk = sample_topk
        self.sample_n_replace = sample_n_replace
        self.token_constraints = token_constraints
        self.use_retokenize = use_retokenize
        self.oversample_factor = oversample_factor
        self.momentum = momentum
        self.skip_visited = skip_visited
        self.use_token_eval = use_token_eval
        self.proxy_filter_k = proxy_filter_k

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:
        self._log_run_config_to_tracker(templates, initial_trigger, targets)

        # --- Initialization ---
        proxy = self.proxy_model
        proxy_tokenizer = proxy.tokenizer

        proxy.set_inputs_from_tokens(templates=templates, targets=targets)
        if not self.use_token_eval:
            self.model.set_inputs_from_texts(templates=templates, targets=targets)
        else:
            self.model.set_inputs_from_tokens(templates=templates, targets=targets)

        trigger_ids: Int[Tensor, "trigger_seq_len"] = (
            proxy_tokenizer.encode(
                initial_trigger, add_special_tokens=False, return_tensors="pt"
            )
            .to(proxy.device, torch.int64)
            .squeeze(0)
        )

        vocab_size = proxy.vocab_size
        blacklist_ids = self.token_constraints.get_blacklist_ids(proxy_tokenizer, vocab_size)

        best = RunningBest()
        visited: Set[str] = set()
        momentum_buffer: Optional[Tensor] = None

        # Number of candidates to generate (oversample if retokenize is on)
        n_generate = self.n_candidates
        if self.use_retokenize and self.oversample_factor > 1.0:
            n_generate = math.ceil(self.n_candidates * self.oversample_factor)

        # Initial loss
        current_loss = self._evaluate_candidates(
            trigger_ids.unsqueeze(0), proxy_tokenizer
        ).item()
        trigger_str = proxy_tokenizer.decode(trigger_ids, skip_special_tokens=True)
        if self.skip_visited:
            visited.add(trigger_str)
        self.tracker.log({"loss": current_loss, **self.loss_func.get_loss_log_dict(), **self.model.get_usage_stats()})

        pbar = tqdm(range(self.num_steps))

        for _ in pbar:
            # === Stage 1: Candidate Selection (on proxy) ===
            if self.candidate_selection == "gradient":
                trigger_grad: Float[Tensor, "trigger_seq_len vocab_size"] = (
                    proxy.compute_grad_from_tokens(
                        candidate_trigger_ids=trigger_ids.unsqueeze(0),
                        loss_func=self.loss_func,
                        normalize_grads=True,
                    ).squeeze(0)
                )
                # Apply momentum
                if self.momentum > 0:
                    if momentum_buffer is None:
                        momentum_buffer = trigger_grad.clone()
                    else:
                        momentum_buffer = self.momentum * momentum_buffer + trigger_grad
                    ranking_signal = momentum_buffer
                else:
                    ranking_signal = trigger_grad

                candidate_trigger_ids = self._sample_ids_from_grad(
                    trigger_ids=trigger_ids,
                    trigger_grad=ranking_signal,
                    blacklist_ids=blacklist_ids,
                    n_generate=n_generate,
                )
            else:  # "random"
                candidate_trigger_ids = self._sample_random_candidates(
                    trigger_ids=trigger_ids,
                    blacklist_ids=blacklist_ids,
                    n_generate=n_generate,
                )

            # === Retokenization filtering ===
            if self.use_retokenize:
                candidate_trigger_ids = retokenize_filtering(
                    candidate_trigger_ids, proxy_tokenizer
                )
            # Truncate to n_candidates (after oversample + filter)
            if len(candidate_trigger_ids) > self.n_candidates:
                candidate_trigger_ids = candidate_trigger_ids[: self.n_candidates]

            if len(candidate_trigger_ids) == 0:
                logger.warning("All candidates filtered out, skipping step.")
                continue

            # === Skip visited ===
            if self.skip_visited:
                candidate_trigger_ids, _ = self._filter_visited(
                    candidate_trigger_ids, proxy_tokenizer, visited
                )
                if len(candidate_trigger_ids) == 0:
                    logger.warning("All candidates already visited, skipping step.")
                    continue

            # === PAL proxy filtering ===
            if self.proxy_filter_k is not None and self.proxy_filter_k < len(candidate_trigger_ids):
                proxy_losses = proxy.compute_loss_from_tokens(
                    candidate_trigger_ids, loss_func=self.loss_func
                )  # (n_templates, n_candidates)
                # Average over templates, take top-K
                avg_proxy_losses = proxy_losses.mean(dim=0) if proxy_losses.dim() > 1 else proxy_losses
                topk_indices = avg_proxy_losses.topk(
                    self.proxy_filter_k, largest=False
                ).indices
                candidate_trigger_ids = candidate_trigger_ids[topk_indices]

            # === Stage 2: Candidate Evaluation (on target) ===
            losses = self._evaluate_candidates(candidate_trigger_ids, proxy_tokenizer)

            current_loss = losses.min().item()
            trigger_ids = candidate_trigger_ids[losses.argmin()]
            trigger_str = proxy_tokenizer.decode(trigger_ids, skip_special_tokens=True)

            if self.skip_visited:
                # Only the selected (best) suffix is marked as visited,
                # matching the official PAL "visited" skip mode.
                visited.add(trigger_str)

            best.update(loss=current_loss, trigger_ids=trigger_ids, trigger_str=trigger_str)
            self.tracker.log({"loss": current_loss, **self.loss_func.get_loss_log_dict(), **self.model.get_usage_stats()})
            pbar.set_description(f"loss={current_loss: .4f}, trigger={trigger_str}")

        # --- Finalize ---
        result = OptimizerResult(
            best_loss=best.loss,
            best_trigger_str=best.trigger_str,
            best_trigger_ids=best.trigger_ids,
            losses=best.losses,
            trigger_strs=best.trigger_strs,
        )
        self.tracker.log({"best_loss": result.best_loss, "best_trigger_str": result.best_trigger_str})

        proxy.reset_inputs_from_tokens()
        if not self.use_token_eval:
            self.model.reset_inputs_from_texts()
        else:
            self.model.reset_inputs_from_tokens()

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_candidates(
        self,
        candidate_trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"],
        proxy_tokenizer,
    ) -> Float[Tensor, "n_candidates"]:
        """Evaluate candidates on the target model. Returns per-candidate loss."""
        if self.use_token_eval:
            losses = self.model.compute_loss_from_tokens(
                candidate_trigger_ids, loss_func=self.loss_func
            )  # (n_templates, n_candidates)
        else:
            candidate_strs = [
                proxy_tokenizer.decode(cid, skip_special_tokens=True)
                for cid in candidate_trigger_ids
            ]
            losses = self.model.compute_loss_from_texts(
                candidate_strs, loss_func=self.loss_func
            )  # (n_candidates,) or (n_templates, n_candidates)

        # Reduce to (n_candidates,) if needed
        if losses.dim() > 1:
            losses = losses.mean(dim=0)
        return losses

    def _sample_ids_from_grad(
        self,
        trigger_ids: Int[Tensor, "trigger_seq_len"],
        trigger_grad: Float[Tensor, "trigger_seq_len vocab_size"],
        blacklist_ids: List[int],
        n_generate: int,
    ) -> Int[Tensor, "n_generate trigger_seq_len"]:
        """Sample candidate token sequences based on gradient ranking (GCG-style)."""
        trigger_seq_len, vocab_size = trigger_grad.shape
        device = trigger_grad.device
        candidate_trigger_ids = trigger_ids.repeat(n_generate, 1).clone()

        trigger_grad = trigger_grad.clone()
        trigger_grad[:, blacklist_ids] = float("inf")

        topk_ids = (-trigger_grad).topk(self.sample_topk, dim=-1).indices

        # Random positions to flip
        sampled_ids_pos = torch.rand(
            n_generate, trigger_seq_len, device=device
        ).argsort(dim=-1)[..., : self.sample_n_replace]

        # Select relevant top-k lists and sample one token from each
        relevant_topk_lists = topk_ids[sampled_ids_pos]
        rand_k_indices = torch.randint(
            0,
            self.sample_topk,
            (n_generate, self.sample_n_replace, 1),
            device=device,
        )
        sampled_ids_val = torch.gather(
            input=relevant_topk_lists,
            dim=-1,
            index=rand_k_indices,
        ).squeeze(-1)

        candidate_trigger_ids = candidate_trigger_ids.scatter_(
            dim=-1,
            index=sampled_ids_pos,
            src=sampled_ids_val,
        )
        return candidate_trigger_ids

    def _sample_random_candidates(
        self,
        trigger_ids: Int[Tensor, "trigger_seq_len"],
        blacklist_ids: List[int],
        n_generate: int,
    ) -> Int[Tensor, "n_generate trigger_seq_len"]:
        """Sample candidates by randomly replacing tokens (RAL-style).

        Follows the official RAL implementation: pre-computes valid token IDs
        and samples uniformly from them (rather than rejection sampling).
        """
        trigger_seq_len = trigger_ids.shape[0]
        device = trigger_ids.device
        vocab_size = self.proxy_model.vocab_size

        candidate_trigger_ids = trigger_ids.repeat(n_generate, 1).clone()

        # Random positions to flip
        sampled_ids_pos = torch.rand(
            n_generate, trigger_seq_len, device=device
        ).argsort(dim=-1)[..., : self.sample_n_replace]

        # Pre-compute valid token IDs (exclude blacklist) — matches RAL's approach
        token_mask = torch.ones(vocab_size, device=device, dtype=torch.bool)
        if blacklist_ids:
            token_mask[blacklist_ids] = False
        valid_token_ids = token_mask.nonzero(as_tuple=False).squeeze(-1)

        # Sample uniformly from valid tokens
        rand_indices = torch.randint(
            0, len(valid_token_ids), (n_generate, self.sample_n_replace), device=device
        )
        random_tokens = valid_token_ids[rand_indices]

        candidate_trigger_ids = candidate_trigger_ids.scatter_(
            dim=-1,
            index=sampled_ids_pos,
            src=random_tokens,
        )
        return candidate_trigger_ids

    def _filter_visited(
        self,
        candidate_trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"],
        tokenizer,
        visited: Set[str],
    ) -> tuple:
        """Filter out candidates whose decoded string is in the visited set."""
        keep_mask = []
        for cid in candidate_trigger_ids:
            s = tokenizer.decode(cid, skip_special_tokens=True)
            keep_mask.append(s not in visited)
        keep_mask = torch.tensor(keep_mask, device=candidate_trigger_ids.device)
        filtered = candidate_trigger_ids[keep_mask]
        return filtered, keep_mask
