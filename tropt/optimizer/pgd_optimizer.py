import logging
import math
from typing import Optional

import torch
import torch.nn.functional as F
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
    LossTokenAccessMixin,
)
from tropt.optimizer import BaseOptimizer, OptimizerResult
from tropt.optimizer.utils.retokenization import retokenize_transform
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)


class PGDOptimizer(BaseOptimizer):
    """
    Projected Gradient Descent (PGD) for LLMs.
    Paper: https://arxiv.org/abs/2402.09154

    Optimizes continuous probability distributions over tokens with two projections:
    1. Simplex projection (Duchi et al. 2008): ensures valid probabilities
    2. Tsallis entropy projection (Gini index): controls relaxation error

    The entropy projection strength is annealed and coupled to the LR schedule,
    and dynamically scaled by the relaxation gap (discrete vs. relaxed loss).
    """

    model_requirements = (LossTokenAccessMixin, GradientTokenAccessMixin)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # PGD-specific parameters:
        num_steps: int = 5000,
        learning_rate: float = 0.11,
        target_entropy: float = 0.4,
        # Gradient clipping:
        grad_clip_value: float = 20.0,
        # LR scheduling (Appendix A):
        lr_warmup_steps: int = 100,
        cosine_T_0: int = 60,
        cosine_eta_min: float = 0.325,
        # Entropy annealing:
        entropy_anneal_steps: int = 250,
        # Patience (Appendix A):
        patience: int = 100,
        # Numerical stability:
        eps: float = 1e-12,
    ):
        """
        Args:
            num_steps: Number of optimization iterations.
            learning_rate: Base learning rate for Adam.
            target_entropy: Target Tsallis entropy (Gini index) after annealing.
            grad_clip_value: Per-token L2 gradient norm clip.
            lr_warmup_steps: Steps for linear LR warmup before cosine annealing.
            cosine_T_0: Period for CosineAnnealingWarmRestarts.
            cosine_eta_min: Minimum LR for cosine annealing.
            entropy_anneal_steps: Steps over which entropy target ramps from 0 to target_entropy.
            patience: Reset to best after this many steps without improvement.
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.target_entropy = target_entropy
        self.grad_clip_value = grad_clip_value
        self.lr_warmup_steps = lr_warmup_steps
        self.cosine_T_0 = cosine_T_0
        self.cosine_eta_min = cosine_eta_min
        self.entropy_anneal_steps = entropy_anneal_steps
        self.patience = patience
        self.eps = eps

    # ------------------------------------------------------------------ #
    #  Simplex projection (Algorithm 2 in the paper)
    # ------------------------------------------------------------------ #
    def _simplex_projection(
        self, s: Float[Tensor, "... vocab_size"]
    ) -> Float[Tensor, "... vocab_size"]:
        """
        Project onto the probability simplex. Supports batched input.
        Based on Duchi et al. 2008 / Blondel et al. 2014.
        """
        orig_shape = s.shape
        if s.dim() == 1:
            s = s.unsqueeze(0)

        s = torch.clamp(s, min=0.0)  # ensure non-negative before projection

        b, d = s.shape
        mu_sorted = torch.sort(s, dim=-1, descending=True).values
        mu_cumsum = torch.cumsum(mu_sorted, dim=-1)
        indices = torch.arange(1, d + 1, device=s.device, dtype=s.dtype)
        condition = mu_sorted - (mu_cumsum - 1) / indices > 0
        rho = condition.sum(dim=-1)  # (b,)
        theta = (mu_cumsum[torch.arange(b, device=s.device), rho - 1] - 1) / rho
        result = torch.clamp(s - theta.unsqueeze(-1), min=0)

        return result.view(orig_shape)

    # ------------------------------------------------------------------ #
    #  Tsallis entropy projection (Algorithm 3 in the paper)
    # ------------------------------------------------------------------ #
    def _tsallis_entropy_projection(
        self,
        s: Float[Tensor, "... vocab_size"],
        target_entropy: float,
    ) -> Float[Tensor, "... vocab_size"]:
        """
        Project onto the Tsallis entropy ball (q=2, Gini index) intersected
        with the simplex. Follows Algorithm 3 from the paper exactly.
        """
        orig_shape = s.shape
        if s.dim() == 1:
            s = s.unsqueeze(0)

        results = []
        for i in range(s.shape[0]):
            si = s[i]
            non_zero_mask = si > 0
            d = non_zero_mask.sum()

            if d == 0:
                results.append(si)
                continue

            # Center: uniform over non-zero entries
            center = torch.zeros_like(si)
            center[non_zero_mask] = 1.0 / d

            # Radius from target entropy
            radius_sq = max(0.0, 1.0 - target_entropy - 1.0 / d.item())
            radius = math.sqrt(radius_sq)

            direction = si - center
            dist = torch.linalg.norm(direction)

            if dist <= radius:
                # Already inside the entropy ball
                results.append(si)
            else:
                # Project onto sphere boundary, then re-project onto simplex
                projected = (radius / (dist + self.eps)) * direction + center
                projected = self._simplex_projection(projected)
                results.append(projected)

        result = torch.stack(results)
        return result.view(orig_shape)

    # ------------------------------------------------------------------ #
    #  Per-token gradient clipping (Appendix A)
    # ------------------------------------------------------------------ #
    def _clip_grad_per_token(
        self, grad: Float[Tensor, "trigger_seq_len vocab_size"]
    ) -> None:
        """Clip gradient L2 norm per token position, in-place."""
        norms = torch.linalg.norm(grad, dim=-1, keepdim=True)  # (seq, 1)
        scale = torch.where(
            norms > self.grad_clip_value,
            self.grad_clip_value / (norms + self.eps),
            torch.ones_like(norms),
        )
        grad.mul_(scale)

    # ------------------------------------------------------------------ #
    #  Entropy factor scheduling
    # ------------------------------------------------------------------ #
    def _get_entropy_factor(
        self, step: int, lr_ratio: float, relaxation_gap: float = 1.0
    ) -> float:
        """Compute effective entropy factor for this step.

        Combines: linear annealing (0 → target), LR coupling, and relaxation
        gap scaling (Appendix A).
        """
        # Linear ramp: 0 → target_entropy over entropy_anneal_steps
        progress = min(1.0, step / max(1, self.entropy_anneal_steps))
        annealed = progress * self.target_entropy

        # Scale in tandem with LR (Appendix A)
        factor = annealed * lr_ratio

        # Weaken when discrete and relaxed losses are close (Appendix A)
        gap = max(0.0, min(1.0, relaxation_gap))
        factor *= gap

        return factor

    def _get_lr_ratio(self, scheduler) -> float:
        """Current LR / base LR ratio for entropy-LR coupling."""
        last_lr = scheduler.get_last_lr()[0]
        base_lr = max(self.learning_rate, self.cosine_eta_min)
        return last_lr / base_lr if base_lr > 0 else 1.0

    # ------------------------------------------------------------------ #
    #  Discretization with retokenization (Appendix A)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _discretize(
        trigger_probs: Float[Tensor, "trigger_seq_len vocab_size"],
        tokenizer,
    ) -> Int[Tensor, "trigger_seq_len"]:
        """Discretize probabilities via argmax, then retokenize for consistency.

        Paper Appendix A: d(X̃) = tokenizer.encode(tokenizer.decode(argmax(X̃)))
        """
        raw_ids = trigger_probs.argmax(dim=-1)
        return retokenize_transform(raw_ids, tokenizer)

    # ------------------------------------------------------------------ #
    #  Main optimization loop (Algorithm 1)
    # ------------------------------------------------------------------ #
    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Targets = None,
    ) -> OptimizerResult:
        self._log_run_config_to_tracker(templates, initial_trigger, targets)

        # --- Initialization ---
        self.model.set_inputs_from_tokens(templates=templates, targets=targets)
        tokenizer = self.model.tokenizer
        trigger_ids = (
            tokenizer.encode(initial_trigger, add_special_tokens=False, return_tensors="pt")
            .to(self.model.device, torch.int64)
        )
        vocab_size = self.model.vocab_size
        device = self.model.device
        dtype = self.model.dtype

        trigger_ids_init = trigger_ids.squeeze(0)  # (trigger_seq_len,)

        # Initialize one-hot probability distribution (Algorithm 1, line 3)
        trigger_probs = F.one_hot(trigger_ids_init, num_classes=vocab_size).to(
            dtype=dtype, device=device
        )
        trigger_probs.requires_grad_(True)

        # Adam optimizer
        optimizer = torch.optim.Adam([trigger_probs], lr=self.learning_rate)

        # LR scheduler: linear warmup then CosineAnnealingWarmRestarts (Appendix A)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0,
            total_iters=max(1, self.lr_warmup_steps),
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.cosine_T_0, eta_min=self.cosine_eta_min,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.lr_warmup_steps],
        )

        # --- Tracking ---
        loss_per_step = []
        trigger_strings = []

        # Best tracking / early stopping (Algorithm 1, lines 11-12)
        best_loss = float("inf")
        best_trigger_ids = None
        best_trigger_str = None
        best_trigger_probs = None  # for patience resets
        steps_since_improvement = 0
        relaxation_gap = 1.0  # starts at 1 (no weakening)

        pbar = tqdm(range(self.num_steps), desc="PGD Optimization")

        for step in pbar:
            optimizer.zero_grad()

            # --- Compute gradient (Algorithm 1, line 5) ---
            trigger_grad = self.model.compute_grad_from_tokens(
                candidate_trigger_probs=trigger_probs.unsqueeze(0),
                loss_func=self.loss_func,
                do_gumbel_softmax=False,
                normalize_grads=False,
            )
            # Remove candidate dim: (1, seq, vocab) → (seq, vocab)
            trigger_grad = trigger_grad.squeeze(0)

            # Per-token gradient clipping (Appendix A)
            self._clip_grad_per_token(trigger_grad)

            # Set gradient for Adam
            trigger_probs.grad = trigger_grad

            # --- Adam step (Algorithm 1, line 6) ---
            optimizer.step()
            scheduler.step()

            # --- Projections (Algorithm 1, lines 7-8) ---
            lr_ratio = self._get_lr_ratio(scheduler)
            current_entropy_target = self._get_entropy_factor(
                step, lr_ratio, relaxation_gap
            )

            with torch.no_grad():
                # Simplex projection (Algorithm 1, line 7)
                trigger_probs.data = self._simplex_projection(trigger_probs.data)

                # Entropy projection (Algorithm 1, line 8)
                if current_entropy_target > 0:
                    trigger_probs.data = self._tsallis_entropy_projection(
                        trigger_probs.data, current_entropy_target
                    )

            # --- Discretize and evaluate (Algorithm 1, lines 9-10) ---
            with torch.no_grad():
                current_trigger_ids = self._discretize(trigger_probs.data, tokenizer)

                current_loss = self.model.compute_loss_from_tokens(
                    current_trigger_ids.unsqueeze(0),
                    loss_func=self.loss_func,
                ).item()

                current_trigger_str = tokenizer.decode(
                    current_trigger_ids, skip_special_tokens=True
                )

            # --- Relaxation gap for dynamic entropy scaling (Appendix A) ---
            # gap = (discrete_loss - relaxed_loss) / discrete_loss
            # When close to 0, the relaxation is tight → weaken entropy projection
            if current_loss > self.eps:
                # The "relaxed loss" uses the soft probs; discrete uses argmax ids.
                # current_loss is already the discrete loss from retokenized argmax.
                # We approximate relaxed loss by computing loss on raw argmax ids
                # (without retokenization), which is cheaper than a full soft forward.
                with torch.no_grad():
                    raw_argmax_ids = trigger_probs.data.argmax(dim=-1)
                    relaxed_loss_val = self.model.compute_loss_from_tokens(
                        raw_argmax_ids.unsqueeze(0),
                        loss_func=self.loss_func,
                    ).item()
                relaxation_gap = abs(current_loss - relaxed_loss_val) / (
                    abs(current_loss) + self.eps
                )
            else:
                relaxation_gap = 1.0

            # --- Best tracking (Algorithm 1, lines 11-12) ---
            if current_loss < best_loss:
                best_loss = current_loss
                best_trigger_ids = current_trigger_ids.clone()
                best_trigger_str = current_trigger_str
                best_trigger_probs = trigger_probs.data.clone()
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            # --- Patience: reset to best (Appendix A) ---
            if self.patience > 0 and steps_since_improvement >= self.patience:
                if best_trigger_probs is not None:
                    with torch.no_grad():
                        # Reinitialize to one-hot of best discrete trigger
                        reset_probs = F.one_hot(
                            best_trigger_ids, num_classes=vocab_size
                        ).to(dtype=dtype, device=device)
                        trigger_probs.data.copy_(reset_probs)
                    steps_since_improvement = 0
                    logger.debug(f"Patience reset at step {step}")

            # --- Logging ---
            loss_per_step.append(current_loss)
            trigger_strings.append(current_trigger_str)

            self.tracker.log({
                "loss": current_loss,
                "best_loss": best_loss,
                "entropy_target": current_entropy_target,
                "lr": scheduler.get_last_lr()[0],
                **self.loss_func.get_loss_log_dict(),
                **self.model.get_usage_stats(),
            })

            pbar.set_description(
                f"loss={current_loss:.4f} best={best_loss:.4f} "
                f"trigger={current_trigger_str[:30]}"
            )

        # --- Use the best trigger found during optimization ---
        if best_trigger_ids is None:
            best_trigger_ids = self._discretize(trigger_probs.data, tokenizer)
            best_trigger_str = tokenizer.decode(
                best_trigger_ids, skip_special_tokens=True
            )
            best_loss = loss_per_step[-1] if loss_per_step else float("inf")

        # Construct full prompts
        assert isinstance(best_trigger_str, str)
        full_prompt = [
            t.replace(OPTIMIZED_TRIGGER_PLACEHOLDER, best_trigger_str)
            for t in templates
        ]

        result = OptimizerResult(
            best_loss=best_loss,
            best_trigger_str=best_trigger_str,
            best_trigger_ids=best_trigger_ids,
            losses=loss_per_step,
            trigger_strs=trigger_strings,
            full_prompt=full_prompt,
        )

        self.tracker.log({
            "best_loss": result.best_loss,
            "best_trigger_str": result.best_trigger_str,
        })
        self.model.reset_inputs_from_tokens()
        return result
