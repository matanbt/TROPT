import logging
from enum import Enum
from typing import Literal, Optional, Type

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
from tropt.optimizer.utils.running_best import RunningBest
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)


class GBDAOptimizer(BaseOptimizer):
    """
    Gradient-Based Distributional Attack (GBDA).
    Paper: https://arxiv.org/abs/2104.13733
    Reference implementation: https://github.com/facebookresearch/text-adversarial-attack/blob/main/whitebox_attack.py

    Optimizes a continuous logit matrix theta (model distribution over L tokens, where L is the
    trigger sequence length) that can be used to sample triggers (w/ Gumbel-softmax).
    Throughout the optimization, the matrix theta used to provide a weighted sum of input embedding,
    on which the loss and gradients can be computed, and subsequently update theta.
    After each optimization step, and in particular at the end, theta can be used to sample discrete triggers.
    """

    model_requirements = (LossTokenAccessMixin, GradientTokenAccessMixin)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # GBDA-specific parameters:
        num_steps: int = 100,
        n_grad_samples: int = 10,
        n_final_gumbel_samples: int = 100,

        # Init paramaters:
        initial_coeff: float = 15.0,
        init_mode: Literal["from_trigger", "random"] = "from_trigger",
        init_noise_scale: float = 2.0,

        # Temperature schedule parameters:
        temp_schedule: Literal["linear", "gradual"] = "linear",
        temp_start: float = 1.0,
        temp_end: float = 0.1,

        # Optimization parameters:
        gd_optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam,
        use_lr_schedule: bool = True,
        learning_rate: float = 0.3,
        grad_clip_norm: Optional[float] = None,
    ):
    # TODO rearrange and categorize the docstring parameters 
        """
        Args:
            num_steps: Number of optimization steps.
            n_grad_samples: Gumbel-softmax samples per gradient step.
            initial_coeff: Initial logit value at original token positions (used when init_mode="from_trigger").
            temp_schedule: Temperature schedule type. "linear" for linear annealing, "gradual" for
                3-phase schedule (explore 2.5->1.0, refine 1.0->0.5, discretize 0.5->0.01).
                When "gradual", temp_start and temp_end are ignored.
            temp_start: Starting Gumbel-softmax temperature (used only with "linear" schedule).
            temp_end: Ending Gumbel-softmax temperature (used only with "linear" schedule).
            n_final_gumbel_samples: Samples to draw for final trigger selection (if 0, use argmax).
            gd_optimizer: The gradient descent optimizer Torch class to use.
            use_lr_schedule: If True, apply cosine annealing to the learning rate.
            grad_clip_norm: If set, clip gradient norms to this value before each optimizer step.
            init_mode: How to initialize the logit matrix. "from_trigger" sets initial_coeff at
                the initial trigger token positions. "random" samples from N(0, init_noise_scale).
            init_noise_scale: Std of random initialization (used only with init_mode="random").
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        self.num_steps = num_steps
        self.n_grad_samples = n_grad_samples
        self.learning_rate = learning_rate
        self.initial_coeff = initial_coeff
        self.temp_schedule = temp_schedule
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.n_final_gumbel_samples = n_final_gumbel_samples
        self.GDOptimizer = gd_optimizer
        self.use_lr_schedule = use_lr_schedule
        self.grad_clip_norm = grad_clip_norm
        self.init_mode = init_mode
        self.init_noise_scale = init_noise_scale

    def _get_temperature(self, step: int) -> float:
        """Get temperature for the current step based on the configured schedule."""
        if self.temp_schedule == "gradual":
            return self._gradual_temperature(step)
        # Linear annealing
        progress = min(1.0, step / max(1, self.num_steps - 1))
        return self.temp_start + progress * (self.temp_end - self.temp_start)

    def _gradual_temperature(self, step: int) -> float:
        """3-phase schedule (50/25/25): explore -> refine -> discretize."""
        ratio = step / max(1, self.num_steps)
        if ratio < 0.5:
            p = ratio / 0.5
            return 2.5 - 1.5 * p          # 2.5 → 1.0
        elif ratio < 0.75:
            p = (ratio - 0.5) / 0.25
            return 1.0 - 0.5 * p           # 1.0 → 0.5
        else:
            p = (ratio - 0.75) / 0.25
            return 0.5 * (0.01 / 0.5) ** p  # 0.5 → 0.01 (exponential decay)

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
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
        trigger_seq_len = trigger_ids.shape[1]

        # Initialize logit matrix theta (here called `trigger_probs`)
        if self.init_mode == "random":
            trigger_probs: Float[Tensor, "seq_len vocab_size"] = (
                torch.randn(trigger_seq_len, vocab_size, device=device, dtype=self.model.dtype)
                * self.init_noise_scale
            )
        else:  # "from_trigger"
            trigger_probs: Float[Tensor, "seq_len vocab_size"] = torch.zeros(
                trigger_seq_len, vocab_size, device=device, dtype=self.model.dtype,
            )
            for i in range(trigger_seq_len):
                trigger_probs[i, trigger_ids[0, i]] = self.initial_coeff

        # Initialize optimizer and learning rate scheduler
        optimizer = self.GDOptimizer([trigger_probs], lr=self.learning_rate)
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_steps)
            if self.use_lr_schedule
            else torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        )

        best = RunningBest()

        # --- Optimization of `trigger_probs` ---
        pbar = tqdm(range(self.num_steps), desc="GBDA Optimization")

        for step in pbar:
            temperature = self._get_temperature(step)
            optimizer.zero_grad()

            # --- Gradient step ---
            # compute the gradient (wrt `trigger_probs_samples`) w/ Gumbel-softmax sampling
            # (we repeat `trigger_probs_samples` so the gradient computation will draw multiple samples (w/ gumbel-softmax) from the same (optimized) distribution.)
            trigger_probs_samples = trigger_probs.unsqueeze(0).repeat(self.n_grad_samples, 1, 1)
            trigger_grad = self.model.compute_grad_from_tokens(
                candidate_trigger_probs=trigger_probs_samples,  # (n_grad_samples, trigger_seq_len, vocab_size)
                loss_func=self.loss_func,
                do_gumbel_softmax=True,
                gumbel_softmax_temp=temperature,
                normalize_grads=False,
            )  # (n_grad_samples, trigger_seq_len, vocab_size)
            # Average gradients across drawn samples
            avg_grad = trigger_grad.mean(dim=0)  # -> (trigger_seq_len, vocab_size)

            # take grad step:
            trigger_probs.grad = avg_grad
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_([trigger_probs], self.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            # --- Evaluate discrete trigger ---
            with torch.no_grad():
                current_trigger_ids: Float[Tensor, "seq_len"] = trigger_probs.argmax(dim=-1)

                current_loss = self.model.compute_loss_from_tokens(
                    current_trigger_ids.unsqueeze(0),
                    loss_func=self.loss_func,
                ).item()
                current_trigger_str = tokenizer.decode(
                    current_trigger_ids, skip_special_tokens=True
                )

            best.update(loss=current_loss, trigger_ids=current_trigger_ids, trigger_str=current_trigger_str)

            self.log(loss=current_loss, trigger_str=current_trigger_str, best_loss=best.loss, temperature=temperature, lr=scheduler.get_last_lr()[0])

            pbar.set_description(
                f"loss={current_loss:.4f} best={best.loss:.4f} "
                f"T={temperature:.4f} trigger={current_trigger_str[:30]}"
            )

        # --- Final sampling ---
        # Collect candidates: argmax from final theta, best from optimization, and drawn gumbel samples
        candidates = [trigger_probs.argmax(dim=-1)]
        if best.trigger_ids is not None:
            candidates.append(best.trigger_ids)

        for _ in range(self.n_final_gumbel_samples):
            sampled_probs = F.gumbel_softmax(
                trigger_probs.unsqueeze(0), tau=self.temp_end, hard=True, dim=-1,
            )
            candidates.append(sampled_probs.argmax(dim=-1).squeeze(0))

        candidates = torch.stack(candidates, dim=0)
        candidate_losses = self.model.compute_loss_from_tokens(
            candidates, loss_func=self.loss_func,
        )
        final_best_idx = candidate_losses.argmin().item()
        final_best_ids = candidates[final_best_idx]
        final_best_str = tokenizer.decode(final_best_ids, skip_special_tokens=True)
        final_best_loss = candidate_losses[final_best_idx].item()

        result = OptimizerResult(
            best_loss=final_best_loss,
            best_trigger_str=final_best_str,
            best_trigger_ids=final_best_ids,
            losses=best.losses,
            trigger_strs=best.trigger_strs,
            best_trigger_probs=trigger_probs.detach(),
        )

        self.tracker.log({
            "best_loss": result.best_loss,
            "best_trigger_str": result.best_trigger_str,
        })
        self.model.reset_inputs_from_tokens()
        return result
