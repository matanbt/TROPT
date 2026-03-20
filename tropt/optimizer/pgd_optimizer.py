import logging
from typing import List, Optional

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
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)


class PGDOptimizer(BaseOptimizer):
    """
    Projected Gradient Descent (PGD) for LLMs
    Paper: https://arxiv.org/abs/2402.09154

    PGD optimizes continuous probability distributions over tokens with two key projections:
    1. Simplex projection: Ensures probabilities sum to 1
    2. Entropy projection: Controls discreteness using Tsallis entropy (Gini index)

    Key differences from GBDA:
    - Uses simplex projection instead of just softmax normalization
    - Uses entropy projection to control the relaxation error
    - Much more effective at finding discrete adversarial examples
    """

    model_requirements = (LossTokenAccessMixin, GradientTokenAccessMixin)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # PGD-specific parameters:
        num_steps: int = 100,
        batch_size: int = 10,
        learning_rate: float = 0.3,
        target_entropy: float = 0.5,  # Gini index target (0 = uniform, 1 = one-hot)
        n_gumbel_samples: int = 100,
        eps: float = 1e-12,  # Numerical stability
    ):
        """
        Implements the PGD optimization algorithm for LLMs.

        Args:
            model: The target model to attack
            loss: The loss function to optimize
            tracker: Experiment tracker
            seed: Random seed

            num_steps: Number of optimization iterations
            batch_size: Number of Gumbel-softmax samples for final discretization
            learning_rate: Learning rate for Adam optimizer
            target_entropy: Target Tsallis entropy (Gini index), 0=uniform, 1=discrete
            n_gumbel_samples: Number of samples to draw for final trigger selection
            eps: Small constant for numerical stability
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.target_entropy = target_entropy
        self.n_gumbel_samples = n_gumbel_samples
        self.eps = eps

    def _simplex_projection(
        self, s: Float[Tensor, "vocab_size"]
    ) -> Float[Tensor, "vocab_size"]:
        """
        Project onto the probability simplex.
        Solves: argmin_{s'} ||s - s'||^2 s.t. sum(s') = 1 and s' >= 0

        Based on Duchi et al. 2008: "Efficient projections onto the l1-ball"
        """
        vocab_size = s.shape[0]
        device = s.device

        # Sort in descending order
        mu, _ = torch.sort(s, descending=True)

        # Compute cumulative sums
        mu_cumsum = torch.cumsum(mu, dim=0)

        # Find rho (number of non-zero elements in projection)
        indices = torch.arange(1, vocab_size + 1, device=device, dtype=s.dtype)
        condition = mu - (mu_cumsum - 1) / indices > 0
        rho = torch.sum(condition)

        # Compute threshold
        theta = (mu_cumsum[rho - 1] - 1) / rho

        # Project
        return torch.clamp(s - theta, min=0)

    def _tsallis_entropy_projection(
        self,
        s: Float[Tensor, "vocab_size"],
        target_entropy: float,
    ) -> Float[Tensor, "vocab_size"]:
        """
        Project onto the intersection of simplex and Tsallis entropy ball.
        Uses Tsallis entropy with q=2 (Gini index).

        Tsallis entropy: S_q(p) = (1/(q-1)) * (1 - sum(p_i^q))
        For q=2 (Gini index): S_2(p) = 1 - sum(p_i^2)
        """
        # Only project non-zero elements
        non_zero_mask = s > 0
        non_zero_count = non_zero_mask.sum().item()

        if non_zero_count == 0:
            return s

        # Center of the simplex (uniform distribution over non-zero elements)
        center = torch.zeros_like(s)
        center[non_zero_mask] = 1.0 / non_zero_count

        # Radius of the entropy ball
        # Target entropy for Gini index: 1 - target_entropy = sum(p_i^2)
        radius_squared = max(0.0, 1 - target_entropy - (1.0 / non_zero_count))
        radius = torch.sqrt(torch.tensor(radius_squared, device=s.device))

        # Distance from center
        direction = s - center
        distance = torch.linalg.norm(direction)

        # If already within the ball, no projection needed
        if distance <= radius + self.eps:
            return s

        # Project onto the sphere, then back to simplex
        projected = radius / (distance + self.eps) * direction + center
        return self._simplex_projection(projected)

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Targets = None,
    ) -> OptimizerResult:
        super().optimize_trigger(templates, initial_trigger=initial_trigger, targets=targets)

        # Initialization
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
        trigger_seq_len = trigger_ids_init.shape[0]

        # Initialize probability distribution
        # Start from one-hot encoding of initial trigger
        # Shape: (trigger_seq_len, vocab_size)
        trigger_probs = F.one_hot(trigger_ids_init, num_classes=vocab_size).to(
            dtype=dtype, device=device
        )

        trigger_probs.requires_grad_(True)

        # Initialize Adam optimizer
        optimizer = torch.optim.Adam([trigger_probs], lr=self.learning_rate)

        # Tracking
        loss_per_step = []
        trigger_strings = []
        trigger_ids_per_step = []

        pbar = tqdm(range(self.num_steps), desc="PGD Optimization")

        for step in pbar:
            optimizer.zero_grad()

            # Expand probabilities for batch processing
            # Shape: (batch_size, trigger_seq_len, vocab_size)
            probs_batch = trigger_probs.unsqueeze(0).repeat(self.batch_size, 1, 1)

            # Compute gradients
            # Note: We don't use Gumbel-softmax during optimization in PGD
            # We work directly with the probability distributions
            trigger_grad = self.model.compute_grad_from_tokens(
                candidate_trigger_probs=probs_batch,
                loss_func=self.loss_func,
                do_gumbel_softmax=False,  # PGD doesn't use Gumbel during optimization
            )

            # Average gradients across batch
            avg_grad = trigger_grad.mean(dim=0)  # (trigger_seq_len, vocab_size)

            # Set gradient manually
            trigger_probs.grad = avg_grad

            # Adam step
            optimizer.step()

            # Apply projections (no_grad because these are projections, not optimization)
            with torch.no_grad():
                # Project each token onto the simplex
                for i in range(trigger_seq_len):
                    trigger_probs[i] = self._simplex_projection(trigger_probs[i])

                # Apply entropy projection
                for i in range(trigger_seq_len):
                    trigger_probs[i] = self._tsallis_entropy_projection(
                        trigger_probs[i], self.target_entropy
                    )

            # Evaluate current discrete trigger
            with torch.no_grad():
                current_trigger_ids = trigger_probs.argmax(dim=-1)

                # Compute loss on discrete tokens
                current_loss = self.model.compute_loss_from_tokens(
                    current_trigger_ids.unsqueeze(0),
                    loss_func=self.loss_func,
                ).item()

                current_trigger_str = tokenizer.decode(
                    current_trigger_ids, skip_special_tokens=True
                )

            # Track
            loss_per_step.append(current_loss)
            trigger_ids_per_step.append(current_trigger_ids.clone())
            trigger_strings.append(current_trigger_str)

            self.tracker.log({
                "loss": current_loss,
                **self.loss_func.get_loss_log_dict(),
                **self.model.get_usage_stats()
            })

            pbar.set_description(
                f"loss={current_loss:.4f}, trigger={current_trigger_str[:30]}"
            )

        # Final sampling: draw multiple samples and select the best
        logger.info(f"Final sampling: drawing {self.n_gumbel_samples} samples...")

        best_loss = float('inf')
        best_trigger_ids = None
        best_trigger_str = None

        with torch.no_grad():
            for _ in range(self.n_gumbel_samples):
                # Sample using argmax with small noise for diversity
                noise = torch.rand_like(trigger_probs) * 0.01
                sampled_ids = (trigger_probs + noise).argmax(dim=-1)

                # Compute loss
                sample_loss = self.model.compute_loss_from_tokens(
                    sampled_ids.unsqueeze(0),
                    loss_func=self.loss_func,
                ).item()

                if sample_loss < best_loss:
                    best_loss = sample_loss
                    best_trigger_ids = sampled_ids.clone()
                    best_trigger_str = tokenizer.decode(
                        sampled_ids, skip_special_tokens=True
                    )

        # If no better sample found, use argmax
        if best_trigger_ids is None:
            best_trigger_ids = trigger_probs.argmax(dim=-1)
            best_trigger_str = tokenizer.decode(
                best_trigger_ids, skip_special_tokens=True
            )
            best_loss = loss_per_step[-1]

        # Construct full prompts
        full_prompt = [
            t.replace(OPTIMIZED_TRIGGER_PLACEHOLDER, best_trigger_str)
            for t in templates
        ]

        result = OptimizerResult(
            best_loss=best_loss,
            best_trigger_str=best_trigger_str,
            best_trigger=best_trigger_ids,
            losses=loss_per_step,
            trigger_strs=trigger_strings,
            full_prompt=full_prompt,
        )

        self.tracker.log({
            "best_loss": result.best_loss,
            "best_trigger_str": result.best_trigger_str
        })
        self.model.reset_inputs_from_tokens()
        return result
