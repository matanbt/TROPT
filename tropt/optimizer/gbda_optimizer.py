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
    Texts,
)
from tropt.loss.base import BaseLoss
from tropt.models import (
    BaseModel,
    GradientTokenAccessMixin,
    LossTokenAccessMixin,
)
from tropt.optimizer.base import BaseOptimizer, OptimizerResult
from tropt.tracker.base import BaseTracker

logger = logging.getLogger(__name__)

# TODO re-read, test and validate the implementation below

class GBDAOptimizer(BaseOptimizer):
    """
    Gradient-Based Distributional Attack (GBDA)
    Paper: https://arxiv.org/abs/2104.13733

    GBDA optimizes a distribution over adversarial triggers rather than a single
    discrete trigger. It parameterizes the distribution with a continuous matrix Θ
    (logits) and uses Gumbel-softmax to enable gradient-based optimization.

    Key differences from GCG:
    - Optimizes continuous logits Θ rather than discrete tokens
    - Uses Gumbel-softmax for differentiable sampling
    - Employs Adam optimizer instead of greedy coordinate descent
    - Anneals temperature from high to low during optimization
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
        batch_size: int = 10,
        learning_rate: float = 0.3,
        initial_coeff: float = 15.0,
        temp_start: float = 1.0,
        temp_end: float = 0.001,
        n_gumbel_samples: int = 100,
    ):
        """
        Implements the GBDA optimization algorithm.

        Args:
            model: The target model to attack (must support gradient computation)
            loss: The loss function to optimize
            tracker: Experiment tracker for logging
            seed: Random seed for reproducibility

            num_steps: Number of optimization iterations
            batch_size: Number of Gumbel-softmax samples per gradient estimation
            learning_rate: Learning rate for Adam optimizer
            initial_coeff: Initial logit value at original token positions (paper uses 12-15)
            temp_start: Starting temperature for Gumbel-softmax annealing (paper uses 1.0)
            temp_end: Ending temperature for Gumbel-softmax annealing (paper uses 0.001)
            n_gumbel_samples: Number of samples to draw for final trigger selection
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.initial_coeff = initial_coeff
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.n_gumbel_samples = n_gumbel_samples

    def _get_temperature(self, step: int) -> float:
        """
        Compute annealed temperature for current step.
        Linear annealing from temp_start to temp_end over num_steps.
        """
        progress = min(1.0, step / self.num_steps)
        return self.temp_start - progress * (self.temp_start - self.temp_end)

    def optimize_trigger(
        self,
        texts: Texts,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Targets = None,
    ) -> OptimizerResult:
        super().optimize_trigger(texts, initial_trigger=initial_trigger, targets=targets)

        # Initialization
        self.model.set_token_inputs(texts=texts, targets=targets)
        tokenizer = self.model.tokenizer
        trigger_ids: Int[Tensor, "1, trigger_seq_len"] = (
            tokenizer.encode(initial_trigger, add_special_tokens=False, return_tensors="pt")
            .to(self.model.device, torch.int64)
        )
        vocab_size = self.model.vocab_size
        device = self.model.device

        trigger_ids_init = trigger_ids.squeeze(0)  # (trigger_seq_len,)
        trigger_seq_len = trigger_ids_init.shape[0]

        # Initialize logits matrix Θ
        # Shape: (trigger_seq_len, vocab_size)
        trigger_probs = torch.zeros(
            trigger_seq_len, vocab_size,
            device=device,
            dtype=self.model.dtype
        )

        # Initialize at original token positions with initial_coeff
        # All other positions start at 0 (uniform after softmax)
        for i in range(trigger_seq_len):
            trigger_probs[i, trigger_ids_init[i]] = self.initial_coeff

        trigger_probs.requires_grad_(True)  # TODO ??

        # Initialize Adam optimizer on the logits
        optimizer = torch.optim.Adam([trigger_probs], lr=self.learning_rate)

        # Tracking
        loss_per_step = []
        trigger_strings = []
        trigger_ids_per_step = []

        pbar = tqdm(range(self.num_steps), desc="GBDA Optimization")

        for step in pbar:
            optimizer.zero_grad()

            # Get current temperature for this step
            temperature = self._get_temperature(step)

            # Expand logits for batch sampling
            # Shape: (batch_size, trigger_seq_len, vocab_size)
            logits_batch = trigger_probs.unsqueeze(0).repeat(self.batch_size, 1, 1)
            # TODO why reapting here?

            # Compute gradients using Gumbel-softmax
            # Pass logits directly - the backend will apply Gumbel-softmax when do_gumbel_softmax=True
            trigger_grad = self.model.compute_grad_from_tokens(
                candidate_trigger_probs=logits_batch,
                loss_func=self.loss_func,
                do_gumbel_softmax=True,
                gumbel_softmax_temp=temperature,
            )  # Shape: (batch_size, trigger_seq_len, vocab_size)

            # Average gradients across batch
            avg_grad = trigger_grad.mean(dim=0)  # (trigger_seq_len, vocab_size)

            # Set gradient on trigger_probs manually
            trigger_probs.grad = avg_grad

            # Adam step
            optimizer.step()

            # Compute current loss for tracking
            # Use soft probabilities (no Gumbel noise) for evaluation
            with torch.no_grad():
                current_probs = F.softmax(trigger_probs, dim=-1).unsqueeze(0)
                current_trigger_ids = current_probs.argmax(dim=-1).squeeze(0)

                # Evaluate loss on discrete tokens
                current_loss = self.model.compute_loss_from_tokens(
                    current_trigger_ids.unsqueeze(0),
                    loss_func=self.loss_func
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
                "temperature": temperature,
                **self.model.get_usage_stats()
            })

            pbar.set_description(
                f"loss={current_loss:.4f}, T={temperature:.4f}, trigger={current_trigger_str[:30]}"
            )

        # Final sampling: draw multiple samples from the optimized distribution
        # and select the one with the lowest loss
        logger.info(f"Final sampling: drawing {self.n_gumbel_samples} samples from optimized distribution...")

        best_loss = float('inf')
        best_trigger_ids = None
        best_trigger_str = None

        with torch.no_grad():
            for sample_idx in range(self.n_gumbel_samples):
                # Sample using Gumbel-softmax with hard=True to get discrete tokens
                sampled_probs = F.gumbel_softmax(
                    trigger_probs.unsqueeze(0),
                    tau=self.temp_end,  # Use final temperature
                    hard=True,
                    dim=-1,
                )
                sampled_ids = sampled_probs.argmax(dim=-1).squeeze(0)

                # Compute loss for this sample
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

        # If no better sample found (shouldn't happen), use argmax
        if best_trigger_ids is None:
            best_trigger_ids = F.softmax(trigger_probs, dim=-1).argmax(dim=-1)
            best_trigger_str = tokenizer.decode(
                best_trigger_ids, skip_special_tokens=True
            )
            best_loss = loss_per_step[-1]

        # Construct full prompts
        full_prompt = [
            t.replace(OPTIMIZED_TRIGGER_PLACEHOLDER, best_trigger_str)
            for t in texts
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
        self.model.reset_token_inputs()
        return result
