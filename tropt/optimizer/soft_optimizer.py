import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from tropt.common import DEFAULT_INIT_TRIGGER, OPTIMIZED_TRIGGER_PLACEHOLDER, Targets
from tropt.loss.base import BaseLoss
from tropt.models import (
    BaseModel,
    GradientEmbedAccessMixin,
    TokenInputsManager,
)
from tropt.optimizer.base import BaseOptimizer, OptimizerResult
from tropt.tracker.base import BaseTracker

logger = logging.getLogger(__name__)

# TODO re-read, test and validate the implementation below

class SoftPromptOptimizer(BaseOptimizer):
    """
    Optimizing soft prompts
    """

    model_requirements = (GradientEmbedAccessMixin,)  # TODO huggingfacemodel

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # Soft prompt optimization parameters:
        num_steps: int = 100,
        batch_size: int = 10,
        learning_rate: float = 2e-3,
    ):
        """

        Args:
            model: The target model to attack (must support gradient computation)
            loss: The loss function to optimize
            tracker: Experiment tracker for logging
            seed: Random seed for reproducibility

            num_steps: Number of optimization iterations

        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        self.num_steps = num_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def optimize_trigger(
        self,
        texts: List[str],
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Targets = None,
    ) -> OptimizerResult:
        # Initialization
        inputs: TokenInputsManager
        trigger_ids: Int[Tensor, "1 trigger_seq_len"]
        inputs, trigger_ids = self.model.prepare_token_inputs(
            texts=texts,
            initial_trigger=initial_trigger,
            targets=targets,
        )

        # TODO can also init with random trigger_ids / trigger_embeds?
        trigger_embeds = inputs.embed_func(trigger_ids)  # (1, trigger_seq_len, embd_dim)
        trigger_embeds.requires_grad_(True)  # TODO ??

        # Initialize Adam optimizer on the logits
        optimizer = torch.optim.Adam([trigger_embeds], lr=self.learning_rate)

        # Tracking
        loss_per_step = []

        pbar = tqdm(range(self.num_steps), desc="Soft Prompt Optimization")

        for step in pbar:
            optimizer.zero_grad()

            # Compute gradients w.r.t. trigger embeddings
            trigger_grad, curr_loss = self.model.compute_grad_from_embeds(
                inputs=inputs,
                loss_func=self.loss_func,
                candidate_trigger_embeds=trigger_embeds,
                return_loss=True,
            )  # Shape: (1, trigger_seq_len, vocab_size)

            # Set gradient on trigger embeddings
            trigger_embeds.grad = trigger_grad

            # Adam step
            optimizer.step()

            # Track
            loss_per_step.append(curr_loss)

            self.tracker.log({
                "loss": curr_loss,
                **self.model.get_usage_stats()
            })

            pbar.set_description(
                f"loss={curr_loss:.4f}, trigger={trigger_embeds[0, :2, :2].detach().cpu().tolist()}..."
            )

        result = OptimizerResult(
            best_loss=min(loss_per_step),
            best_trigger=trigger_embeds.detach().tolist(),
            losses=loss_per_step,
        )

        self.tracker.log({
            "best_loss": result.best_loss,
            "best_trigger_embeds": result.best_trigger
        })

        return result
