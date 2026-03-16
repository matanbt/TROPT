import logging
from typing import List, Optional

import torch
import torch.nn.functional as F
from jaxtyping import Float
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
    GradientEmbedAccessMixin,
)
from tropt.optimizer import BaseOptimizer, OptimizerResult
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)

# TODO re-read, test and validate the implementation below


class SoftPromptOptimizer(BaseOptimizer):
    """
    Optimizing soft prompts
    """

    model_requirements = (GradientEmbedAccessMixin,)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # Soft prompt optimization parameters:
        num_steps: int = 100,
        learning_rate: float = 0.001,
        gd_optimizer: Optional[torch.optim.Optimizer] = torch.optim.Adam,
    ):
        """

        Args:
            model: The target model to attack (must support gradient computation)
            loss: The loss function to optimize
            tracker: Experiment tracker for logging
            seed: Random seed for reproducibility

            num_steps: Number of optimization iterations
            learning_rate: Learning rate for the gradient descent optimizer
            gd_optimizer: The gradient descent optimizer Torch class to use (e.g., Adam, SGD).

        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.GDOptimizer = gd_optimizer

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:
        super().optimize_trigger(templates, initial_trigger=initial_trigger, targets=targets)

        # Initialization
        self.model.set_token_inputs(templates=templates, targets=targets)
        tokenizer = self.model.tokenizer
        trigger_ids = (
            tokenizer.encode(initial_trigger, add_special_tokens=False, return_tensors="pt")
            .to(self.model.device, torch.int64)
        )

        trigger_embeds = self.model._embedding_layer(trigger_ids)  # (1, trigger_seq_len, embd_dim)
        # trigger_embeds.requires_grad_(True)  # TODO ??

        # Initialize Adam optimizer on the logits
        optimizer = self.GDOptimizer([trigger_embeds], lr=self.learning_rate)

        # Tracking
        loss_per_step = []

        pbar = tqdm(range(self.num_steps), desc="Soft Prompt Optimization")

        for step in pbar:
            optimizer.zero_grad()

            # Compute gradients w.r.t. trigger embeddings
            trigger_grad, curr_loss = self.model.compute_grad_from_embeds(
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
        # TODO save to result to pt file?

        # TODO allow inference with this soft prompt?

        self.tracker.log({
            "best_loss": result.best_loss,
            "best_trigger_embeds": result.best_trigger
        })
        self.model.reset_token_inputs()
        return result
