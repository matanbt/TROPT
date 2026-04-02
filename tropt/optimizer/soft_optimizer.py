import logging
from typing import Callable, Optional

import torch

from tropt.common import (
    DEFAULT_INIT_TRIGGER,
    Targets,
    TextTemplates,
)
from tropt.loss import BaseLoss
from tropt.model import (
    BaseModel,
    GradientEmbedAccessMixin,
)
from tropt.optimizer.base import BaseOptimizer, OptimizerResult
from tropt.optimizer.utils.running_best import RunningBest
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)



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
        gd_optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
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

        # Initialization
        self.model.set_inputs_from_tokens(templates=templates, targets=targets)
        tokenizer = self.model.tokenizer
        trigger_ids = tokenizer.encode_trigger(initial_trigger).to(self.model.device)

        trigger_embeds = self.model._embedding_layer(trigger_ids.unsqueeze(0))  # (1, trigger_seq_len, embd_dim)

        # Initialize Adam optimizer on the logits
        optimizer = self.GDOptimizer([trigger_embeds], lr=self.learning_rate)

        best = RunningBest()

        pbar = self.register_tqdm(range(self.num_steps), desc="Soft Prompt Optimization")

        for step in pbar:
            optimizer.zero_grad()

            # Compute gradients w.r.t. trigger embeddings
            trigger_grad, curr_loss = self.model.compute_grad_from_embeds(
                loss_func=self.loss_func,
                candidate_trigger_embeds=trigger_embeds,
                normalize_grads=False,
                return_loss=True,
            )  # Shape: (1, trigger_seq_len, vocab_size)

            # Set gradient on trigger embeddings
            trigger_embeds.grad = trigger_grad

            # Adam step
            optimizer.step()

            best.update(loss=curr_loss, trigger_emb=trigger_embeds.detach().squeeze(0))
            self.log(loss=curr_loss, lr=optimizer.param_groups[0]["lr"], grad_norm=trigger_grad.norm().item())

        result = best.to_result()

        return result
