import logging
import random
from typing import Callable, List, Optional

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
        # Per-step batch sampling:
        template_batch_size: Optional[int] = None,
        template_batch_sampler: Optional[Callable[[TextTemplates, Optional[Targets]], List[int]]] = None,
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
            template_batch_size: If set, sample this many templates (and their targets) per step instead of
                using all templates.
            template_batch_sampler: If set, called as `template_batch_sampler(templates, targets)` each step to pick the
                template indices for that step (e.g., stratified batches); defaults to uniform sampling of `template_batch_size` templates.
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)
        assert template_batch_sampler is None or template_batch_size is None, (
            "Pass either `template_batch_sampler` or `template_batch_size`, not both."
        )

        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.GDOptimizer = gd_optimizer
        self.template_batch_size = template_batch_size
        if template_batch_sampler is None and template_batch_size is not None:  # default: uniform random batch
            template_batch_sampler = lambda templates, targets: random.sample(range(len(templates)), template_batch_size)  # noqa: E731
        self.template_batch_sampler = template_batch_sampler

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
        model_dtype = trigger_embeds.dtype
        trigger_embeds = trigger_embeds.float()  # stored in high precision for the optimizer
        use_batch_sampling = self.template_batch_sampler is not None and (
            self.template_batch_size is None or self.template_batch_size < len(templates)
        )

        # Initialize the optimizer on the trigger embeddings
        optimizer = self.GDOptimizer([trigger_embeds], lr=self.learning_rate)

        best = RunningBest()

        for step in self.track_steps(range(self.num_steps), desc="Soft Prompt Optimization"):
            optimizer.zero_grad()

            if use_batch_sampling:
                batch_indices = self.template_batch_sampler(templates, targets)
                self.model.set_inputs_from_tokens(
                    templates=[templates[i] for i in batch_indices],
                    targets=targets.select_indices(batch_indices) if targets is not None else None,
                )

            # Compute gradients w.r.t. trigger embeddings
            trigger_grad, curr_loss = self.model.compute_grad_from_embeds(
                loss_func=self.loss_func,
                candidate_trigger_embeds=trigger_embeds.to(model_dtype),
                normalize_grads=False,
                return_loss=True,
            )  # grad: (1, trigger_seq_len, embed_dim); loss: (1,)
            curr_loss = curr_loss.item()

            # Record the embeddings that produced this step's loss (before the update below)
            best.update(
                loss=curr_loss,
                trigger_emb=trigger_embeds.detach().clone().squeeze(0).to(model_dtype),
            )

            # Set gradient on trigger embeddings, and step
            trigger_embeds.grad = trigger_grad.float()
            optimizer.step()

            self.log(loss=curr_loss, lr=optimizer.param_groups[0]["lr"], grad_norm=trigger_grad.norm().item())

        result = best.to_result()

        return result
