import logging
from typing import Optional, Type

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
    GradientEmbedAccessMixin,
    LossTokenAccessMixin,
)
from tropt.optimizer import BaseOptimizer, OptimizerResult
from tropt.optimizer.utils.running_best import RunningBest
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)


class PEZOptimizer(BaseOptimizer):
    """
    Optimizer of PEZ: optimizes contiuous trigger (AKA soft trigger),
    but projects it to discrete tokens every optimization step.

    Paper: https://arxiv.org/abs/2302.03668
    Reference implementation: https://github.com/YuxinWenRick/hard-prompts-made-easy/blob/main/optim_utils.py
    """

    model_requirements = (LossTokenAccessMixin, GradientEmbedAccessMixin)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # PEZ-specific parameters:
        num_steps: int = 300,
        learning_rate: float = 0.1,
        weight_decay: float = 0.1,
        gd_optimizer: type = torch.optim.SGD,
    ):
        """
        Args:
            model (BaseModel): The language model to be attacked.
            loss (BaseLoss): The loss function to be optimized.
            tracker (BaseTracker, optional): An optional tracker for logging optimization progress.
            seed (int, optional): Random seed for reproducibility.

            num_steps: Number of optimization iterations.
            learning_rate: Learning rate for the optimizer.
            weight_decay: Weight decay for optimizer.
            gd_optimizer: Torch optimizer to use.
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.GDOptimizer = gd_optimizer

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:
        self._log_run_config_to_tracker(templates, initial_trigger, targets)

        # Initialization
        self.model.set_inputs_from_tokens(templates=templates, targets=targets)
        tokenizer = self.model.tokenizer
        trigger_ids = (
            tokenizer.encode(initial_trigger, add_special_tokens=False, return_tensors="pt")
            .to(self.model.device, torch.int64)
        )

        embedding_matrix = self.model.embedding_matrix  # (vocab_size, embed_dim)

        # Initialize continuous embeddings from the initial trigger tokens
        trigger_embeds = self.model._embedding_layer(trigger_ids)  # (1, trigger_seq_len, embed_dim)

        # Initialize optimizer on continuous embeddings
        optimizer = self.GDOptimizer(
            [trigger_embeds], lr=self.learning_rate, weight_decay=self.weight_decay
        )

        pbar = tqdm(range(self.num_steps), desc="PEZ Optimization")
        best = RunningBest()

        for step in pbar:
            optimizer.zero_grad()

            # Forward projection: project continuous embeddings to nearest vocab tokens
            projected_ids, projected_embeds = self._project_to_vocab(
                trigger_embeds.squeeze(0), embedding_matrix
            )

            # Compute gradient w.r.t. the projected embeddings.
            # (this means grad are only calculates wrt actual vocab tokens, but then the grad is
            # appplied to the continuous embeddings `trigger_embeds`)
            trigger_grad, curr_loss = self.model.compute_grad_from_embeds(
                loss_func=self.loss_func,
                candidate_trigger_embeds=projected_embeds.unsqueeze(0),  # (1, trigger_seq_len, embed_dim)
                normalize_grads=False,
                return_loss=True,
            )  # Shape: (1, trigger_seq_len, embed_dim)

            # Set gradient on the continuous embeddings and step
            trigger_embeds.grad = trigger_grad
            optimizer.step()

            # Decode current discrete trigger for tracking
            current_trigger_str = tokenizer.decode(
                projected_ids, skip_special_tokens=True
            )

            self.log(loss=curr_loss, trigger_str=current_trigger_str)

            pbar.set_description(
                f"loss={curr_loss:.4f}, trigger={current_trigger_str[:30]}"
            )
            best.update(loss=curr_loss, trigger_ids=projected_ids.cpu(), trigger_str=current_trigger_str)

        # Final projection and evaluation on discrete tokens
        final_ids, _ = self._project_to_vocab(
            trigger_embeds.squeeze(0), embedding_matrix
        )
        final_trigger_str = tokenizer.decode(final_ids, skip_special_tokens=True)
        final_loss = self.model.compute_loss_from_tokens(
            final_ids.unsqueeze(0),
            loss_func=self.loss_func,
        ).item()
        best.update(loss=final_loss, trigger_ids=final_ids.cpu(), trigger_str=final_trigger_str)

        self.tracker.log({
            "best_loss": best.loss,
            "best_trigger_str": best.trigger_str
        })
        self.model.reset_inputs_from_tokens()
        return OptimizerResult(
            best_loss=best.loss,
            best_trigger_str=best.trigger_str,
            best_trigger_ids=best.trigger_ids,
            losses=best.losses,
            trigger_strs=best.trigger_strs,
        )

    @torch.no_grad()
    def _project_to_vocab(
        self,
        embeds: Float[Tensor, "trigger_seq_len embed_dim"],
        embedding_matrix: Float[Tensor, "vocab_size embed_dim"],
    ) -> tuple[Int[Tensor, "trigger_seq_len"], Float[Tensor, "trigger_seq_len embed_dim"]]:
        """Project each embedding vector to its nearest neighbor in the vocabulary.
        Uses cosine similarity (normalized dot product) as in the paper.

        Returns:
            Tuple of (projected token IDs, projected embeddings).
        """
        # Normalize both queries and vocabulary embeddings
        embeds_norm = F.normalize(embeds, dim=-1)  # (trigger_seq_len, embed_dim)
        matrix_norm = F.normalize(embedding_matrix, dim=-1)  # (vocab_size, embed_dim)

        # Cosine similarity via dot product of normalized vectors
        # (trigger_seq_len, vocab_size)
        cosim = embeds_norm @ matrix_norm.T
        nearest_ids = cosim.argmax(dim=-1)  # (trigger_seq_len,)
        projected_embeds = embedding_matrix[nearest_ids]  # (trigger_seq_len, embed_dim)
        return nearest_ids, projected_embeds
