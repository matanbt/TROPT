import logging
from typing import List, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from tropt.common import (
    DEFAULT_INIT_TRIGGER,
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
from tropt.optimizer.utils.running_best import RunningBest
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)


class HotFlipOptimizer(BaseOptimizer):
    """
    HotFlip: White-Box Adversarial Examples for Text Classification.
    https://arxiv.org/abs/1712.06751

    Uses first-order Taylor approximation of the loss to greedily select
    token substitutions. Each flip is chosen as the (position, token) pair
    that maximally decreases the estimated loss, without requiring a forward
    pass for candidate evaluation. We implement the greedy variant introduced in the paper.
    """

    model_requirements = (LossTokenAccessMixin, GradientTokenAccessMixin)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # attack parameters:
        num_steps: int = 500,
        token_constraints: TokenConstraints = TokenConstraints(),
        use_retokenize: bool = True,
    ):
        """
        Args:
            num_steps: Number of optimization steps (gradient is recomputed each step).
            token_constraints: Token blacklist constraints. Was not originally included in the paper.
            use_retokenize: Retokenize after flipping for decode/encode consistency. Was not originally included in the paper.
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        self.num_steps = num_steps
        self.token_constraints = token_constraints
        self.use_retokenize = use_retokenize

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
        trigger_ids = tokenizer.encode(
            initial_trigger, add_special_tokens=False, return_tensors="pt"
        ).to(self.model.device, torch.int64)
        vocab_size = self.model.vocab_size
        blacklist_ids = self.token_constraints.get_blacklist_ids(tokenizer, vocab_size)

        trigger_ids: Int[Tensor, "trigger_seq_len"] = trigger_ids.squeeze(0)
        best = RunningBest()

        # Compute loss before optimization
        current_loss = self.model.compute_loss_from_tokens(
            trigger_ids.unsqueeze(0), loss_func=self.loss_func
        ).item()
        self.log(loss=current_loss, trigger_str=initial_trigger)

        pbar = tqdm(range(self.num_steps))

        for _ in pbar:
            # Step 1: Compute gradient wrt trigger
            trigger_grad: Float[Tensor, "trigger_seq_len vocab_size"] = (
                self.model.compute_grad_from_tokens(
                    candidate_trigger_ids=trigger_ids.unsqueeze(0),
                    loss_func=self.loss_func,
                ).squeeze(0)
            )

            # Step 2: Flip token(s) via first-order approximation
            trigger_ids = self._apply_best_flip(trigger_ids, trigger_grad, blacklist_ids)

            # Step 3: Retokenize for decode/encode consistency
            if self.use_retokenize:
                trigger_ids = retokenize_transform(trigger_ids, tokenizer)

            # Step 4: Evaluate actual loss
            current_loss = self.model.compute_loss_from_tokens(
                trigger_ids.unsqueeze(0), loss_func=self.loss_func
            ).item()
            trigger_str = tokenizer.decode(trigger_ids, skip_special_tokens=True)
            self.log(loss=current_loss, trigger_str=trigger_str)
            best.update(
                loss=current_loss, trigger_ids=trigger_ids, trigger_str=trigger_str
            )
            pbar.set_description(f"loss={current_loss: .4f}, trigger={trigger_str}")

        result = OptimizerResult(
            best_loss=best.loss,
            best_trigger_str=best.trigger_str,
            best_trigger_ids=best.trigger_ids,
            losses=best.losses,
            trigger_strs=best.trigger_strs,
        )
        self.tracker.log(
            {"best_loss": result.best_loss, "best_trigger_str": result.best_trigger_str}
        )
        self.model.reset_inputs_from_tokens()
        return result

    def _apply_best_flip(
        self,
        trigger_ids: Int[Tensor, "trigger_seq_len"],
        trigger_grad: Float[Tensor, "trigger_seq_len vocab_size"],
        blacklist_ids: List[int],
    ) -> Int[Tensor, "trigger_seq_len"]:
        """Select and apply token flip(s) using the first-order Taylor approximation.

        The estimated loss change from flipping position i (token a_i -> b) is:
            

        """
        trigger_seq_len = trigger_ids.shape[0]
        device = trigger_ids.device

        # Compute the flips' derivatives for every (position, replacement_token) pair
        # Implementer note: this substraction is not really required, as it doesn't affect the candidate ranking; 
        #                   indeed, later adaptations of HotFlip (such as GCG) omit this substraction.
        current_grad = trigger_grad[torch.arange(trigger_seq_len, device=device), trigger_ids]
        delta = trigger_grad - current_grad.unsqueeze(1)  # (trigger_seq_len, vocab_size)
        # Equivalently, delta[i, b] = grad[i, b] - grad[i, a_i]

        # Mask out blacklisted and current (no-op) tokens
        delta[:, blacklist_ids] = float("inf")
        delta[torch.arange(trigger_seq_len, device=device), trigger_ids] = float("inf")

        # Best replacement token and score for each position
        best_delta_per_pos, best_token_per_pos = delta.min(dim=1)

        # Greedy: single best flip
        best_pos = best_delta_per_pos.argmin().item()
        new_trigger_ids = trigger_ids.clone()
        new_trigger_ids[best_pos] = best_token_per_pos[best_pos]
        return new_trigger_ids
