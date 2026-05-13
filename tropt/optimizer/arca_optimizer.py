import logging
from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor

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
from tropt.optimizer.utils.retokenization import retokenize_filtering
from tropt.optimizer.utils.running_best import RunningBest
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)


class ARCAOptimizer(BaseOptimizer):
    """Gradient-based cyclic coordinate descent (Jones et al., 2023).

    Each step advances to the next trigger position (cyclically), averages
    gradients over multiple random-token perturbations at that position,
    then evaluates all top-k candidates there.

    Reference: https://arxiv.org/abs/2303.04381
    """

    model_requirements = (LossTokenAccessMixin, GradientTokenAccessMixin)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        num_steps: int = 500,
        n_candidates: int = 512,
        sample_topk: int = 256,
        token_constraints: TokenConstraints = TokenConstraints(),
        use_retokenize: bool = True,
        n_grad_avg: int = 32,
    ):
        """
        Args:
            n_grad_avg: Number of random-token perturbations at the current
                position to average gradients over. Higher values give a more
                robust gradient signal at the cost of more forward/backward passes.
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        self.num_steps = num_steps
        self.n_candidates = n_candidates
        self.sample_topk = sample_topk
        self.token_constraints = token_constraints
        self.use_retokenize = use_retokenize
        self.n_grad_avg = n_grad_avg

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:

        self.model.set_inputs_from_tokens(templates=templates, targets=targets)
        tokenizer = self.model.tokenizer

        trigger_ids: Int[Tensor, "trigger_seq_len"] = tokenizer.encode_trigger(initial_trigger).to(self.model.device)

        _vocab_size = self.model.vocab_size
        blacklist_ids = self.token_constraints.get_blacklist_ids(tokenizer, _vocab_size)
        valid_token_ids = self.token_constraints.get_whitelist_ids(tokenizer, _vocab_size, return_tensor=True).to(self.model.device)

        best = RunningBest()

        # Initial loss
        current_loss = self.model.compute_loss_from_tokens(
            trigger_ids.unsqueeze(0), loss_func=self.loss_func
        ).item()
        self.log(loss=current_loss, trigger_str=initial_trigger)

        for step_idx in self.track_steps(range(self.num_steps)):
            trigger_seq_len = trigger_ids.shape[0]

            # Cyclic position selection
            pos = step_idx % trigger_seq_len

            # Build gradient-averaging variations: place random tokens at `pos`
            grad_triggers = self._get_grad_variations(
                trigger_ids, valid_token_ids, pos
            )

            # Compute and average gradients (across variantions)
            trigger_grad: Float[Tensor, "trigger_seq_len vocab_size"] = (
                self.model.compute_grad_from_tokens(
                    candidate_trigger_ids=grad_triggers,
                    loss_func=self.loss_func,
                    normalize_grads=False,
                ).mean(dim=0)
            )

            # Top-k candidates per position
            trigger_grad *= -1
            trigger_grad[:, blacklist_ids] = float("-inf")
            topk_ids = trigger_grad.topk(self.sample_topk, dim=-1).indices

            # Build candidates: all top-k at the cyclic position
            n_cands = min(self.n_candidates, self.sample_topk)
            candidate_trigger_ids = trigger_ids.repeat(n_cands, 1).clone()
            candidate_trigger_ids[:, pos] = topk_ids[pos, :n_cands]

            # Retokenization filtering
            if self.use_retokenize:
                candidate_trigger_ids = retokenize_filtering(
                    candidate_trigger_ids, tokenizer
                )
                if len(candidate_trigger_ids) == 0:
                    logger.warning("All candidates filtered out, skipping step.")
                    continue

            # Evaluate
            losses = self.model.compute_loss_from_tokens(
                candidate_trigger_ids, loss_func=self.loss_func
            )
            if losses.dim() > 1:
                losses = losses.mean(dim=0)

            current_loss = losses.min().item()
            trigger_ids = candidate_trigger_ids[losses.argmin()]
            trigger_str = tokenizer.decode_trigger(trigger_ids)

            best.update(loss=current_loss, trigger_ids=trigger_ids, trigger_str=trigger_str)
            self.log(loss=current_loss, trigger_str=trigger_str)

        return best.to_result()

    def _get_grad_variations(
        self,
        trigger_ids: Int[Tensor, "trigger_seq_len"],
        valid_token_ids: Int[Tensor, "n_valid"],
        pos: int,
    ) -> Int[Tensor, "n_grad_avg trigger_seq_len"]:
        """Create trigger copies with random tokens at `pos` for gradient averaging."""
        if self.n_grad_avg <= 1:
            return trigger_ids.unsqueeze(0)

        device = trigger_ids.device
        grad_triggers = trigger_ids.repeat(self.n_grad_avg, 1).clone()
        rand_toks = valid_token_ids[
            torch.randint(0, len(valid_token_ids), (self.n_grad_avg,), device=device)
        ]
        grad_triggers[:, pos] = rand_toks
        return grad_triggers
