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


class AutoPromptOptimizer(BaseOptimizer):
    """Gradient-based discrete prompt optimization (Shin et al., 2020).

    Each step picks a single random trigger position and evaluates all
    gradient-ranked top-k candidate tokens at that position.

    Reference: https://arxiv.org/abs/2010.15980
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
    ):
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        self.num_steps = num_steps
        self.n_candidates = n_candidates
        self.sample_topk = sample_topk
        self.token_constraints = token_constraints
        self.use_retokenize = use_retokenize

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:

        self.model.set_inputs_from_tokens(templates=templates, targets=targets)
        tokenizer = self.model.tokenizer

        trigger_ids: Int[Tensor, "trigger_seq_len"] = tokenizer.encode_trigger(initial_trigger).to(self.model.device)

        blacklist_ids = self.token_constraints.get_blacklist_ids(tokenizer, self.model.vocab_size)

        best = RunningBest()

        # Initial loss
        current_loss = self.model.compute_loss_from_tokens(
            trigger_ids.unsqueeze(0), loss_func=self.loss_func
        ).item()
        self.log(loss=current_loss, trigger_str=initial_trigger)

        pbar = self.register_tqdm(range(self.num_steps))

        for step_idx in pbar:
            trigger_seq_len = trigger_ids.shape[0]

            # Pick one random position for this step
            pos = torch.randint(
                0, trigger_seq_len, (1,), device=trigger_ids.device
            ).item()
            assert isinstance(pos, int)

            # Compute gradient
            trigger_grad: Float[Tensor, "trigger_seq_len vocab_size"] = (
                self.model.compute_grad_from_tokens(
                    candidate_trigger_ids=trigger_ids.unsqueeze(0),
                    loss_func=self.loss_func,
                    normalize_grads=True,
                ).squeeze(0)
            )

            # Top-k candidates per position
            trigger_grad *= -1
            trigger_grad[:, blacklist_ids] = float("-inf")
            topk_ids = trigger_grad.topk(self.sample_topk, dim=-1).indices

            # Build candidates: all top-k at the single position
            n_cands = min(self.n_candidates, self.sample_topk)
            candidate_trigger_ids = trigger_ids.repeat(n_cands, 1).clone()
            candidate_trigger_ids[:, pos] = topk_ids[pos, :n_cands]

            # Retokenization filtering
            if self.use_retokenize:
                candidate_trigger_ids = retokenize_filtering(
                    candidate_trigger_ids, tokenizer
                )

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
