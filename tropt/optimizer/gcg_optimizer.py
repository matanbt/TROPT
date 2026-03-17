import logging
from typing import Annotated, Any, List, Optional

import torch
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
from tropt.optimizer.utils.retokenization import retokenize_filtering
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)

class GCGOptimizer(BaseOptimizer):
    """
    https://arxiv.org/abs/2307.15043
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
        n_candidates: int = 512,
        sample_topk: int = 256,
        sample_n_replace: int = 1,
        token_constraints: TokenConstraints = TokenConstraints(),
        use_retokenize: bool = True,
    ):
        """
        Implements the Greedy Coordinate Gradient (GCG) optimization algorithm for finding adversarial text triggers.

        Args:
            model (BaseModel): The language model to be attacked.
            loss (BaseLoss): The loss function to be optimized.

            # Attack parameters:
            num_steps (int): Number of optimization steps to perform.
            n_candidates (int): Number of candidate sequences to generate.
            sample_topk (int): Number of top tokens to consider for each position.
            sample_n_replace (int): Number of token positions to update per candidate.
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        # save params:
        self.num_steps = num_steps
        self.n_candidates = n_candidates
        self.sample_topk = sample_topk
        self.sample_n_replace = sample_n_replace
        self.token_constraints = token_constraints
        self.use_retokenize = use_retokenize

    def _sample_ids_from_grad(
        self,
        trigger_ids: Int[Tensor, "trigger_seq_len"],
        trigger_grad: Float[Tensor, "trigger_seq_len vocab_size"],
        blacklist_ids: List[int] = [],
    ) -> Int[Tensor, "n_candidates trigger_seq_len"]:
        """
        Samples `n_candidates` combinations of token ids based on the token gradient.

        Args:
            trigger_ids (Tensor): shape = (n_type, trigger_seq_len)
            The sequence of token ids being optimized.
            trigger_grad (Tensor): shape = (n_type, trigger_seq_len, vocab_size)
            The gradient of the loss with respect to the one-hot token embeddings.

        Returns:
            Tensor: shape = (n_type, n_candidates, trigger_seq_len)
            Sampled token ids for each candidate.
        """
        trigger_seq_len, vocab_size = trigger_grad.shape
        device = trigger_grad.device
        candidate_trigger_ids = trigger_ids.repeat(self.n_candidates, 1).clone()

        trigger_grad[:, blacklist_ids] = float("inf")

        topk_ids: Float[Tensor, "trigger_seq_len sample_topk"] = (
            (-trigger_grad).topk(self.sample_topk, dim=-1).indices
        )

        # Create random indices for each item in the batch and for each candidate.
        sampled_ids_pos = torch.rand(
            self.n_candidates, trigger_seq_len, device=device
        ).argsort(dim=-1)[
            ..., : self.sample_n_replace
        ]  # shape: (n_candidates, sample_n_replace)  # noqa
        # Select the relevant lists of top-k tokens for each candidate and position
        relevant_topk_lists = topk_ids[sampled_ids_pos]
        # Randomly choose one token from each of the top-k lists
        rand_k_indices = torch.randint(
            0,
            self.sample_topk,
            (self.n_candidates, self.sample_n_replace, 1),
            device=device,
        )

        # Gather the selected token ids using the random indices
        sampled_ids_val = torch.gather(
            input=relevant_topk_lists,  # shape: (n_candidates, sample_n_replace, sample_topk)
            dim=-1,
            index=rand_k_indices,  # shape: (n_candidates, sample_n_replace, 1)
        ).squeeze(-1)  # shape: (n_candidates, sample_n_replace)
        # Scatter the sampled token ids in the selected positions, within the trigger (=apply the flips)
        candidate_trigger_ids = candidate_trigger_ids.scatter_(
            dim=-1,  # -> trigger_seq_len dimension
            index=sampled_ids_pos,
            src=sampled_ids_val,
        )

        return candidate_trigger_ids

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        # objective-specific args:
        targets: Optional[Targets] = None,  # depends on the objective
    ) -> OptimizerResult:
        super().optimize_trigger(templates, initial_trigger=initial_trigger, targets=targets)

        # Initialization:
        self.model.set_inputs_from_tokens(templates=templates, targets=targets)
        tokenizer = self.model.tokenizer
        trigger_ids = (
            tokenizer.encode(initial_trigger, add_special_tokens=False, return_tensors="pt")
            .to(self.model.device, torch.int64)
        )
        vocab_size = self.model.vocab_size
        blacklist_ids = self.token_constraints.get_blacklist_ids(tokenizer, vocab_size)

        trigger_ids: Int[Tensor, "trigger_seq_len"] = trigger_ids.squeeze(0)  # take the only trigger
        trigger_str: str = initial_trigger

        loss_per_step = []
        trigger_strings = []
        trigger_ids_per_step = []

        # Compute loss before optimization
        current_loss = self.model.compute_loss_from_tokens(
            trigger_ids.unsqueeze(0), loss_func=self.loss_func
        ).item()
        self.tracker.log({"loss": current_loss, **self.model.get_usage_stats()})

        pbar = tqdm(range(self.num_steps))

        for _ in pbar:
            # Compute the trigger gradient
            trigger_grad: Float[Tensor, "trigger_seq_len vocab_size"] = (
                self.model.compute_grad_from_tokens(
                    candidate_trigger_ids=trigger_ids.unsqueeze(0),
                    loss_func=self.loss_func,
                ).squeeze(0)  # take the only trigger
            )  # shape: (trigger_seq_len, vocab_size)
            # Sample candidate token sequences based on the token gradient
            candidate_trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"] = (
                self._sample_ids_from_grad(
                    trigger_ids=trigger_ids,
                    trigger_grad=trigger_grad,
                    blacklist_ids=blacklist_ids,
                )
            )

            if self.use_retokenize:
                candidate_trigger_ids = retokenize_filtering(
                    candidate_trigger_ids, tokenizer
                )

            # Compute loss on all candidate sequences
            losses = self.model.compute_loss_from_tokens(
                candidate_trigger_ids, loss_func=self.loss_func
            )  # shape: (n_templates, n_candidates)
            current_loss = losses.min().item()
            self.tracker.log({"loss": current_loss,**self.model.get_usage_stats()})
            trigger_ids = candidate_trigger_ids[losses.argmin()]

            # Update the buffer based on the loss
            loss_per_step.append(current_loss)
            trigger_ids_per_step.append(trigger_ids)

            trigger_str = tokenizer.decode(trigger_ids, skip_special_tokens=True)
            trigger_strings.append(trigger_str)

            pbar.set_description(f"loss={current_loss: .4f}, trigger={trigger_str}")

        min_loss_index = loss_per_step.index(min(loss_per_step))
        best_trigger_str = trigger_strings[min_loss_index]

        full_prompt = [t.replace(OPTIMIZED_TRIGGER_PLACEHOLDER, best_trigger_str) for t in templates]

        result = OptimizerResult(
            best_loss=loss_per_step[min_loss_index],
            best_trigger_str=best_trigger_str,
            best_trigger=trigger_ids_per_step[min_loss_index],
            losses=loss_per_step,
            trigger_strs=trigger_strings,
            full_prompt=full_prompt,
        )
        self.tracker.log({"best_loss": result.best_loss, "best_trigger_str": result.best_trigger_str})
        self.model.reset_inputs_from_tokens()
        return result
