import logging
from typing import Any, List, Optional

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from tropt.optimizer.base import BaseOptimizer, OptimizerResult
from tropt.optimizer.utils.retokenization import retokenize_filtering
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker.base import BaseTracker
from tropt.loss.base import BaseLoss
from tropt.models.base import (
    BaseModel,
    LMBaseModel,
    LogitsTokenAccessMixin,
    LossTextAccessMixin,
    TargetsDict,
)

logger = logging.getLogger(__name__)

# TODO add outer-beam search (like BEAST)


class LASLITEOptimizer(BaseOptimizer):
    """
    Implements a variant of the GASLITE optimization algorithm, that relies on a util LM's logits (instead of gradients).
    Originally from the paper: "GASLITEing the Retrieval: Exploring Vulnerabilities in Dense Embedding-based Search"
    (https://arxiv.org/abs/2412.20953)

    """

    model_requirements = (LossTextAccessMixin,)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # attack parameters:
        num_steps: int = 100,
        n_flip: int = 20,
        n_candidates: int = 128,
        token_constraints: TokenConstraints = TokenConstraints(),
        use_retokenize: bool = True,
        util_lm: Optional[LMBaseModel] = None,  # for logits calc
        use_random_logits: bool = False,  # for possible ablation
        flip_pos_method: str = "random",  # "random" or "ordered"
    ):
        """
        Initializes the LASLITE Optimizer.

        Args:
            model (HuggingFaceModel): The model to be attacked.
            loss (BaseLoss): The loss function to be optimized.
            seed (int, optional): Random seed for reproducibility.

            num_steps (int): Number of optimization iterations.
            n_flip (int): Number of token positions to greedily optimize per step.
            n_candidates (int): Number of top candidate tokens to evaluate for each position.

            token_constraints (TokenConstraints): An object to manage token blacklisting.
            use_retokenize (bool): Whether to filter candidates that are not reversible by the tokenizer.

            util_lm (LMHFModel, optional): Utility model for logits calculation. Defaults to `model`.
            use_random_logits (bool): If True, use random logits instead of actual logits from `util_lm` (for ablation).
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        if seed is not None:
            from transformers import set_seed

            set_seed(seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        # save params:
        self.num_steps = num_steps
        self.n_flip = n_flip
        self.n_candidates = n_candidates
        self.token_constraints = token_constraints
        self.use_retokenize = use_retokenize

        self.util_lm = util_lm if util_lm is not None else model
        assert isinstance(self.util_lm, LMBaseModel) and isinstance(
            self.util_lm, LogitsTokenAccessMixin
        ), "LASLITE requires util_lm to be LM with token logits access"

        self.use_random_logits = use_random_logits
        self.flip_pos_method = flip_pos_method

    def optimize_trigger(
        self,
        texts: List[str],
        initial_trigger: Optional[str] = "! " * 20,
        targets: TargetsDict = None,
    ) -> OptimizerResult:
        # Initialization:
        # We prepare inputs for both models. The optimization will be done on `util_trigger_ids`.
        inputs, _ = self.model.prepare_text_inputs(
            texts=texts,
            initial_trigger=initial_trigger,
            targets=targets,
        )
        util_inputs, util_trigger_ids = self.util_lm.prepare_token_inputs(
            texts=texts,
            targets=targets,
            initial_trigger=initial_trigger,
        )
        util_tokenizer = util_inputs.tokenizer
        util_blacklist_ids = self.token_constraints.get_blacklist_ids(
            util_inputs.tokenizer, util_inputs.vocab_size
        )

        util_trigger_ids: Int[Tensor, "trigger_seq_len"] = util_trigger_ids.to(self.util_lm.device).squeeze(0)  # take the only trigger
        trigger_seq_len = len(util_trigger_ids)
        trigger_str = initial_trigger

        loss_per_step = []
        trigger_strings = []
        trigger_ids_per_step = []
        current_loss = float("inf")

        # TODO calc loss before, for logger

        pbar = tqdm(range(self.num_steps), desc="Optimizing with LASLITE...")

        for step in pbar:
            pbar.set_description(
                f"Step {step+1}/{self.num_steps} | loss={current_loss: .4f} | trigger={trigger_str}..."
            )

            # --- Candidate selection step (logit-based) ---
            if self.use_random_logits:
                # Use random logits for ablation study
                trigger_logits = torch.rand(
                    trigger_seq_len, self.util_lm.vocab_size, device=self.util_lm.device
                )
            else:
                # Use util model logits as the candidate selection score
                trigger_logits = self.util_lm.compute_logits_from_tokens(
                    util_trigger_ids.unsqueeze(0),  # (1, trigger_seq_len)
                    util_inputs,
                    return_trigger_logits_only=True,
                    keep_message_dim=False,
                ).squeeze(0)  # (trigger_seq_len, vocab_size)

            # The logits practically replace the gradients in GASLITE
            trigger_grad = trigger_logits

            # Get Top-k Candidates *per position*
            trigger_grad[:,util_blacklist_ids] = float("-inf")

            topk_ids: Float[Tensor, "trigger_seq_len n_candidates"]
            topk_ids = trigger_grad.topk(self.n_candidates, dim=-1).indices

            # --- Greedy coordinate ascent step ---
            current_trigger_ids = util_trigger_ids.clone()

            # Sample `n_flip` unique positions to optimize
            if self.flip_pos_method == "ordered":
                sampled_positions = (
                    step + torch.arange(self.n_flip, device=self.util_lm.device)
                ) % trigger_seq_len
                sampled_positions, _ = sampled_positions.sort()
            else:  # default ("random")
                sampled_positions = torch.randperm(
                    trigger_seq_len, device=self.util_lm.device
                )[: self.n_flip]
                sampled_positions, _ = sampled_positions.sort()

            # Sequentially optimize each position
            for pos in sampled_positions:
                # Get candidate tokens for this position
                all_candidate_tokens = torch.unique(
                    torch.cat(
                        [
                            current_trigger_ids[pos].unsqueeze(0),  # keep the "no flip" option
                            topk_ids[pos],
                        ]
                    )
                )
                n_unique_candidates = len(all_candidate_tokens)

                # Create all candidate triggers by flipping this *single* position
                candidate_triggers = current_trigger_ids.repeat(n_unique_candidates, 1)
                candidate_triggers[:, pos] = all_candidate_tokens

                # (Optional) Retokenize filtering
                if self.use_retokenize:
                    candidate_triggers = retokenize_filtering(
                        candidate_triggers, util_tokenizer
                    )

                candidate_trigger_strs = [
                    util_inputs.toks_to_strs(ids) for ids in candidate_triggers
                ]

                # Compute losses on candidate flips on the target model
                losses = self.model.compute_loss_from_texts(
                    candidate_trigger_strs,
                    inputs,
                    self.loss_func,
                    keep_message_dim=False,
                )  # (n_cands,)

                # Find the best token for this position
                best_candidate_idx = losses.argmin()

                # Update `current_trigger_ids` *in-place* for the next iteration of the greedy (inner) loop
                current_trigger_ids = candidate_triggers[best_candidate_idx].clone()
                current_loss = losses[best_candidate_idx].item()
                # TODO keep multiple candidate (effictively forming an outer beam search)

            # After the inner loop, `current_trigger_ids` is the best trigger for this *entire* step
            util_trigger_ids = current_trigger_ids
            trigger_str = util_inputs.toks_to_strs(util_trigger_ids)

            # Logging:
            self.tracker.log({"loss": current_loss})
            loss_per_step.append(current_loss)
            trigger_strings.append(trigger_str)
            trigger_ids_per_step.append(util_trigger_ids)

        # Return the best trigger found
        best_loss_idx = np.argmin(loss_per_step)
        best_trigger_str = trigger_strings[best_loss_idx]
        best_trigger_ids = trigger_ids_per_step[best_loss_idx]

        return OptimizerResult(
            best_loss=loss_per_step[best_loss_idx],
            best_trigger_str=best_trigger_str,
            best_trigger=best_trigger_ids,
            losses=loss_per_step,
            trigger_strs=trigger_strings,
        )
