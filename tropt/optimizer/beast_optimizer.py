import logging
from typing import Any, List, Optional

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from tropt.optimizer.base import BaseOptimizer, OptimizerResult
from tropt.optimizer.utils.beast_utils import sample_top_p
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.loss.base import BaseLoss
from tropt.models.base import (
    BaseModel,
    LMBaseModel,
    LogitsTokenAccessMixin,
    LossTokenAccessMixin,
    TargetsDict,
)
from tropt.models.huggingface.lm import LMHFModel
from tropt.tracker.base import BaseTracker

logger = logging.getLogger(__name__)

# TODO need to verify BEAST implementation & it's alignment with AdvDecoding

class BEASTOptimizer(BaseOptimizer):
    """
    BEAST optimizer
    https://arxiv.org/abs/2402.15570

    Using a util LM on a target encoder model, this optimizer is effectively AdvDecoding's optimizer
    https://arxiv.org/abs/2410.02163
    """

    model_requirements = (LossTokenAccessMixin,)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # attack parameters:
        util_lm: LMBaseModel = None,  # if None, use the same as `model`
        num_steps: int = 40,
        beam_size: int = 15,
        branching_factor: int = 15,
        top_p: float = 0.25,  # TODO what's common in models for fluent text?
        temperature: float = 1.0,
        token_constraints: TokenConstraints = TokenConstraints(),
    ):
        """
        Initializes the BEAST Optimizer.

        Args:
            model (HuggingFaceModel): The model to be attacked.
            loss (BaseLoss): The loss function to be optimized.
            seed (int, optional): Random seed for reproducibility.

            num_steps (int): Number of optimization iterations.
            beam_size (int): Beam size for beam search.
            branching_factor (int): Number of candidate tokens to sample per beam at each step.
            top_p (float): Top-p sampling threshold for candidate token selection.
            temperature (float): Sampling temperature for candidate token selection.
            token_constraints (TokenConstraints): An object to manage token blacklisting.
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        # define the util LM model (defaults to the attacked model)
        self.util_lm = util_lm if util_lm is not None else model
        assert isinstance(self.util_lm, LMBaseModel) and isinstance(
            self.util_lm, LogitsTokenAccessMixin
        ), "BEAST requires util_lm to be LM with token logits access"

        self.num_steps = num_steps
        self.beam_size = beam_size
        self.branching_factor = branching_factor
        self.top_p = top_p
        self.temperature = temperature
        self.token_constraints = token_constraints

    def optimize_trigger(
        self,
        texts: List[str],  # of length n_messages
        targets: TargetsDict = None,
        initial_trigger: str = "",
    ) -> OptimizerResult:

        if initial_trigger != "":
            logger.warning("BEAST optimizer does not support non-empty initial triggers; ignoring it.")
        initial_trigger = ""

        inputs, _ = self.model.prepare_token_inputs(
            texts=texts,
            targets=targets,
            initial_trigger="",  # BEAST starts with an empty trigger
        )

        util_inputs, util_trigger_ids = self.util_lm.prepare_token_inputs(
            texts=texts,
            targets=targets,
            initial_trigger="",  # BEAST starts with an empty trigger
        )
        util_tokenizer = util_inputs.tokenizer
        util_blacklist_ids = self.token_constraints.get_blacklist_ids(
            util_tokenizer, util_inputs.vocab_size
        )

        # BEAST works token-by-token, so we start with an empty trigger
        beam_trigger_ids: Float[Tensor, "beam 0"] = torch.cat(
            [util_trigger_ids.clone() for _ in range(self.beam_size)], dim=0
        )

        loss_per_step = []
        trigger_strings = []
        trigger_tensors = []

        pbar = tqdm(range(self.num_steps))

        for step in pbar:
            # 1. Get logits for the next trigger token (adv[-1]'s)
            next_token_logits = self.util_lm.compute_logits_from_tokens(
                beam_trigger_ids, util_inputs, return_after_trigger_logits_only=True
            ) # (beam 1 vocab_size)
            next_token_logits = next_token_logits.squeeze(1)  # (beam, vocab_size)

            # 2. Sample candidate next trigger tokens
            probs = torch.softmax(next_token_logits / self.temperature, dim=-1)
            # block out disallowed tokens
            probs[:, util_blacklist_ids] = 0
            # sample
            candidate_next_tokens = sample_top_p(
                probs, self.top_p, return_tokens=self.branching_factor
            )

            # 3. Create new trigger candidate sequences by appending candidate token to each beam
            #    I.e., expand each beam by branching_factor, for a single depth step.
            repeated_triggers = beam_trigger_ids.repeat_interleave(
                self.branching_factor, dim=0
            )  # (beam, len) -> (beam * branching_factor, len)
            candidate_next_tokens = candidate_next_tokens.reshape(
                -1, 1
            )  # flatten to (beam * branching_factor,)
            candidate_triggers = torch.cat(
                [repeated_triggers, candidate_next_tokens], dim=-1
            )  # append candidate tokens

            # 4. Cast to the targeted model tokenizer
            candidate_triggers, model_candidate_triggers = (
                LMHFModel.cast_to_model_tokenizer(  # TODO move to tokInputs? anyway should be better
                    candidate_triggers,
                    model_from=self.util_lm,
                    model_to=self.model,
                )
            )

            # 5. Compute losses for all candidate triggers
            losses = self.model.compute_loss_from_tokens(
                model_candidate_triggers, inputs, loss_func=self.loss_func
            )

            # 6. Select best candidates (beam search)
            top_losses, top_indices = torch.topk(losses, self.beam_size, largest=False)

            beam_trigger_ids = candidate_triggers[top_indices]
            current_loss = top_losses[0].item()

            best_trigger_ids = beam_trigger_ids[0]
            trigger_str = util_inputs.toks_to_strs(best_trigger_ids)
            trigger_strings.append(trigger_str)
            trigger_tensors.append(best_trigger_ids)
            loss_per_step.append(current_loss)
            self.tracker.log({"loss": current_loss})

            pbar.set_description(f"loss={current_loss: .4f}, trigger={trigger_str}")

        min_loss_index = np.argmin(loss_per_step)

        return OptimizerResult(
            best_loss=loss_per_step[min_loss_index],
            best_trigger_str=trigger_strings[min_loss_index],
            best_trigger=trigger_tensors[min_loss_index].squeeze(),
            trigger_strs=trigger_strings,
        )
