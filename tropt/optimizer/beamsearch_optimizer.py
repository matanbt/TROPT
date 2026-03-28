import logging
from typing import Annotated, Any, List, Optional

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from tropt.common import Targets, TextTemplates
from tropt.loss import BaseLoss
from tropt.model import (
    BaseModel,
    LMBaseModel,
    LogitsTokenAccessMixin,
    LossTokenAccessMixin,
)
from tropt.model.huggingface.lm import LMHFModel
from tropt.model.model_mixins import LossTextAccessMixin
from tropt.optimizer import BaseOptimizer, OptimizerResult
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)


class BeamSearchOptimizer(BaseOptimizer):
    """
    An LM beam search-based optimizer.
        The general idea is to sample tokens while generating from a util LM, and
        steer the generation towards the desired objective(s) on the target model.

    Combines the implementations of BEAST and AdvDecoding optimizers:
    - BEAST optimizer: https://arxiv.org/abs/2402.15570
    - AdvDecoding optimizer: https://arxiv.org/abs/2410.02163
    """

    model_requirements = (LossTextAccessMixin,)
    # Optimizer also have optional flow for token-level loss access (`use_model_with_token_inputs`)
    # [TODO consider decouple to two different optimizers (`TokenBeamSearchOptimizer`) for clarity (according to the `use_model_with_token_inputs` flows)]

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # attack parameters:
        util_lm: Optional[LMBaseModel] = None,  # if None, use the same as `model`
        util_lm_prefix: Optional[str] = None,
        num_steps: int = 40,
        beam_size: int = 15,
        branching_factor: int = 15,
        top_k: Optional[int] = None,
        temperature: float = 1.0,
        token_constraints: TokenConstraints = TokenConstraints(),
        use_model_with_token_inputs: bool = False,
    ):
        """
        Initializes the BeamSearch Optimizer.

        Args:
            model (HuggingFaceModel): The model to be attacked.
            loss (BaseLoss): The loss function to be optimized.
            seed (int, optional): Random seed for reproducibility.

            util_lm (LMBaseModel, optional): Utility LM for generating candidates.
                If None, internally uses the same as `model` (as in original BEAST paper).
            util_lm_prefix (str, optional): A prefix to seed the util LM for next-token computation of the trigger.
                If None, defaults to the same `templates` provided to `optimize_trigger()`.

            num_steps (int): Number of optimization iterations (L in paper); also represents the length of the crafted trigger. Default: 40
            beam_size (int): Number of beams to maintain (k1 in paper). Default: 15
            branching_factor (int): Number of candidates (=tokens) per beam (k2 in paper). Default: 15
            top_k (int, optional): Optional top-k filtering before multinomial sampling.
                If None, samples from full distribution (as in original BEAST paper).
            temperature (float): Sampling temperature. Default: 1.0 (as in paper)
            token_constraints (TokenConstraints): An object to manage token blacklisting.
            use_model_with_token_inputs (bool): Whether we should compute loss on the target model in token-level. Otherwise, we compute loss in text-level by decoding the candidate triggers and querying the model with the full text (including the decoded trigger). Defaults to the latter (False).
                - For instance, in BEAST the target model and the util-lm are the same.
                - In other attacks, such as AdvDecoding, the util-lm and the target model are different (could even be of different architectures), so we'd set this to False.
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        # define the util LM model (defaults to the attacked model)
        if util_lm is None:
            # shallow copies the model object:
            # (we do this so each model will have its own state, specifically for inputmanagement)
            import copy
            util_lm = copy.copy(model)
        self.util_lm = util_lm
        assert isinstance(self.util_lm, LMBaseModel) and isinstance(
            self.util_lm, LogitsTokenAccessMixin
        ), "BEAST requires util_lm to be LM with token logits access"
        self.util_lm_prefix = util_lm_prefix

        self.num_steps = num_steps
        self.beam_size = beam_size
        self.branching_factor = branching_factor
        self.top_k = top_k
        self.temperature = temperature
        self.token_constraints = token_constraints
        self.use_model_with_token_inputs = use_model_with_token_inputs
        if use_model_with_token_inputs:
            if not isinstance(self.model, LossTokenAccessMixin):
                raise ValueError("use_model_with_token_inputs is True, but the model does not support token-level loss access. Please set use_model_with_token_inputs to False or use a model that supports token-level loss access.")
            if self.util_lm.tokenizer != self.model.tokenizer:
                raise ValueError("use_model_with_token_inputs is True, but util_lm and model have different tokenizers. Please set use_model_with_token_inputs to False or ensure that util_lm and model have the same tokenizer.")

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = None,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:
        """
        Optimize the trigger using BEAST algorithm.

        Args:
            templates (TextTemplates): List of text templates to optimize the trigger against.
            targets (Optional[Targets], optional): Target values for the loss function.

        Implementation notes:
        - We use the auxiliary LM (`util_lm`) to samples candidate tokens for the trigger.
          (Note that in the original BEAST it was the same as the attacked LM; other attack use separate utility LM)
        - Then, we evaluate the candidate triggers on the targeted model (`model`) to compute the losses.
        - This loss evaluation against the target model is usually done in a black-box manner using text-level access (i.e., we query the model with the full text including the decoded candidate triggers), to enable the attack of fully black-box models; however, if util and target model share the same tokenizer, we can compute loss in token-level using the `use_model_with_token_inputs` option.
        """

        self._log_run_config_to_tracker(templates, targets=targets)

        # Prepare inputs for both target model and util LM
        self.util_lm.set_inputs_from_tokens(
            templates=[self.util_lm_prefix] * len(templates) if self.util_lm_prefix is not None else templates,
            targets=targets,
        )
        if self.use_model_with_token_inputs:
            # If model has token-level loss access, we use token-level inputs
            self.model.set_inputs_from_tokens(
                templates=templates,
                targets=targets,
            )
        else:
            self.model.set_inputs_from_texts(templates=templates, targets=targets)

        util_tokenizer = self.util_lm.tokenizer
        util_blacklist_ids = self.token_constraints.get_blacklist_ids(
            util_tokenizer, self.util_lm.vocab_size
        )

        # starts with an empty trigger
        util_trigger_ids = torch.zeros((1, 0), dtype=torch.long, device=self.model.device)

        # Initialize by sampling beam_size diverse starting tokens
        # (BEAST: Algorithm 1 in paper; lines 2-7)
        # Get logits for first token position
        initial_logits = self.util_lm.compute_logits_from_tokens(
            util_trigger_ids,  # empty trigger
            return_after_trigger_logits_only=True
        ).squeeze(1)  # (1, vocab_size)

        # Sample beam_size different initial tokens
        initial_probs = torch.softmax(initial_logits / self.temperature, dim=-1)
        initial_probs[:, util_blacklist_ids] = 0
        initial_probs = initial_probs / initial_probs.sum(dim=-1, keepdim=True)

        # Sample beam_size different tokens using multinomial sampling (without replacement)
        initial_tokens = self._sample_multinomial(
            initial_probs,
            return_tokens=self.beam_size,
            top_k=self.top_k
        )  # (1, beam_size)

        # Initialize beam with different starting tokens
        beam_trigger_ids = initial_tokens.t()  # (beam_size, 1)

        loss_per_step = []
        trigger_strings = []
        trigger_tensors = []

        # Iterate for num_steps steps (BEAST: Algorithm 1 in paper; lines 8-23)
        # We already have 1 token, so iterate num_steps - 1 times
        pbar = tqdm(range(self.num_steps - 1), desc="Beam Search optimization")

        for step in pbar:
            # 1. Get logits for the next trigger token  (adv[-1]'s)
            next_token_logits = self.util_lm.compute_logits_from_tokens(
                beam_trigger_ids, return_after_trigger_logits_only=True
            )  # (beam, 1, vocab_size)
            next_token_logits = next_token_logits.squeeze(1)  # (beam, vocab_size)

            # 2. Sample candidate next trigger tokens using multinomial sampling
            probs = torch.softmax(next_token_logits / self.temperature, dim=-1)
            probs[:, util_blacklist_ids] = 0  # Block out disallowed tokens [TODO should we block before the softmax w/ -inf?]
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize

            # Sample branching_factor candidates per beam using multinomial sampling (without replacement)
            candidate_next_tokens = self._sample_multinomial(
                probs,
                return_tokens=self.branching_factor,
                top_k=self.top_k
            )  # (beam, branching_factor)

            # 3. Expand beam: create beam_size x branching_factor candidate sequences
            #    Append each of branching_factor tokens to each of beam_size beams
            repeated_triggers = beam_trigger_ids.repeat_interleave(
                self.branching_factor, dim=0
            )  # (beam, len) -> (beam * branching_factor, len)
            candidate_next_tokens = candidate_next_tokens.reshape(
                -1, 1
            )  # (beam, branching_factor) -> (beam * branching_factor, 1)
            candidate_triggers = torch.cat(
                [repeated_triggers, candidate_next_tokens], dim=-1
            )  # append candidate tokens -> (beam * branching_factor, len+1)

            # 5. Compute losses for all beam x branching_factor candidate triggers
            if not self.use_model_with_token_inputs:
                # Model computes loss in text-level
                candidate_triggers_text = [self.util_lm.tokenizer.decode(trigger, skip_special_tokens=True) for trigger in candidate_triggers]
                losses = self.model.compute_loss_from_texts(
                    candidate_triggers_text,
                    loss_func=self.loss_func
                )
            else:
                # In that case, models share tokenizer, we can directly compute loss without decoding
                assert self.util_lm.tokenizer == self.model.tokenizer, "Models must share tokenizer to compute loss in token-level."
                model_candidate_triggers = candidate_triggers
                losses = self.model.compute_loss_from_tokens(
                    model_candidate_triggers, loss_func=self.loss_func
                )

            # 6. Select top beam_size candidates with lowest loss, keep their trigger ids
            top_losses, top_indices = torch.topk(losses, self.beam_size, largest=False)

            beam_trigger_ids = candidate_triggers[top_indices]
            current_loss = top_losses[0].item()

            # Track best trigger
            best_trigger_ids = beam_trigger_ids[0]
            trigger_str = util_tokenizer.decode(best_trigger_ids, skip_special_tokens=True)
            trigger_strings.append(trigger_str)
            trigger_tensors.append(best_trigger_ids)
            loss_per_step.append(current_loss)
            self.tracker.log({"loss": current_loss, "trigger_str": trigger_str, **self.loss_func.get_loss_log_dict(), **self.model.get_usage_stats()})

            pbar.set_description(f"loss={current_loss: .4f}, trigger={trigger_str}")

        min_loss_index = np.argmin(loss_per_step)

        result = OptimizerResult(
            best_loss=loss_per_step[min_loss_index],
            losses=loss_per_step,
            best_trigger_str=trigger_strings[min_loss_index],
            best_trigger_ids=trigger_tensors[min_loss_index].squeeze(),
            trigger_strs=trigger_strings,
        )
        self.tracker.log({"best_loss": result.best_loss, "best_trigger_str": result.best_trigger_str})
        self.model.reset_inputs_from_tokens()
        self.util_lm.reset_inputs_from_tokens()
        return result

    @staticmethod
    @torch.no_grad()
    def _sample_multinomial(probs, return_tokens=0, top_k=None):
        """
        Sample tokens from probability distribution using multinomial sampling.
        Optionally filter to top-k tokens before sampling.

        Params:
            probs: probability distribution (should sum to 1)
            return_tokens: number of tokens to sample
            top_k: if provided, only sample from top-k highest probability tokens
        Return:
            sampled tokens: (batch_size, return_tokens)
        """
        # If probs do not sum to (roughly) 1, apply softmax
        if not torch.allclose(
            probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-3
        ):
            probs = torch.softmax(probs, dim=-1)

        if top_k is not None:
            # Filter to top-k tokens
            top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

            # Sample from top-k distribution
            sampled_indices = torch.multinomial(
                top_k_probs,
                num_samples=min(return_tokens, top_k),
                replacement=False
            )

            # Map back to original vocabulary indices
            next_tokens = torch.gather(top_k_indices, -1, sampled_indices)
        else:
            # Sample directly from full distribution
            next_tokens = torch.multinomial(
                probs,
                num_samples=return_tokens,
                replacement=False
            )

        return next_tokens

