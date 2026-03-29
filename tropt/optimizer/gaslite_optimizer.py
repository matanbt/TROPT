import logging
from typing import Optional

import torch
from jaxtyping import Float
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
from tropt.optimizer.utils.running_best import RunningBest
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)


class GASLITEOptimizer(BaseOptimizer):
    """
    Implements the GASLITE optimization algorithm (Algorithm 1) from the paper:
    "GASLITEing the Retrieval: Exploring Vulnerabilities in Dense Embedding-based Search"
    (https://arxiv.org/abs/2412.20953)

    """

    model_requirements = (LossTokenAccessMixin, GradientTokenAccessMixin)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # attack parameters:
        num_steps: int = 100,
        n_grad: int = 50,
        n_flip: int = 20,
        n_candidates: int = 128,
        token_constraints: TokenConstraints = TokenConstraints(),
        use_retokenize: bool = True,
        use_random_gradient: bool = False,
        **kwargs
    ):
        """
        Initializes the GASLITE Optimizer.

        Args:
            model (HuggingFaceModel): The model to be attacked.
            loss (BaseLoss): The loss function to be optimized.
            seed (int, optional): Random seed for reproducibility.

            num_steps (int): Number of optimization iterations.
            n_grad (int): Number of random flips for gradient averaging.
                          Set to 1 to disable averaging.
            n_flip (int): Number of token positions to greedily optimize per step.
            n_candidates (int): Number of top candidate tokens to evaluate for each position.

            token_constraints (TokenConstraints): An object to manage token blacklisting.
            use_retokenize (bool): Whether to filter candidates that are not reversible by the tokenizer.
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        # save params:
        self.num_steps = num_steps
        self.n_grad = n_grad
        self.n_flip = n_flip
        self.n_candidates = n_candidates
        self.token_constraints = token_constraints
        self.use_retokenize = use_retokenize
        self.use_random_gradient = use_random_gradient

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:
        self._log_run_config_to_tracker(templates, initial_trigger, targets)

        # Initialization:
        self.model.set_inputs_from_tokens(templates=templates, targets=targets)
        tokenizer = self.model.tokenizer
        trigger_ids = (
            tokenizer.encode(initial_trigger, add_special_tokens=False, return_tensors="pt")
            .to(self.model.device, torch.int64)
        ).squeeze(0)  # shape: (trigger_seq_len,)

        vocab_size = self.model.vocab_size
        blacklist_ids = self.token_constraints.get_blacklist_ids(tokenizer, vocab_size)
        token_mask = torch.ones(vocab_size, device=self.model.device, dtype=torch.bool)
        if blacklist_ids:
            token_mask[blacklist_ids] = False
        valid_token_ids = token_mask.nonzero(as_tuple=False).squeeze(-1)

        trigger_ids: Float[Tensor, "trigger_seq_len"] = trigger_ids.to(self.model.device)
        trigger_seq_len = len(trigger_ids)
        trigger_str = initial_trigger
        best = RunningBest()

        # Calculate initial loss for logging
        current_loss = self.model.compute_loss_from_tokens(
            trigger_ids.unsqueeze(0),
            self.loss_func,
        ).item()
        self.log(loss=current_loss, trigger_str=trigger_str)

        pbar = tqdm(range(self.num_steps), desc="Optimizing with GASLITE...")

        for step in pbar:
            pbar.set_description(
                f"Step {step+1}/{self.num_steps} | loss={current_loss: .4f} | trigger={trigger_str}..."
            )

            # --- (I) Gradient and candidate selection step ---
            if self.use_random_gradient:
                # Replace model gradient with random values
                trigger_grad = torch.randn(
                    (trigger_seq_len, vocab_size), device=self.model.device
                )
            else:
                # Compute grad over a list of `n_grad` triggers one-flip away from the current
                trigger_vars = self._get_trigger_variations(trigger_ids, valid_token_ids)
                grads = self.model.compute_grad_from_tokens(
                    candidate_trigger_ids=trigger_vars,
                    loss_func=self.loss_func,
                    normalize_grads=True,
                )  # (n_trigger_vars, trigger_seq_len, vocab_size)

                # Average the gradients to get the final approximation
                trigger_grad = grads.mean(dim=0)
                trigger_grad: Float[Tensor, "trigger_seq_len vocab_size"]
                trigger_grad *= -1  # we want to minimize the loss

            # Get Top-k Candidates *per position*
            trigger_grad[:, blacklist_ids] = float("-inf")
            topk_ids: Float[Tensor, "trigger_seq_len n_candidates"]
            topk_ids = trigger_grad.topk(self.n_candidates, dim=-1).indices

            # --- (II) Greedy coordinate ascent step ---
            current_trigger_ids = trigger_ids.clone()
            # Sample `n_flip` unique positions to optimize
            sampled_positions = torch.randperm(trigger_seq_len, device=self.model.device)[
                : self.n_flip
            ]
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
                        candidate_triggers, tokenizer
                    )
                    if len(candidate_triggers) == 0:
                        logger.debug(
                            f"[WARNING] Retokenize filtering removed all candidates for pos {pos}. Skipping."
                        )
                        continue  # Keep `current_trigger_ids` as is for this position

                # Compute losses on candidate flips
                losses = self.model.compute_loss_from_tokens(
                    candidate_triggers,
                    self.loss_func,
                )  # (n_cands,)

                # Find the best token for this position
                best_candidate_idx = losses.argmin()

                # Update `current_trigger_ids` for the next iteration of the greedy (inner) loop
                current_trigger_ids = candidate_triggers[best_candidate_idx].clone()
                current_loss = losses[best_candidate_idx].item()

            # --- (III) Update the main trigger ----
            # After the inner loop, `current_trigger_ids` is the best trigger for this *entire* step
            trigger_ids = current_trigger_ids
            trigger_str = tokenizer.decode(trigger_ids, skip_special_tokens=True)

            # Logging:
            self.log(loss=current_loss, trigger_str=trigger_str)
            best.update(loss=current_loss, trigger_ids=trigger_ids, trigger_str=trigger_str)

        result = OptimizerResult(
            best_loss=best.loss,
            best_trigger_str=best.trigger_str,
            best_trigger_ids=best.trigger_ids,
            losses=best.losses,
            trigger_strs=best.trigger_strs,
        )
        self.tracker.log({"best_loss": result.best_loss, "best_trigger_str": result.best_trigger_str})
        self.model.reset_inputs_from_tokens()
        return result


    def _get_trigger_variations(
        self,
        trigger_ids: Float[Tensor, "trigger_seq_len"],
        valid_token_ids: Float[Tensor, "n_valid"],
    ) -> Float[Tensor, "n_grad trigger_seq_len"]:
        """
        Creates a list of `n_grad` trigger variations. The first is the
        original trigger, and the rest are random single-token flips of its.
        """
        trigger_seq_len = len(trigger_ids)
        device = self.model.device
        trigger_vars_ids = trigger_ids.repeat(
            self.n_grad, 1
        )  # shape: (n_grad, trigger_seq_len)

        for idx in range(1, self.n_grad):  # (keep the first intact)
            # select a random position and a random token
            pos_to_flip = torch.randint(0, trigger_seq_len, (1,), device=device).item()
            tok_to_flip_to = valid_token_ids[
                torch.randint(0, len(valid_token_ids), (1,), device=device)
            ].item()  # apply the flip
            trigger_vars_ids[idx, pos_to_flip] = tok_to_flip_to

        return trigger_vars_ids
