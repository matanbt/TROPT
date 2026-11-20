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
from tropt.optimizer.utils.retokenization import retokenize_mask
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

        # Initialization:
        self.model.set_inputs_from_tokens(templates=templates, targets=targets)
        tokenizer = self.model.tokenizer
        trigger_ids: Int[Tensor, "trigger_seq_len"] = tokenizer.encode_trigger(initial_trigger).to(self.model.device)

        vocab_size = self.model.vocab_size
        blacklist_ids = self.token_constraints.get_blacklist_ids(tokenizer, vocab_size)
        valid_token_ids = self.token_constraints.get_whitelist_ids(tokenizer, vocab_size, return_tensor=True).to(self.model.device)

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

        for step in self.track_steps(range(self.num_steps), desc="Optimizing with GASLITE..."):

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
            topk_ids: list = trigger_grad.topk(self.n_candidates, dim=-1).indices.tolist()

            # --- (II) Greedy coordinate ascent step ---
            # Candidates are built host-side, so each flip needs a single host sync (for the losses)
            current_trigger_ids = trigger_ids.tolist()
            # Sample `n_flip` unique positions to optimize
            sampled_positions = sorted(
                torch.randperm(trigger_seq_len, device=self.model.device)[: self.n_flip].tolist()
            )

            # Sequentially optimize each position
            for pos in sampled_positions:
                # Candidate tokens for this position (sorted & unique; keeps the "no flip" option)
                tokens = sorted(set(topk_ids[pos]) | {current_trigger_ids[pos]})

                # Create all candidate triggers by flipping this *single* position
                candidate_triggers = torch.tensor(current_trigger_ids).repeat(len(tokens), 1)
                candidate_triggers[:, pos] = torch.tensor(tokens)
                candidate_triggers = candidate_triggers.to(self.model.device)

                # Compute losses on candidate flips
                losses = self.model.compute_loss_from_tokens(
                    candidate_triggers,
                    self.loss_func,
                ).tolist()  # (n_cands,)

                # Find the best token for this position: the first minimum (as `argmin`) among the candidates that
                # survive (optional) retokenize filtering, checked lazily in ascending-loss order.
                for idx in sorted(range(len(losses)), key=losses.__getitem__):
                    if not self.use_retokenize or retokenize_mask(candidate_triggers[idx : idx + 1], tokenizer)[0]:
                        # Update `current_trigger_ids` for the next iteration of the greedy (inner) loop
                        current_trigger_ids = candidate_triggers[idx].tolist()
                        current_loss = losses[idx]
                        break
                else:
                    # Mirrors `retokenize_filtering`'s behavior when no candidate survives
                    raise RuntimeError(
                        "No token sequences are the same after decoding and re-encoding. "
                        "Consider disabling retokenization filtering by setting `use_retokenize=False`."
                    )

            # --- (III) Update the main trigger ----
            # After the inner loop, `current_trigger_ids` is the best trigger for this *entire* step
            trigger_ids = torch.tensor(current_trigger_ids, device=self.model.device)
            trigger_str = tokenizer.decode_trigger(trigger_ids)

            # Logging:
            self.log(loss=current_loss, trigger_str=trigger_str)
            best.update(loss=current_loss, trigger_ids=trigger_ids, trigger_str=trigger_str)

        return best.to_result()


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
        trigger_vars_ids = trigger_ids.repeat(self.n_grad, 1)  # shape: (n_grad, trigger_seq_len)

        # For each variation but the first (kept intact), flip a random position to a random valid token
        n_flips = self.n_grad - 1
        pos_to_flip = torch.randint(0, trigger_seq_len, (n_flips,), device=device)
        tok_to_flip_to = valid_token_ids[torch.randint(0, len(valid_token_ids), (n_flips,), device=device)]
        trigger_vars_ids[torch.arange(1, self.n_grad, device=device), pos_to_flip] = tok_to_flip_to

        return trigger_vars_ids
