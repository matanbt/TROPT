import itertools
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
        independent_templates: bool = False,
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
            independent_templates (bool): Optimize a separate trigger per template (e.g., one per target, for an attack
                budget > 1), batching all of them through shared forward passes. Templates must share token lengths.
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
        self.independent_templates = independent_templates

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:

        # Initialization:
        self.model.set_inputs_from_tokens(templates=templates, targets=targets)
        tokenizer = self.model.tokenizer
        device = self.model.device
        # Each chain optimizes its own trigger (one chain, unless `independent_templates`)
        n_chains = len(templates) if self.independent_templates else 1
        chain_template_idx = torch.arange(n_chains, device=device) if self.independent_templates else None
        trigger_ids = tokenizer.encode_trigger(initial_trigger).to(device).repeat(n_chains, 1)
        trigger_ids: Int[Tensor, "n_chains trigger_seq_len"]

        vocab_size = self.model.vocab_size
        blacklist_ids = self.token_constraints.get_blacklist_ids(tokenizer, vocab_size)
        valid_token_ids = self.token_constraints.get_whitelist_ids(tokenizer, vocab_size, return_tensor=True).to(device)

        trigger_seq_len = trigger_ids.shape[1]
        bests = [RunningBest() for _ in range(n_chains)]

        # Calculate initial loss for logging
        chain_losses = self.model.compute_loss_from_tokens(
            trigger_ids, self.loss_func, candidate_template_idx=chain_template_idx,
        ).tolist()
        self.log(loss=sum(chain_losses) / n_chains, trigger_str=initial_trigger)

        for step in self.track_steps(range(self.num_steps), desc="Optimizing with GASLITE..."):

            # --- (I) Gradient and candidate selection step (all chains at once) ---
            if self.use_random_gradient:
                # Replace model gradient with random values
                trigger_grad = torch.randn((n_chains, trigger_seq_len, vocab_size), device=device)
            else:
                # Compute grad over a list of `n_grad` triggers one-flip away from the current (per chain)
                trigger_vars = torch.cat([self._get_trigger_variations(ids, valid_token_ids) for ids in trigger_ids])
                # Average the (normalized) gradients to get the final approximation (per chain)
                trigger_grad = self.model.compute_grad_from_tokens(
                    candidate_trigger_ids=trigger_vars,
                    loss_func=self.loss_func,
                    normalize_grads=True,
                    mean_over_candidates=True,
                    candidate_template_idx=(
                        chain_template_idx.repeat_interleave(self.n_grad) if self.independent_templates else None
                    ),
                ).view(n_chains, trigger_seq_len, vocab_size)
                trigger_grad *= -1  # we want to minimize the loss
            trigger_grad: Float[Tensor, "n_chains trigger_seq_len vocab_size"]

            # Get Top-k Candidates *per position*
            trigger_grad[..., blacklist_ids] = float("-inf")
            topk_ids: list = trigger_grad.topk(self.n_candidates, dim=-1).indices.tolist()

            # --- (II) Greedy coordinate ascent step ---
            # Candidates are built host-side; all chains' candidates share each forward pass (one host sync per flip)
            current_trigger_ids = trigger_ids.tolist()
            # Sample `n_flip` unique positions to optimize (per chain)
            sampled_positions = [
                sorted(torch.randperm(trigger_seq_len, device=device)[: self.n_flip].tolist())
                for _ in range(n_chains)
            ]

            # Sequentially optimize each position
            for flip_idx in range(len(sampled_positions[0])):
                chain_candidates = []
                for chain, curr in enumerate(current_trigger_ids):
                    pos = sampled_positions[chain][flip_idx]
                    # Candidate tokens for this position (sorted & unique; keeps the "no flip" option)
                    tokens = sorted(set(topk_ids[chain][pos]) | {curr[pos]})

                    # Create all candidate triggers by flipping this *single* position
                    candidate_triggers = torch.tensor(curr).repeat(len(tokens), 1)
                    candidate_triggers[:, pos] = torch.tensor(tokens)
                    chain_candidates.append(candidate_triggers)

                # Compute losses on candidate flips
                n_chain_candidates = [len(c) for c in chain_candidates]
                all_candidates = torch.cat(chain_candidates)
                losses = self.model.compute_loss_from_tokens(
                    all_candidates.to(device),
                    self.loss_func,
                    candidate_template_idx=(
                        chain_template_idx.repeat_interleave(torch.tensor(n_chain_candidates, device=device))
                        if self.independent_templates else None
                    ),
                ).tolist()  # (sum(n_chain_candidates),)

                # Find the best token for this position, per chain: the first minimum (as `argmin`) among the
                # candidates that survive (optional) retokenize filtering, checked lazily in ascending-loss order.
                offsets = list(itertools.accumulate(n_chain_candidates, initial=0))
                ranked = [
                    sorted(range(offsets[c], offsets[c + 1]), key=losses.__getitem__) for c in range(n_chains)
                ]
                rank = [0] * n_chains
                pending = list(range(n_chains))
                while pending:  # each round checks the current best of all pending chains in one batch
                    if any(rank[c] == len(ranked[c]) for c in pending):
                        # Mirrors `retokenize_filtering`'s behavior when no candidate survives
                        raise RuntimeError("No token sequences are the same after decoding and re-encoding. "
                                           "Consider disabling retokenization filtering by setting `use_retokenize=False`.")
                    best = [ranked[c][rank[c]] for c in pending]
                    ok = retokenize_mask(all_candidates[best], tokenizer) if self.use_retokenize else [True] * len(best)
                    for c, idx, is_ok in zip(pending, best, ok):
                        if is_ok:
                            # Update the chain's trigger for the next iteration of the greedy (inner) loop
                            current_trigger_ids[c] = all_candidates[idx].tolist()
                            chain_losses[c] = losses[idx]
                        rank[c] += 1
                    pending = [c for c, is_ok in zip(pending, ok) if not is_ok]

            # --- (III) Update the main trigger ----
            # After the inner loop, `current_trigger_ids` is the best trigger for this *entire* step
            trigger_ids = torch.tensor(current_trigger_ids, device=device)
            trigger_strs = [tokenizer.decode_trigger(ids) for ids in current_trigger_ids]

            # Logging:
            self.log(loss=sum(chain_losses) / n_chains, trigger_str=trigger_strs[0] if n_chains == 1 else None)
            for best, loss, ids, trigger_str in zip(bests, chain_losses, trigger_ids, trigger_strs):
                best.update(loss=loss, trigger_ids=ids, trigger_str=trigger_str)

        if not self.independent_templates:
            return bests[0].to_result()
        return OptimizerResult(
            best_loss=sum(b.loss for b in bests) / n_chains,
            best_trigger_ids=torch.stack([b.trigger_ids for b in bests]),
            best_trigger_strs=[b.trigger_str for b in bests],
        )

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
