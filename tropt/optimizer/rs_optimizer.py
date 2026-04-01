"""
Random Search (PRS) optimizer.

Andriushchenko et al., "Jailbreaking Leading Safety-Aligned LLMs with Simple
Adaptive Attacks" (2024).  https://arxiv.org/abs/2404.02151

Zeroth-order (gradient-free) token optimizer that mutates contiguous blocks
of tokens.  A coarse-to-fine schedule shrinks the block size over time
(exploration -> exploitation), and a patience mechanism triggers random
restarts when the search stalls.

"""

import logging
from typing import Optional, Literal

import torch
from jaxtyping import Int
from torch import Tensor

from tropt.common import DEFAULT_INIT_TRIGGER, Targets, TextTemplates
from tropt.loss import BaseLoss
from tropt.model import BaseModel, LossTextAccessMixin
from tropt.model.model_base import BaseTokenizer
from tropt.optimizer import BaseOptimizer, OptimizerResult
from tropt.optimizer.utils.running_best import RunningBest
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.optimizer.utils.token_initializers import get_printable_random_trigger

logger = logging.getLogger(__name__)


class RandomSearchOptimizer(BaseOptimizer):
    """RandomSearch: batched zeroth-order token optimization with block mutation.

    Per step:
      1. Compute block size from coarse-to-fine schedule
      2. For each candidate, pick a random start position and replace a
         contiguous block with random tokens from the allowed set
      3. Decode candidates to strings and evaluate via ``compute_loss_from_texts``
      4. Keep best if it improves current loss
      5. If no improvement for ``patience`` steps, restart from random init

    
    Implementation Notes:
    - Candidate evaluation is always text-based (``compute_loss_from_texts``); even for HF model, 
    we decode to strings and re-encode for model input.
    - A tokenizer is needed for the optimizer's token-level mutations; it should eitehr be provided, or we fall back to the model's tokenizer if it has one. 
    - The original implementation employs a "warm" intiial trigger (eg another GCG suffix), and uses it as the starting point for all restarts.  Here, we sample random triggers for all restarts for diversity.
    - The original implementation employs an LLM judge for early stopping; here we use a simple patience counter for restarts.
    - The original implementation mostly use a loss-based scheduler. For generality (e.g., different potential loss values) we avoid using it. 
    

    Reference implementation:
    - The original implementation: https://github.com/tml-epfl/llm-adaptive-attacks/blob/main/main.py
    - Another (more simplified) implementation: https://github.com/romovpa/claudini/blob/main/claudini/methods/original/prs/optimizer.py

    """

    model_requirements = (LossTextAccessMixin,)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker=None,
        seed: Optional[int] = None,
        # optimization parameters:
        num_steps: int = 500,
        n_candidates: int = 128,
        mutation_mode: Literal["block_random", "single_cyclic"] = "block_random",
        # Block parameters
        schedule: Literal["fixed", "none"] = "fixed",
        initial_block_len: int = 4,
        # misc:
        patience: int = 25,
        token_constraints: TokenConstraints = TokenConstraints(),
        # external tokenizer (required when model has no tokenizer):
        tokenizer: Optional[BaseTokenizer] = None,
    ):
        """
        Args:
            num_steps: Total optimization steps.
            n_candidates: Number of mutated candidates per step. 
            mutation_mode: ``"block_random"`` for random contiguous block mutation
                (original PRS), ``"single_cyclic"`` for single-token mutations
                spread across positions (candidate ``i`` mutates position
                ``i % trigger_len``).
            
            schedule: Schedule for block size decay. Relevant for block mutation mode(s).
                ``"fixed"`` for step-based coarse-to-fine decay,
                ``"none"`` to keep block size constant.
            initial_block_len: Initial block size for mutation. Relevant for block mutation mode(s).

            patience: Restart from random init after this many steps without
                improvement. Set to 0 to disable restarts.
            tokenizer: Tokenizer for encoding/decoding trigger tokens.
                If None, uses model's tokenizer if it has one, else raises an error.
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        assert schedule in ("fixed", "none"), "Unsupported schedule type"

        self.num_steps = num_steps
        self.n_candidates = n_candidates

        self.mutation_mode = mutation_mode

        # Block parameters:
        self.initial_block_len = initial_block_len
        self.schedule = schedule

        self.patience = patience
        self.token_constraints = token_constraints

        # Resolve tokenizer: prefer explicit, fall back to model's
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif hasattr(model, "tokenizer"):
            self.tokenizer = model.tokenizer
        else:
            raise ValueError(
                "No tokenizer available. Pass an external tokenizer when the "
                "model does not expose one."
            )

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #
    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:

        # --- Setup ---
        self.model.set_inputs_from_texts(templates=templates, targets=targets)
        tokenizer = self.tokenizer
        device = self.model.device

        trigger_ids: Int[Tensor, "trigger_seq_len"] = (
            tokenizer.encode_trigger(initial_trigger).to(device)
        )
        trigger_len = trigger_ids.shape[0]

        valid_token_ids = self.token_constraints.get_whitelist_ids(
            tokenizer, tokenizer.vocab_size, device, return_tensor=True
        )
        n_valid = len(valid_token_ids)

        best = RunningBest()

        # Initial loss
        trigger_str = tokenizer.decode_trigger(trigger_ids)
        current_loss = self.model.compute_loss_from_texts(
            [trigger_str], loss_func=self.loss_func
        ).item()
        self.log(loss=current_loss, trigger_str=trigger_str)
        best.update(loss=current_loss, trigger_ids=trigger_ids, trigger_str=trigger_str)

        # Restart bookkeeping
        steps_without_improvement = 0  # consecutive steps without improvement
        step_of_curr_restart = 0  # the step of the most recent restart
        restart_count = 0  # number of restarts (for logging purposes)

        # --- Optimization loop ---
        pbar = self.register_tqdm(range(self.num_steps))
        for step_i in pbar:
            # Check patience — restart if stuck
            if self.patience > 0 and steps_without_improvement >= self.patience:
                restart_count += 1
                trigger_ids = get_printable_random_trigger(
                    trigger_len, return_ids=True, tokenizer=tokenizer
                ).to(device)
                trigger_str = tokenizer.decode_trigger(trigger_ids)
                current_loss = self.model.compute_loss_from_texts(
                    [trigger_str], loss_func=self.loss_func
                ).item()
                steps_without_improvement = 0
                step_of_curr_restart = step_i
                logger.info(
                    "PRS restart #%d at step %d (loss=%.4f)",
                    restart_count, step_i, current_loss,
                )

            # --- Sample candidates ---
            candidates = trigger_ids.unsqueeze(0).expand(self.n_candidates, -1).clone()

            if self.mutation_mode == "single_cyclic":
                block_len = self._mutate_single_cyclic(
                    candidates, valid_token_ids
                )
            else:  # default to block_random
                local_step = step_i - step_of_curr_restart
                block_len = self._mutate_block_random(
                    candidates, valid_token_ids, local_step, trigger_len
                )

            # --- Evaluate candidates (as texts) ---
            candidate_strs = tokenizer.decode_triggers(candidates)
            losses = self.model.compute_loss_from_texts(
                candidate_strs, loss_func=self.loss_func
            )

            # If improved, update current trigger; else increment patience counter
            best_idx = losses.argmin()
            candidate_loss = losses[best_idx].item()

            if candidate_loss < current_loss:
                trigger_ids = candidates[best_idx]
                current_loss = candidate_loss
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            # logging stuff:
            trigger_str = tokenizer.decode_trigger(trigger_ids)
            self.log(
                loss=current_loss,
                trigger_str=trigger_str,
                n_token_flip=block_len,
                restarts=restart_count,
                patience_counter=steps_without_improvement,
            )
            best.update(loss=current_loss, trigger_ids=trigger_ids, trigger_str=trigger_str)

        # --- Finalize ---
        return best.to_result()

    # ------------------------------------------------------------------ #
    #  Mutation strategies
    # ------------------------------------------------------------------ #
    def _mutate_single_cyclic(
        self,
        candidates: Int[Tensor, "n_candidates trigger_len"],
        valid_token_ids: Int[Tensor, "n_valid"],
    ) -> int:
        """Single-token mutation spread across positions round-robin.

        Candidate ``i`` mutates position ``i % trigger_len``.
        Returns block_len (always 1).
        """
        B = candidates.shape[0]
        device = candidates.device
        n_valid = len(valid_token_ids)
        arange_B = torch.arange(B, device=device)

        positions = arange_B % candidates.shape[1]
        random_tokens = valid_token_ids[
            torch.randint(n_valid, (B,), device=device)
        ]
        candidates[arange_B, positions] = random_tokens
        return 1

    def _mutate_block_random(
        self,
        candidates: Int[Tensor, "n_candidates trigger_len"],
        valid_token_ids: Int[Tensor, "n_valid"],
        local_step: int,
        trigger_len: int,
    ) -> int:
        """Contiguous block mutation at random positions.

        Returns the block length used.
        """
        B = candidates.shape[0]
        device = candidates.device
        n_valid = len(valid_token_ids)
        arange_B = torch.arange(B, device=device)

        block_len = min(self._get_curr_block_len(local_step), trigger_len)
        max_start = trigger_len - block_len
        block_starts = torch.randint(0, max_start + 1, (B,), device=device)
        random_tokens = valid_token_ids[
            torch.randint(n_valid, (B, block_len), device=device)
        ]
        for offset in range(block_len):
            candidates[arange_B, block_starts + offset] = random_tokens[:, offset]
        return block_len

    # ------------------------------------------------------------------ #
    #  Coarse-to-fine block-len schedule
    # ------------------------------------------------------------------ #
    def _get_curr_block_len(self, local_step: int) -> int:
        """Block size at a given step within the current restart.

        Follows ``schedule_n_to_change_fixed`` from the paper's official code.
        """
        if self.schedule == "none":
            return self.initial_block_len

        m = self.initial_block_len
        if local_step <= 10:
            return m
        elif local_step <= 25:
            return max(m // 2, 1)
        elif local_step <= 50:
            return max(m // 4, 1)
        elif local_step <= 100:
            return max(m // 8, 1)
        elif local_step <= 500:
            return max(m // 16, 1)
        else:
            return max(m // 32, 1)
