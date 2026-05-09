import logging
import math
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
    LossTextAccessMixin,
    LossTokenAccessMixin,
    TokenAccessMixin,
)
from tropt.model.model_base import BaseTokenizer
from tropt.optimizer import BaseOptimizer, OptimizerResult
from tropt.optimizer.utils.buffer import TriggerBuffer
from tropt.optimizer.utils.retokenization import retokenize_filtering
from tropt.optimizer.utils.running_best import RunningBest
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.optimizer.utils.token_initializers import get_printable_random_trigger
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)


class QCGOptimizer(BaseOptimizer):
    """Greedy Coordinate Query optimizer (Hayase et al., 2024).

    Buffer-based query attack: maintains a buffer of B best triggers, expands
    from the best entry each step via random single-token flips,
    uses the proxy models to filter to top-K, then evaluates on the target model.

    Reference: https://arxiv.org/abs/2402.12329
    """

    model_requirements = (LossTextAccessMixin,)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # Proxy model for candidate filtering:
        proxy_model: Optional[BaseModel] = None,
        # Core QCG params:
        num_steps: int = 500,
        n_proxy_candidates: int = 8192,
        n_target_candidates: int = 32,
        buffer_size: int = 128,
        token_constraints: TokenConstraints = TokenConstraints(),
        candidate_oversample_factor: float = 1.5,
    ):
        """
        Args:
            proxy_model: Model used for proxy loss filtering (tokenizer + loss).
                If None, defaults to `model` (self-proxy / white-box).
            n_proxy_candidates: Number of random candidates to sample per step (evaluated on proxy; b_p in paper).
            n_target_candidates: Number of candidates to evaluate on target after
                proxy candidate filtering (b_q in paper).
                Must be <= n_proxy_candidates.
            buffer_size: Number of triggers to maintain in the buffer (B in the paper).
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        self.proxy_model = model if proxy_model is None else proxy_model
        assert isinstance(self.proxy_model, TokenAccessMixin), (
            "proxy_model must support TokenAccessMixin (tokenizer access)"
        )
        assert isinstance(self.proxy_model, LossTokenAccessMixin), (
            "proxy_model must support LossTokenAccessMixin (proxy loss filtering)"
        )

        # Prefer token-level target evaluation when proxy and target share the same tokenizer
        use_token_eval = (model.tokenizer == self.proxy_model.tokenizer) and isinstance(model, LossTokenAccessMixin)    # TODO find a more accurate way of comparing tokenizers

        self.num_steps = num_steps
        self.n_proxy_candidates = n_proxy_candidates
        self.n_target_candidates = n_target_candidates
        self.buffer_size = buffer_size
        self.token_constraints = token_constraints
        self.candidate_oversample_factor = candidate_oversample_factor
        self.use_token_eval = use_token_eval

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:

        # --- Initialization ---
        proxy_model = self.proxy_model
        target_model = self.model
        proxy_tokenizer: BaseTokenizer = proxy_model.tokenizer

        proxy_model.set_inputs_from_tokens(templates=templates, targets=targets)
        if not self.use_token_eval:
            target_model.set_inputs_from_texts(templates=templates, targets=targets)
        else:
            target_model.set_inputs_from_tokens(templates=templates, targets=targets)

        trigger_ids: Int[Tensor, "trigger_seq_len"] = proxy_tokenizer.encode_trigger(initial_trigger).to(proxy_model.device)

        vocab_size = proxy_model.vocab_size

        valid_token_ids = self.token_constraints.get_whitelist_ids(proxy_tokenizer, vocab_size, proxy_model.device, return_tensor=True)

        best = RunningBest()

        # Buffer initialization: fill the buffer with the initial trigger + random triggers
        buffer: TriggerBuffer = self._init_buffer(trigger_ids)

        # Oversample to compensate for retokenization filtering
        n_proxy_candidates_oversampled = math.ceil(self.n_proxy_candidates * self.candidate_oversample_factor)

        # Log initial state
        current_loss = buffer.get_lowest_loss()
        trigger_ids = buffer.get_best_trigger()
        trigger_str = proxy_tokenizer.decode_trigger(trigger_ids)
        self.log(loss=current_loss, trigger_str=trigger_str)

        for _ in self.track_steps(range(self.num_steps)):
            # Work on the best trigger in the buffer
            trigger_ids = buffer.get_best_trigger()

            # === Stage 1: Generate random candidates from buffer best ===
            candidate_trigger_ids = self._sample_ids_at_random(
                trigger_ids=trigger_ids,
                valid_token_ids=valid_token_ids,
                n_candidates=n_proxy_candidates_oversampled,
            )

            # === Retokenization filtering ===
            candidate_trigger_ids = retokenize_filtering(
                candidate_trigger_ids, proxy_tokenizer
            )
            candidate_trigger_ids = candidate_trigger_ids[: self.n_proxy_candidates]

            # === Stage 2: Proxy filtering to n_target_candidates ===
            # TODO what happens if proxy == target? shouldn't we skip it??
            if self.n_target_candidates < len(candidate_trigger_ids):
                proxy_losses = proxy_model.compute_loss_from_tokens(
                    candidate_trigger_ids, loss_func=self.loss_func
                )
                topk_indices = proxy_losses.topk(
                    self.n_target_candidates, largest=False
                ).indices
                candidate_trigger_ids = candidate_trigger_ids[topk_indices]

            # === Stage 3: Evaluate on target and update buffer ===
            losses = self._evaluate_candidates_on_target_model(candidate_trigger_ids)

            for idx in range(len(candidate_trigger_ids)):
                buffer.add_if_better(
                    trigger_ids=candidate_trigger_ids[idx],
                    loss=losses[idx].item()
                )

            current_loss = buffer.get_lowest_loss()
            trigger_ids = buffer.get_best_trigger()
            trigger_str = proxy_tokenizer.decode_trigger(trigger_ids)

            best.update(loss=current_loss, trigger_ids=trigger_ids, trigger_str=trigger_str)
            self.log(loss=current_loss, trigger_str=trigger_str)

        # --- Finalize ---
        result = best.to_result()


        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_buffer(
        self,
        initial_trigger_ids: Int[Tensor, "trigger_seq_len"],
    ) -> TriggerBuffer:
        """Initialize buffer: first entry from initial trigger, rest printable-random.
        """
        trigger_seq_len = initial_trigger_ids.shape[0]
        device = initial_trigger_ids.device
        tokenizer = self.proxy_model.tokenizer
        blacklist_ids = self.token_constraints.get_blacklist_ids(tokenizer, self.proxy_model.vocab_size)

        triggers_list: list[Tensor] = [initial_trigger_ids]
        for _ in range(self.buffer_size - 1):
            # Oversample length (reencode can drift shorter), then trim to trigger_seq_len.
            rand_str = get_printable_random_trigger(
                trigger_len=2 * trigger_seq_len,
                tokenizer=tokenizer,
                blacklist_ids=blacklist_ids,
            )
            ids = tokenizer.encode_trigger(rand_str).to(device)[:trigger_seq_len]
            triggers_list.append(ids)

        all_triggers = torch.stack(triggers_list, dim=0)
        all_losses = self._evaluate_candidates_on_target_model(all_triggers)

        return TriggerBuffer(
            triggers=list(all_triggers),
            losses=[all_losses[i].item() for i in range(self.buffer_size)],
        )

    def _sample_ids_at_random(
        self,
        trigger_ids: Int[Tensor, "trigger_seq_len"],
        valid_token_ids: Int[Tensor, "n_valid"],
        n_candidates: int,
    ) -> Int[Tensor, "n_candidates trigger_seq_len"]:
        """Sample candidates by replacing one random position with a random token."""
        trigger_seq_len = trigger_ids.shape[0]
        device = trigger_ids.device

        candidate_trigger_ids = trigger_ids.repeat(n_candidates, 1).clone()

        # One random position per candidate
        sampled_pos = torch.randint(
            0, trigger_seq_len, (n_candidates, 1), device=device
        )

        # One random valid token per candidate
        rand_indices = torch.randint(
            0, len(valid_token_ids), (n_candidates, 1), device=device
        )
        random_tokens = valid_token_ids[rand_indices]

        candidate_trigger_ids = candidate_trigger_ids.scatter_(
            dim=-1,
            index=sampled_pos,
            src=random_tokens,
        )
        return candidate_trigger_ids

    def _evaluate_candidates_on_target_model(
        self,
        candidate_trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"],
    ) -> Float[Tensor, "n_candidates"]:
        """Evaluate candidates on the target model. Returns per-candidate loss."""
        if self.use_token_eval:
            losses = self.model.compute_loss_from_tokens(
                candidate_trigger_ids, loss_func=self.loss_func
            )
        else:
            candidate_strs = self.proxy_model.tokenizer.decode_triggers(candidate_trigger_ids)
            losses = self.model.compute_loss_from_texts(
                candidate_strs, loss_func=self.loss_func
            )

        return losses
