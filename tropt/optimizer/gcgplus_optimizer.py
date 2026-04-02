import logging
import math
from typing import Literal, Optional, Tuple, Union

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
    LossTextAccessMixin,
    LossTokenAccessMixin,
    TokenAccessMixin,
)
from tropt.optimizer import BaseOptimizer, OptimizerResult
from tropt.optimizer.utils.buffer import TriggerBuffer
from tropt.optimizer.utils.retokenization import retokenize_filtering
from tropt.optimizer.utils.running_best import RunningBest
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)


class GCGPlusOptimizer(BaseOptimizer):
    """Flexible GCG implementation, supporting tricks from GCG, QCG, and GASLITE.

    Two-stage design:
      1. Candidate selection (on proxy model) — gradient-based, random, or focused.
      2. Candidate evaluation (on target model) — via text or token access.

    References:
      - GCG: https://arxiv.org/abs/2307.15043
      - QCG: https://arxiv.org/abs/2402.12329
      - PAL: https://arxiv.org/abs/2402.09674
      - GASLITE: https://arxiv.org/abs/2412.20953
    """

    model_requirements = (LossTextAccessMixin,)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # Proxy model for candidate selection (defaults to model):
        proxy_model: Optional[BaseModel] = None,
        # Candidate selection strategy:
        candidate_selection: Literal["gradient", "random", "focused"] = "gradient",
        # Core GCG params:
        num_steps: int = 500,
        n_candidates: int = 512,
        sample_topk: int = 256,
        token_constraints: TokenConstraints = TokenConstraints(),
        use_retokenize: bool = True,
        # Later tricks:
        sample_n_replace: Union[int, Tuple[int, int]] = (1, 1),
        candidate_oversample_factor: float = 1.1,
        momentum: float = 0.0,
        # Trigger buffer size:
        buffer_size: Optional[int] = None,
        n_grad_avg: int = 1,
    ):
        """
        Args:
            proxy_model: Model used for candidate selection (gradients/tokenizer).
                If None, defaults to `model` (self-proxy / white-box).
            candidate_selection: "gradient" uses gradient-ranked top-k sampling;
                "random" uses uniform random token sampling; "focused" probes
                all positions with target model loss then focuses on the best one (from QCG paper).
            sample_n_replace: (start, end) number of token positions to replace per
                candidate. Linearly interpolated over optimization steps; from PAL paper.
                Defaults to (1, 1) for single-token flips only.
            candidate_oversample_factor: Craft n_candidates * factor candidates (>1.0), then
                truncate after retokenization filtering, to fill the required number of candidates.
                From PAL paper.
                Defaults to 1.1 (10% oversampling).
            momentum: Gradient momentum coefficient (mu), from PAL paper. When > 0, uses
                m = mu*m + grad for candidate ranking instead of raw gradient.
                Defaults to 0.0 (no momentum).
            buffer_size: If set, maintain a buffer of the best triggers seen (from QCG paper). Each step
                starts from the best buffer entry and updates it with improved candidates.
                Defaults to None (no buffer).
            n_grad_avg: Number of trigger perturbations to average gradients
                over. When > 1, flips a random position per copy (from ARCA/GASLITE papers).
                Defaults to 1 (no averaging, like GCG).
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        # Proxy model setup
        self.proxy_model = model if proxy_model is None else proxy_model
        assert isinstance(self.proxy_model, TokenAccessMixin), (
            "proxy_model must support TokenAccessMixin (tokenizer access)"
        )
        if candidate_selection == "gradient":
            assert isinstance(self.proxy_model, GradientTokenAccessMixin), (
                "candidate_selection='gradient' requires proxy_model with GradientTokenAccessMixin"
            )
        assert candidate_oversample_factor >= 1.0, "candidate_oversample_factor must be >= 1.0"

        # Prefer token-level target evaluation when proxy and target share the same tokenizer
        # TODO BETTER NAMING!
        use_token_eval = (model.tokenizer == self.proxy_model.tokenizer) and isinstance(model, LossTokenAccessMixin)

        # Normalize sample_n_replace to tuple
        if isinstance(sample_n_replace, int):
            sample_n_replace = (sample_n_replace, sample_n_replace)

        # Save params
        self.candidate_selection = candidate_selection
        self.num_steps = num_steps
        self.n_candidates = n_candidates
        self.sample_topk = sample_topk
        self.sample_n_replace = sample_n_replace
        self.token_constraints = token_constraints
        self.use_retokenize = use_retokenize
        self.candidate_oversample_factor = candidate_oversample_factor
        self.momentum = momentum
        self.use_token_eval = use_token_eval
        self.buffer_size = buffer_size
        self.n_grad_avg = n_grad_avg

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:

        # --- Initialization ---
        proxy_model = self.proxy_model
        proxy_tokenizer = proxy_model.tokenizer
        target_model = self.model

        proxy_model.set_inputs_from_tokens(templates=templates, targets=targets)
        if not self.use_token_eval:
            target_model.set_inputs_from_texts(templates=templates, targets=targets)
        else:
            target_model.set_inputs_from_tokens(templates=templates, targets=targets)

        trigger_ids: Int[Tensor, "trigger_seq_len"] = proxy_tokenizer.encode_trigger(initial_trigger).to(proxy_model.device)

        vocab_size = proxy_model.vocab_size

        blacklist_ids = self.token_constraints.get_blacklist_ids(proxy_tokenizer, vocab_size)
        valid_token_ids = self.token_constraints.get_whitelist_ids(proxy_tokenizer, vocab_size, proxy_model.device, return_tensor=True)

        best = RunningBest()
        momentum_buffer: Optional[Tensor] = None

        # Buffer initialization
        buffer: Optional[TriggerBuffer] = None
        if self.buffer_size is not None:
            buffer = self._init_buffer(trigger_ids, valid_token_ids)

        # Number of candidates to generate (oversample if retokenize is on)
        n_candidates_oversampled = self.n_candidates
        if self.use_retokenize and self.candidate_oversample_factor > 1.0:
            n_candidates_oversampled = math.ceil(self.n_candidates * self.candidate_oversample_factor)

        # Initial loss
        current_loss = self._evaluate_candidates(
            trigger_ids.unsqueeze(0)
        ).item()
        trigger_str = proxy_tokenizer.decode_trigger(trigger_ids)
        self.log(loss=current_loss, trigger_str=trigger_str)

        pbar = self.register_tqdm(range(self.num_steps))
        n_replace_start, n_replace_end = self.sample_n_replace

        for step_i in pbar:
            # Linearly interpolate sample_n_replace over steps
            cur_n_replace = round(
                n_replace_start + (n_replace_end - n_replace_start) * step_i / max(self.num_steps - 1, 1)
            )

            # Buffer mode: start each step from the best buffer entry
            if buffer is not None:
                trigger_ids = buffer.get_best_trigger()

            # === Stage 1: Candidate Selection (on proxy) ===

            if self.candidate_selection == "gradient":
                grad_triggers = self._get_grad_trigger_variations(
                    trigger_ids, valid_token_ids
                )
                trigger_grad = proxy_model.compute_grad_from_tokens(
                    candidate_trigger_ids=grad_triggers,
                    loss_func=self.loss_func,
                    normalize_grads=True,
                ).mean(dim=0)  # average over n_grad_avg variations

                # Apply momentum
                if self.momentum > 0:
                    if momentum_buffer is None:
                        momentum_buffer = trigger_grad.clone()
                    else:
                        momentum_buffer = self.momentum * momentum_buffer + trigger_grad
                    trigger_grad = momentum_buffer

                candidate_trigger_ids = self._sample_ids_from_grad(
                    trigger_ids=trigger_ids,
                    trigger_grad=trigger_grad,
                    blacklist_ids=blacklist_ids,
                    n_candidates=n_candidates_oversampled,
                    n_replace=cur_n_replace,
                )
            elif self.candidate_selection == "focused":
                candidate_trigger_ids = self._sample_focused_candidates(
                    trigger_ids=trigger_ids,
                    valid_token_ids=valid_token_ids,
                    n_candidates=n_candidates_oversampled,
                )
            else:  # "random"
                candidate_trigger_ids = self._sample_random_candidates(
                    trigger_ids=trigger_ids,
                    valid_token_ids=valid_token_ids,
                    n_candidates=n_candidates_oversampled,  #TODO ->n_candidates
                    n_replace=cur_n_replace,
                )

            # === Retokenization filtering ===
            if self.use_retokenize:
                candidate_trigger_ids = retokenize_filtering(
                    candidate_trigger_ids, proxy_tokenizer
                )
            # Truncate to n_candidates (after oversample + filter)
            candidate_trigger_ids = candidate_trigger_ids[: self.n_candidates]

            if len(candidate_trigger_ids) == 0:
                logger.warning("All candidates filtered out, skipping step.")
                continue

            # === Stage 2: Candidate Evaluation (on target) ===
            losses = self._evaluate_candidates(candidate_trigger_ids)

            # === Update best / buffer ===
            if buffer is not None:
                # Buffer mode: update buffer with all evaluated candidates
                for idx in range(len(candidate_trigger_ids)):
                    buffer.add_if_better(
                        candidate_trigger_ids[idx], losses[idx].item()
                    )
                # Track overall best from buffer for logging
                current_loss = buffer.get_lowest_loss()
                trigger_ids = buffer.get_best_trigger()
                trigger_str = proxy_tokenizer.decode_trigger(trigger_ids)
            else:
                current_loss = losses.min().item()
                trigger_ids = candidate_trigger_ids[losses.argmin()]
                trigger_str = proxy_tokenizer.decode_trigger(trigger_ids)

            best.update(loss=current_loss, trigger_ids=trigger_ids, trigger_str=trigger_str)
            self.log(loss=current_loss, trigger_str=trigger_str)

        # --- Finalize ---
        result = best.to_result()


        return result

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_buffer(
        self,
        initial_trigger_ids: Int[Tensor, "trigger_seq_len"],
        valid_token_ids: Int[Tensor, "n_valid"],
    ) -> TriggerBuffer:
        """Initialize the buffer with the initial trigger and random variants. Batched."""
        assert self.buffer_size is not None
        trigger_seq_len = initial_trigger_ids.shape[0]
        device = initial_trigger_ids.device

        rand_triggers = valid_token_ids[
            torch.randint(0, len(valid_token_ids), (self.buffer_size - 1, trigger_seq_len), device=device)
        ]
        all_triggers = torch.cat([initial_trigger_ids.unsqueeze(0), rand_triggers], dim=0)
        all_losses = self._evaluate_candidates(all_triggers)

        return TriggerBuffer(
            triggers=list(all_triggers),
            losses=[all_losses[i].item() for i in range(self.buffer_size)],
        )

    # ------------------------------------------------------------------
    # Stage 1: Candidate Selection
    # ------------------------------------------------------------------

    def _sample_ids_from_grad(
        self,
        trigger_ids: Int[Tensor, "trigger_seq_len"],
        trigger_grad: Float[Tensor, "trigger_seq_len vocab_size"],
        blacklist_ids: list,
        n_candidates: int,
        n_replace: int = 1,
    ) -> Int[Tensor, "n_candidates trigger_seq_len"]:
        """Sample candidate token sequences via GCG-style random multi-position replacement."""
        trigger_seq_len = trigger_grad.shape[0]
        device = trigger_grad.device
        n_replace = min(n_replace, trigger_seq_len)

        # Top-k candidates per position
        trigger_grad = trigger_grad.clone()  # avoid mutating momentum buffer
        trigger_grad *= -1
        trigger_grad[:, blacklist_ids] = float("-inf")
        topk_ids = trigger_grad.topk(self.sample_topk, dim=-1).indices

        candidate_trigger_ids = trigger_ids.repeat(n_candidates, 1).clone()

        # Random positions to flip
        sampled_ids_pos = torch.rand(
            n_candidates, trigger_seq_len, device=device
        ).argsort(dim=-1)[..., :n_replace]

        # Select relevant top-k lists and sample one token from each
        relevant_topk_lists = topk_ids[sampled_ids_pos]
        rand_k_indices = torch.randint(
            0,
            self.sample_topk,
            (n_candidates, n_replace, 1),
            device=device,
        )
        sampled_ids_val = torch.gather(
            input=relevant_topk_lists,
            dim=-1,
            index=rand_k_indices,
        ).squeeze(-1)

        candidate_trigger_ids = candidate_trigger_ids.scatter_(
            dim=-1,
            index=sampled_ids_pos,
            src=sampled_ids_val,
        )
        return candidate_trigger_ids

    def _get_grad_trigger_variations(
        self,
        trigger_ids: Int[Tensor, "trigger_seq_len"],
        valid_token_ids: Int[Tensor, "n_valid"],
    ) -> Int[Tensor, "n_grad_avg trigger_seq_len"]:
        """Create trigger variations for gradient averaging (GASLITE-style).

        With n_grad_avg == 1, returns the trigger as-is.
        With n_grad_avg > 1, flips a random position with a random valid token
        per copy (keeping the first copy intact).
        """
        if self.n_grad_avg <= 1:
            return trigger_ids.unsqueeze(0)

        device = trigger_ids.device
        trigger_seq_len = trigger_ids.shape[0]
        grad_triggers = trigger_ids.repeat(self.n_grad_avg, 1).clone()

        # Keep first copy intact, perturb the rest
        for idx in range(1, self.n_grad_avg):
            pos = torch.randint(0, trigger_seq_len, (1,), device=device).item()
            tok = valid_token_ids[
                torch.randint(0, len(valid_token_ids), (1,), device=device)
            ].item()
            assert isinstance(pos, int) and isinstance(tok, int)

            grad_triggers[idx, pos] = tok

        return grad_triggers

    def _sample_random_candidates(
        self,
        trigger_ids: Int[Tensor, "trigger_seq_len"],
        valid_token_ids: Int[Tensor, "n_valid"],
        n_candidates: int,
        n_replace: int = 1,
    ) -> Int[Tensor, "n_candidates trigger_seq_len"]:
        """Sample candidates by randomly replacing tokens (RAL-style)."""
        trigger_seq_len = trigger_ids.shape[0]
        device = trigger_ids.device
        n_replace = min(n_replace, trigger_seq_len)

        candidate_trigger_ids = trigger_ids.repeat(n_candidates, 1).clone()

        # Random positions to flip
        sampled_ids_pos = torch.rand(
            n_candidates, trigger_seq_len, device=device
        ).argsort(dim=-1)[..., :n_replace]

        # Sample uniformly from valid tokens
        rand_indices = torch.randint(
            0, len(valid_token_ids), (n_candidates, n_replace), device=device
        )
        random_tokens = valid_token_ids[rand_indices]

        candidate_trigger_ids = candidate_trigger_ids.scatter_(
            dim=-1,
            index=sampled_ids_pos,
            src=random_tokens,
        )
        return candidate_trigger_ids

    def _sample_focused_candidates(
        self,
        trigger_ids: Int[Tensor, "trigger_seq_len"],
        valid_token_ids: Int[Tensor, "n_valid"],
        n_candidates: int,
    ) -> Int[Tensor, "n_candidates trigger_seq_len"]:
        """Focused position sampling (from QCG paper).

        Phase 1: Probe each position with one random token replacement,
        evaluate all probes on the target (via _evaluate_candidates) to find
        the most promising position.
        Phase 2: Generate n_candidates candidates at the best position.
        """
        trigger_seq_len = trigger_ids.shape[0]
        device = trigger_ids.device

        # Phase 1: Probe each position with one random token
        probe_candidates = trigger_ids.repeat(trigger_seq_len, 1).clone()
        probe_tokens = valid_token_ids[
            torch.randint(0, len(valid_token_ids), (trigger_seq_len,), device=device)
        ]
        # Replace position j in candidate j
        probe_candidates[
            torch.arange(trigger_seq_len, device=device),
            torch.arange(trigger_seq_len, device=device),
        ] = probe_tokens

        probe_losses = self._evaluate_candidates(probe_candidates)
        best_pos = probe_losses.argmin().item()
        assert isinstance(best_pos, int)

        # Phase 2: Generate candidates at best position only
        candidates = trigger_ids.repeat(n_candidates, 1).clone()
        random_tokens = valid_token_ids[
            torch.randint(0, len(valid_token_ids), (n_candidates,), device=device)
        ]
        candidates[:, best_pos] = random_tokens
        return candidates

    # ------------------------------------------------------------------
    # Stage 2: Candidate Evaluation
    # ------------------------------------------------------------------

    def _evaluate_candidates(
        self,
        candidate_trigger_ids: Int[Tensor, "n_candidates trigger_seq_len"],
    ) -> Float[Tensor, "n_candidates"]:
        """Evaluate candidates on the target model. Returns per-candidate loss.

        When use_token_eval is False, decodes candidates via proxy_model.tokenizer.
        """
        if self.use_token_eval:
            losses = self.model.compute_loss_from_tokens(
                candidate_trigger_ids, loss_func=self.loss_func
            )
        else:
            candidate_strs = self.proxy_model.tokenizer.decode_triggers(candidate_trigger_ids)
            losses = self.model.compute_loss_from_texts(
                candidate_strs, loss_func=self.loss_func
            )

        # Reduce to (n_candidates,) if needed
        if losses.dim() > 1:
            losses = losses.mean(dim=0)
        return losses


