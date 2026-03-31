import logging
import math
from typing import Literal, Optional, Set

import torch
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

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
from tropt.optimizer.utils.retokenization import retokenize_filtering
from tropt.optimizer.utils.running_best import RunningBest
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)


class PALOptimizer(BaseOptimizer):
    """Proxy-guided black-box optimizer for PAL and RAL attacks.

    Two-stage design:
      1. Candidate selection (on proxy) — gradient-based or random.
      2. Candidate evaluation (on target) — via text or token access.

    Skip-visited is always enabled. Optional proxy filtering narrows
    candidates by proxy loss before querying the target.

    References:
      - Paper: https://arxiv.org/abs/2402.09674
      - Official Codebase: https://github.com/chawins/pal

    Note: PAL original implementation also support fine-tuning the proxy model (to get it closer to the target model in the "optimized area"), we currently don't support that.
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
        candidate_selection: Literal["gradient", "random"] = "gradient",
        # Core GCG params:
        num_steps: int = 500,
        n_candidates: int = 512,
        sample_topk: int = 256,
        sample_n_replace: int = 1,
        candidate_oversample_factor: float = 1.5,
        token_constraints: TokenConstraints = TokenConstraints(),
        # PAL-specific: proxy filtering (set to None to disable):
        n_candidates_after_proxy_filter: Optional[int] = None,
    ):
        """
        Args:
            model: Target model to attack (used for candidate evaluation).
                By default loss is calculated wrt texts, or, if possible (ie, the model share tokenizer with the proxy model and the required access level), computation on tokens is preferred.
            loss: Loss function to optimize.
            proxy_model: Model used for candidate selection (gradients/tokenizer).
                If None, defaults to `model` (self-proxy / white-box).
            candidate_selection: "gradient" uses gradient-ranked top-k sampling,
                "random" uses uniform random token sampling.
            sample_topk: Number of top tokens to consider for each position when sampling candidates based on gradients.
            sample_n_replace: Number of token positions to replace when generating each candidate.
            candidate_oversample_factor: Generate n_candidates * factor candidates, then
                truncate after retokenization filtering. Only effective when > 1.0.
            n_candidates_after_proxy_filter: If set, filter candidates down to top-K by proxy loss
                before evaluating on the target (PAL's proxy filtering step).
                Expected to be less than n_candidates. Set to None to disable.
        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)

        # Proxy model setup
        self.proxy_model = model if proxy_model is None else proxy_model
        assert isinstance(self.proxy_model, LossTokenAccessMixin), (
            "proxy_model must support LossTokenAccessMixin for candidate selection"
        )
        if candidate_selection == "gradient":
            assert isinstance(self.proxy_model, GradientTokenAccessMixin), (
                f"{candidate_selection=} requires proxy_model with GradientTokenAccessMixin"
            )

        # Prefer token-level target evaluation when proxy and target share the same tokenizer
        # [TODO more informative name to use_token_eval; it should also reflect the its compute-loss and on the target model]
        use_token_eval = (model.tokenizer == self.proxy_model.tokenizer) and isinstance(model, LossTokenAccessMixin)   # TODO find a more accurate way of comparing tokenizers

        if model == proxy_model:
            n_candidates_after_proxy_filter = None  # disable proxy filtering if proxy and target are the same

        # Save params
        self.candidate_selection = candidate_selection
        self.num_steps = num_steps
        self.n_candidates = n_candidates
        self.n_candidates_after_proxy_filter = n_candidates_after_proxy_filter
        self.sample_topk = sample_topk
        self.sample_n_replace = sample_n_replace
        self.token_constraints = token_constraints
        self.candidate_oversample_factor = candidate_oversample_factor
        self.use_token_eval = use_token_eval

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:
        self._log_run_config_to_tracker(templates, initial_trigger, targets)

        # --- Initialization ---
        proxy_model = self.proxy_model
        proxy_tokenizer = proxy_model.tokenizer
        target_model = self.model

        # Proxy uses token access:
        proxy_model.set_inputs_from_tokens(templates=templates, targets=targets)

        # Target model access depends on `use_token_eval`:
        if not self.use_token_eval:
            target_model.set_inputs_from_texts(templates=templates, targets=targets)
        else:
            target_model.set_inputs_from_tokens(templates=templates, targets=targets)

        trigger_ids: Int[Tensor, "trigger_seq_len"] = proxy_tokenizer.encode_trigger(initial_trigger).to(proxy_model.device)

        vocab_size = proxy_model.vocab_size

        blacklist_ids = self.token_constraints.get_blacklist_ids(proxy_tokenizer, vocab_size)
        valid_token_ids = self.token_constraints.get_whitelist_ids(proxy_tokenizer, vocab_size, proxy_model.device, return_tensor=True)

        best = RunningBest()
        visited: Set[str] = set()  # track visited trigger strings to avoid re-eval

        # Number of candidates to sample (oversample if retokenize is on)
        n_candidates_oversampled = math.ceil(self.n_candidates * self.candidate_oversample_factor)

        # Initial loss
        current_loss = self._evaluate_candidates_on_target_model(
            trigger_ids.unsqueeze(0)
        ).item()
        trigger_str = proxy_tokenizer.decode_trigger(trigger_ids)
        visited.add(trigger_str)
        self.log(loss=current_loss, trigger_str=trigger_str)

        pbar = tqdm(range(self.num_steps))

        for _ in pbar:
            # === Stage 1: Candidate Selection (on proxy) ===

            if self.candidate_selection == "gradient":
                trigger_grad = proxy_model.compute_grad_from_tokens(
                    candidate_trigger_ids=trigger_ids.unsqueeze(0),
                    loss_func=self.loss_func,
                    normalize_grads=True,
                ).squeeze(0)

                candidate_trigger_ids = self._sample_ids_from_grad(
                    trigger_ids=trigger_ids,
                    trigger_grad=trigger_grad,
                    blacklist_ids=blacklist_ids,
                    _n_candidates=n_candidates_oversampled,
                )
            else:  # "random"
                candidate_trigger_ids = self._sample_ids_at_random(
                    trigger_ids=trigger_ids,
                    valid_token_ids=valid_token_ids,
                    _n_candidates=n_candidates_oversampled,
                )

            # === Filtering ===
            candidate_trigger_ids = retokenize_filtering(
                candidate_trigger_ids, proxy_tokenizer
            )
            # Skip visited (always on)
            candidate_trigger_ids = self._filter_visited(
                candidate_trigger_ids, visited
            )
            # Truncate to n_candidates (if longer after oversample + filter)
            candidate_trigger_ids = candidate_trigger_ids[: self.n_candidates]

            if len(candidate_trigger_ids) == 0:
                logger.warning("Filtered out all candidates (after tokenization and visited filtering), skipping step.")
                continue

            # === Proxy filtering ===
            # PAL adds another step to further narrow the candidate list
            if self.n_candidates_after_proxy_filter is not None:
                proxy_losses = proxy_model.compute_loss_from_tokens(
                    candidate_trigger_ids, loss_func=self.loss_func
                )  # shape: (n_candidates,)
                topk_indices = proxy_losses.topk(
                    self.n_candidates_after_proxy_filter, largest=False
                ).indices
                candidate_trigger_ids = candidate_trigger_ids[topk_indices]

            # === Stage 2: Candidate Evaluation (on target) ===
            losses = self._evaluate_candidates_on_target_model(candidate_trigger_ids)

            current_loss = losses.min().item()
            trigger_ids = candidate_trigger_ids[losses.argmin()]
            trigger_str = proxy_tokenizer.decode_trigger(trigger_ids)

            visited.add(trigger_str)

            best.update(loss=current_loss, trigger_ids=trigger_ids, trigger_str=trigger_str)
            self.log(loss=current_loss, trigger_str=trigger_str)
            pbar.set_description(f"loss={current_loss: .4f}, trigger={trigger_str}")

        # --- Finalize ---
        result = OptimizerResult(
            best_loss=best.loss,
            best_trigger_str=best.trigger_str,
            best_trigger_ids=best.trigger_ids,
            losses=best.losses,
            trigger_strs=best.trigger_strs,
        )
        self.tracker.log({"best_loss": result.best_loss, "best_trigger_str": result.best_trigger_str})

        proxy_model.reset_inputs_from_tokens()
        target_model.reset_inputs_from_texts()
        target_model.reset_inputs_from_tokens()

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

    def _sample_ids_from_grad(
        self,
        trigger_ids: Int[Tensor, "trigger_seq_len"],
        trigger_grad: Float[Tensor, "trigger_seq_len vocab_size"],
        blacklist_ids: list,
        _n_candidates: int,
    ) -> Int[Tensor, "n_candidates trigger_seq_len"]:
        """
        Follows GCG candidate sampling method.
        - `_n_candidates` may be an oversampled number of triggers (ie possibly larger than `self.n_candidates`), which will be truncated after retokenization filtering.
        """
        trigger_seq_len = trigger_grad.shape[0]
        device = trigger_grad.device

        trigger_grad = trigger_grad.clone()
        trigger_grad *= -1
        trigger_grad[:, blacklist_ids] = float("-inf")
        topk_ids = trigger_grad.topk(self.sample_topk, dim=-1).indices

        candidate_trigger_ids = trigger_ids.repeat(_n_candidates, 1).clone()

        sampled_ids_pos = torch.rand(
            _n_candidates, trigger_seq_len, device=device
        ).argsort(dim=-1)[..., : self.sample_n_replace]

        relevant_topk_lists = topk_ids[sampled_ids_pos]
        rand_k_indices = torch.randint(
            0,
            self.sample_topk,
            (_n_candidates, self.sample_n_replace, 1),
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

    def _sample_ids_at_random(
        self,
        trigger_ids: Int[Tensor, "trigger_seq_len"],
        valid_token_ids: Int[Tensor, "n_valid"],
        _n_candidates: int,
    ) -> Int[Tensor, "n_candidates_oversampled trigger_seq_len"]:
        """Sample candidates by randomly replacing tokens (RAL-style)."""
        trigger_seq_len = trigger_ids.shape[0]
        device = trigger_ids.device

        candidate_trigger_ids = trigger_ids.repeat(_n_candidates, 1).clone()

        sampled_ids_pos = torch.rand(
            _n_candidates, trigger_seq_len, device=device
        ).argsort(dim=-1)[..., : self.sample_n_replace]

        rand_indices = torch.randint(
            0, len(valid_token_ids), (_n_candidates, self.sample_n_replace), device=device
        )
        random_tokens = valid_token_ids[rand_indices]

        candidate_trigger_ids = candidate_trigger_ids.scatter_(
            dim=-1,
            index=sampled_ids_pos,
            src=random_tokens,
        )
        return candidate_trigger_ids

    def _filter_visited(
        self,
        candidate_trigger_ids: Int[Tensor, "n_candidates_oversampled trigger_seq_len"],
        visited: Set[str],
    ) -> Int[Tensor, "n_filtered trigger_seq_len"]:
        """Filter out candidates whose decoded string is in the visited set."""
        tokenizer = self.proxy_model.tokenizer
        keep_mask = torch.tensor(
            [tokenizer.decode_trigger(cid) not in visited for cid in candidate_trigger_ids],
            device=candidate_trigger_ids.device,
        )
        return candidate_trigger_ids[keep_mask]
