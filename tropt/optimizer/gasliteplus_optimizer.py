import logging
import time
from typing import Optional

import torch
from jaxtyping import Float, Int
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
from tropt.optimizer.utils.buffer import TriggerBuffer
from tropt.optimizer.utils.retokenization import retokenize_filtering
from tropt.optimizer.utils.running_best import RunningBest
from tropt.optimizer.utils.scheduler import (
    ConstantScheduler,
    LinearScheduler,
    NFlipScheduler,
)
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.optimizer.utils.token_initializers import get_printable_random_trigger
from tropt.tracker import BaseTracker

logger = logging.getLogger(__name__)


class GASLITEPlusOptimizer(BaseOptimizer):
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

        buffer_size: int = 10,
        decline_n_flip_from_step: Optional[int | float] = None,
        early_stopping_patience: Optional[int] = None,
        early_stopping_threshold: float = 0.005,  # relative improvement threshold

        n_bulk_flips: int = 5,
        flip_pos_method: str = "random",  # "random" or "ordered"
        time_limit: Optional[float] = None,
        n_flip_scheduler: Optional[NFlipScheduler] = None,
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

            use_random_gradient (bool): If True, uses random gradients instead of model gradients (for ablation).
            buffer_size (int): Size of the trigger buffer to maintain.

            decline_n_flip_from_step (int | float, optional): If set, linearly declines `n_flip` to 1
                starting from this step (int) or fraction of total steps (float).

            early_stopping_patience (int, optional): If set, enables early stopping if no improvement
                is seen in the buffer for this many consecutive steps.
            early_stopping_threshold (float): Relative improvement threshold for early stopping.

            n_bulk_flips (int): Number of bulk flips to perform per step (lower => less sequential model calls, faster).

            flip_pos_method (str): Method to select positions to flip - "random" or "ordered".

            n_flip_scheduler (NFlipScheduler, optional): A scheduler object to control `n_flip`.
                If provided, overrides `decline_n_flip_from_step`.

        References:
        - GASLITE: https://arxiv.org/abs/2412.20953
            It is based on the GASLITE algorithm proposed in the paper, and extends it with
            multiple enhancements.
        - ACG: https://www.haizelabs.com/blog/making-a-sota-adversarial-attack-on-llms-38x-faster
            GASLITEPlus implements (i) multiple trigger random intiizliation, (ii) trigger buffer,
            and (iii) flipping bulk of positions at once, (iv) early stopping. Thus, it effectively
            includes most of the enhancements described Haize's ACG.
        - QCG? PAL? RAL?
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

        self.buffer_size = buffer_size
        self.decline_n_flip_from_step = decline_n_flip_from_step

        # early stopping params
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold  # relative improvement threshold
        self.n_bulk_flips = n_bulk_flips
        self.flip_pos_method = flip_pos_method
        self.time_limit = time_limit

        if n_flip_scheduler is not None:
            self.n_flip_scheduler = n_flip_scheduler
        elif decline_n_flip_from_step is not None:
            self.n_flip_scheduler = LinearScheduler(
                initial_n_flip=n_flip,
                total_steps=num_steps,
                decline_start=decline_n_flip_from_step
            )
        else:
            # default: constant n_flip
            self.n_flip_scheduler = ConstantScheduler(n_flip)

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
        trigger_ids: Int[Tensor, "trigger_seq_len"] = tokenizer.encode_trigger(initial_trigger).to(self.model.device)
        vocab_size = self.model.vocab_size
        blacklist_ids = self.token_constraints.get_blacklist_ids(tokenizer, vocab_size)

        token_mask = torch.ones(vocab_size, device=self.model.device, dtype=torch.bool)
        token_mask[blacklist_ids] = False
        valid_token_ids = token_mask.nonzero(as_tuple=False).squeeze(-1)

        trigger_ids: Float[Tensor, "trigger_seq_len"] = trigger_ids.to(self.model.device)
        trigger_seq_len = len(trigger_ids)
        trigger_str = initial_trigger

        best = RunningBest()
        current_loss = float("inf")

        start_time = time.time()
        pbar = tqdm(range(self.num_steps), desc="Optimizing with GASLITE...")

        # Form buffer_size initial triggers
        triggers_for_buffer = [trigger_ids]
        for _ in range(self.buffer_size - 1):
            random_trigger_ids = get_printable_random_trigger(
                trigger_seq_len, tokenizer=tokenizer, return_ids=True
            ).to(self.model.device)
            triggers_for_buffer.append(random_trigger_ids)

        # Compute losses for initial triggers
        losses = self.model.compute_loss_from_tokens(
            torch.stack(triggers_for_buffer),
            self.loss_func,
        ) # (n_cands,)

        # Create the buffer:
        buffer = TriggerBuffer(
            triggers=[triggers_for_buffer[i] for i in range(self.buffer_size)],
            losses=[losses[i].item() for i in range(self.buffer_size)],
        )

        trigger_str = tokenizer.decode(buffer.get_best_trigger(), skip_special_tokens=True)
        self.tracker.log({"loss": buffer.get_lowest_loss(), "trigger_str": trigger_str})

        for step in pbar:
            n_flip = self.n_flip_scheduler.get_n_flip(step)

            pbar.set_description(
                f"Step {step+1}/{self.num_steps} | loss={current_loss: .4f} | trigger={trigger_str}..."
            )

            # Get the best trigger from the buffer
            trigger_ids = buffer.get_best_trigger()
            trigger_str = tokenizer.decode_trigger(trigger_ids)

            # --- Gradient and candidate selection step ---
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
                trigger_grad = -trigger_grad  # we want to minimize the loss

            # Get Top-k Candidates *per position*
            trigger_grad[:, blacklist_ids] = float("-inf")
            topk_ids: Float[Tensor, "trigger_seq_len n_candidates"]
            topk_ids = trigger_grad.topk(self.n_candidates, dim=-1).indices

            # --- Greedy coordinate ascent step ---
            current_trigger_ids = trigger_ids.clone()
            # Sample `n_flip` unique positions to optimize
            sampled_positions = torch.randperm(trigger_seq_len, device=self.model.device)[: n_flip]
            if self.flip_pos_method == "ordered":
                sampled_positions, _ = sampled_positions.sort()

            # Perform bulk flips in chunks
            bulk_pos_list = torch.chunk(sampled_positions, self.n_bulk_flips)
            for bulk_pos in bulk_pos_list:
                candidate_triggers = current_trigger_ids.repeat(self.n_candidates, 1)

                # Inject candidate tokens at all positions in the bulk
                for pos in bulk_pos:
                    # Get candidate tokens for this position
                    all_candidate_tokens = torch.cat([
                        current_trigger_ids[pos].unsqueeze(0),  # keep the "no flip" option
                        topk_ids[pos, :self.n_candidates // 2],
                    ])
                    if len(bulk_pos) > 1:
                        # If the bulk has multiple positions, we add more candidates to increase diversity
                        # sample more token ids, with replacements
                        more_cand_indices = torch.randint(
                            high=topk_ids[pos].size(0),
                            size=(self.n_candidates - len(all_candidate_tokens),),
                            device=topk_ids.device
                        )
                    else:
                        more_cand_indices = torch.arange(
                            self.n_candidates // 2,
                            self.n_candidates // 2 + (self.n_candidates - len(all_candidate_tokens)),
                            device=topk_ids.device
                        )
                    all_candidate_tokens = torch.cat([
                        all_candidate_tokens,
                        topk_ids[pos, more_cand_indices]
                    ])

                    # Create all candidate triggers by flipping this *single* position
                    candidate_triggers[:, pos] = all_candidate_tokens

                # keep only unique candidates
                candidate_triggers = torch.unique(candidate_triggers, dim=0)

                # (Optional) Retokenize filtering
                if self.use_retokenize:
                    candidate_triggers = retokenize_filtering(
                        candidate_triggers, tokenizer
                    )
                    if len(candidate_triggers) == 0:
                        logger.warning(
                            f"Retokenize filtering removed all candidates for pos {pos}. Skipping step."
                        )
                        continue  # Keep `current_trigger_ids` as is for this position

                # Compute losses on candidate flips
                losses = self.model.compute_loss_from_tokens(
                    candidate_triggers,
                    self.loss_func,
                    keep_message_dim=True,  # Get per-message loss
                ).mean(
                    dim=0
                )  # Average over messages -> (n_cands,)

                # Find the best token for this position
                losses_sorted_indices = torch.argsort(losses)
                best_candidate_idx = losses_sorted_indices[0]

                # Update `current_trigger_ids` *in-place* for the next iteration of the greedy (inner) loop
                current_trigger_ids = candidate_triggers[best_candidate_idx].clone()
                current_loss = losses[best_candidate_idx].item()

                # Update the trigger buffer, how much needed
                # We go over the buffer-size best candidates and try to add them to the buffer
                for j in range(min(buffer.size, len(losses))):
                    cand_idx = losses_sorted_indices[j]
                    buffer.add_if_better(
                        candidate_triggers[cand_idx].clone(),
                        losses[cand_idx].item(),
                    )

            # After the inner loop, `current_trigger_ids` is the best trigger for this *entire* step
            trigger_ids = current_trigger_ids
            trigger_str = tokenizer.decode_trigger(trigger_ids)

            # Logging:
            self.log(loss=current_loss, trigger_str=trigger_str)
            best.update(loss=current_loss, trigger_ids=trigger_ids, trigger_str=trigger_str)

            if self.time_limit is not None:
                if time.time() - start_time > self.time_limit:
                    logger.info(f"Time limit of {self.time_limit}s reached. Stopping optimization.")
                    break

            # (Optional) Early stopping if no improvement in the buffer
            if self.early_stopping_patience is not None:
                if step == 0:
                    best_loss_global = current_loss
                    steps_without_improvement = 0

                # define the relative improvement
                denominator = abs(best_loss_global) if best_loss_global != 0 else 1.0
                relative_improvement = (best_loss_global - current_loss) / denominator

                # Check if improvement is greater than the relative threshold
                if relative_improvement > self.early_stopping_threshold:
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1

                if steps_without_improvement >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at step {step+1}. No relative improvement (of > {self.early_stopping_threshold*100:.2%}) in the last {self.early_stopping_patience} steps.")
                    break

                # Update the best loss globally
                best_loss_global = min(best_loss_global, current_loss)


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

