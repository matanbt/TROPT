import json
import logging
import re
from typing import List, Optional

import numpy as np
import torch
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from tqdm.auto import tqdm

from tropt.common import (
    OPTIMIZED_TRIGGER_PLACEHOLDER,
    Targets,
    TextTemplates,
    DEFAULT_INIT_TRIGGER,
)
from tropt.loss.base import BaseLoss
from tropt.model import (
    BaseModel,
    LossTextAccessMixin,
    TokenAccessMixin,
    EncoderBaseModel,
)
from tropt.optimizer.base import BaseOptimizer, OptimizerResult
from tropt.tracker.base import BaseTracker

logger = logging.getLogger(__name__)


class CombiOptimizer(BaseOptimizer):
    """
    Implements the Combination Attack optimizer

    """

    # TODO test against LLM
    model_requirements = (LossTextAccessMixin, TokenAccessMixin, EncoderBaseModel)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # attack parameters:
        hot_start: bool = True,
        total_tokens: int = 100,
        random_num_pool: int = 500,
        random_early_stop_patience: int = 5,
        square_p_init: float = 0.5,
        square_num_iters: int = 2000,
        square_random_pool: int = 300,
        square_early_stop_patience: int = 100,
        **kwargs,
    ):
        """
        Initializes the Combination Attack Optimizer.

        Args:
            model (HuggingFaceModel): The model to be attacked.
            loss (BaseLoss): The loss function to be optimized.
            seed (int, optional): Random seed for reproducibility.

            hot_start (bool): Whether to try and employ hot-start.
            sim (str): Similarity function to employ.
            total_tokens (int): How many tokens should be in the prefix.

            random_num_pool (int): How many tokens to generate on each iteration of random attack.
            random_early_stop_patience (int): Stop random attack and move on to square attack if no improvement for this many iterations.

            square_p_init (float): Initial fraction of tokens replaced in one update (analogous to image pixel fraction).
            square_num_iters (int): Maximum number of iterations.
            square_random_pool (int): For every position in the chosen block we sample uniformly that many candidate tokens. Higher => more diversity.
            square_early_stop_patience (int): Stop if no improvement for this many iterations.

        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)
        self.model = model
        self.loss = loss
        self.tracker = tracker
        self.seed = seed

        # save params:
        self.hot_start = hot_start
        self.total_tokens = total_tokens

        self.random_num_pool = random_num_pool
        self.random_early_stop_patience = random_early_stop_patience

        self.square_p_init = square_p_init
        self.square_num_iters = square_num_iters
        self.square_random_pool = square_random_pool
        self.square_early_stop_patience = square_early_stop_patience

        self.openai_client: Optional[OpenAI] = None

    # TODO allow running just random attack/square attack
    def _random_attack(
        self,
        pbar: tqdm,
        history: List[dict],
        hot_start_str: Optional[str] = None,
        best_sim: Optional[float] = None,
    ):
        """
        Executes the initial random search phase of the attack.

        This method iteratively constructs a trigger by appending tokens one by one (greedy approach).
        At each step, it samples a pool of random candidates (`random_num_pool`) from the vocabulary,
        evaluates them, and selects the token that maximizes the similarity score
        (minimizes loss). It stops early if no improvement is seen for `random_early_stop_patience`
        steps or if `best_sim` is reached.
        """
        curr_p = ""
        tokens = []

        if hot_start_str is not None:
            curr_p = hot_start_str
            tokens = self.model.tokenizer.encode(
                hot_start_str, add_special_tokens=False
            )

        base_score = -self.model.compute_loss_from_texts(
            candidate_trigger_strs=[curr_p],
            loss_func=self.loss_func,
        ).item()
        no_improve = 0
        # print(f"initial similarity: {base_score}")

        iter_best_score = 0
        valid_vocab_ids = self._get_valid_vocab_ids()

        self._update_history(
            pbar=pbar,
            history=history,
            best_score=base_score,
            trigger_str=curr_p,
            trigger_ids=tokens,
        )

        if len(tokens):
            pbar.update(len(tokens))

        # Iteratively append tokens one by one to reach total_tokens. This is a greedy approach:
        # once a token is selected for position N, it is fixed, and we optimize position N+1.
        for n in range(self.total_tokens - len(tokens)):
            # print(f"iteration {n + 1}")
            pool = np.random.choice(valid_vocab_ids, size=(self.random_num_pool,))

            # compute current baseline similarity for this iteration
            iter_best_score = -self.model.compute_loss_from_texts(
                candidate_trigger_strs=[curr_p],
                loss_func=self.loss_func,
            ).item()
            best_id = None
            best_token = None

            # We construct prompts, each with a different candidate token appended.
            batch_tokens = self.model.tokenizer.batch_decode(pool)
            check_ps = [curr_p + " " + t for t in batch_tokens]

            # loss against q_emb
            losses = self.model.compute_loss_from_texts(
                candidate_trigger_strs=check_ps,
                loss_func=self.loss_func,
            )
            min_loss, min_idx = torch.min(losses, dim=0)
            prop_best_score = -min_loss.item()
            best_idx = min_idx.item()

            # Update only if loss improved
            if prop_best_score > iter_best_score:
                iter_best_score = prop_best_score
                best_id = pool[best_idx].item()
                best_token = batch_tokens[best_idx]

            if best_token is not None:
                tokens.append(best_id)
                curr_p += " " + best_token
                no_improve = 0
            else:
                no_improve += 1

            self._update_history(
                pbar=pbar,
                history=history,
                best_score=iter_best_score,
                trigger_str=curr_p,
                trigger_ids=tokens,
            )

            if (
                self.random_early_stop_patience is not None
                and no_improve >= self.random_early_stop_patience
            ):
                pbar.update(self.total_tokens - len(tokens))
                break
            if best_sim is not None and iter_best_score > best_sim:
                break

        # print(f"final similarity: {iter_best_score}")

    def _square_attack(
        self, pbar: tqdm, history: List[dict], best_sim: Optional[float] = None
    ):
        """A 1D adaptation of the image Square Attack for token sequence (prompt) optimization.

        Instead of square patches over image pixels, we sample contiguous blocks ("windows")
        over the sequence of appended tokens and refresh the entire block with randomly
        sampled vocabulary tokens. A proposal is accepted if it increases cosine similarity.

        The size of the block changes over iterations based on a schedule (`_p_selection`).

        Raises:
            ValueError: If `total_tokens` is not positive.
        """
        initial_tokens = history[-1]["trigger_ids"] if len(history) else []

        valid_vocab_ids = self._get_valid_vocab_ids()
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be > 0")

        # Initialize appended tokens. If the trigger from the previous phase is shorter
        # than total_tokens, fill the remaining slots. We try to repeat the existing pattern
        # (cyclic repetition), otherwise fall back to random tokens.
        appended_tokens = []
        if initial_tokens:
            appended_tokens = list(initial_tokens)

        for i in range(len(appended_tokens), self.total_tokens):
            if initial_tokens and len(initial_tokens) != 0:
                appended_tokens.append(initial_tokens[i % len(initial_tokens)])
            else:
                tok_id = np.random.choice(valid_vocab_ids).item()
                tok = self.model.tokenizer.decode(tok_id)
                appended_tokens.append(tok)

        def build_prompt(tokens_list):
            return self.model.tokenizer.decode(tokens_list, skip_special_tokens=True)

        current_prompt = build_prompt(appended_tokens)
        with torch.no_grad():
            curr_score = -self.model.compute_loss_from_texts(
                candidate_trigger_strs=[current_prompt],
                loss_func=self.loss_func,
            ).item()

        self._update_history(
            pbar=pbar,
            history=history,
            best_score=curr_score,
            trigger_str=current_prompt,
            trigger_ids=appended_tokens,
        )
        best_tokens = list(appended_tokens)
        no_improve = 0

        # Main Square Attack loop: Refine the existing trigger by perturbing blocks of tokens.
        for it in range(1, self.square_num_iters + 1):
            # Calculate the size of the window to perturb.
            # Early iterations perturb large blocks (exploration); later iterations fine-tune small blocks (exploitation).
            p = self._p_selection(it - 1)
            block_size = max(1, round(p * self.total_tokens))
            block_size = min(block_size, self.total_tokens)

            start = np.random.randint(0, self.total_tokens - block_size + 1)
            end = start + block_size

            # Generate a pool of candidate triggers (proposals).
            # Each proposal takes the current best trigger and randomizes the tokens
            # within the [start, end] window.
            proposals_tokens = []
            num_proposals = max(1, self.square_random_pool)
            for _ in range(num_proposals):
                proposal = list(best_tokens)
                for pos in range(start, end):
                    proposal[pos] = np.random.choice(valid_vocab_ids).item()
                proposals_tokens.append(proposal)

            # Build all candidate prompts and evaluate in batches
            batch_prompts = [build_prompt(toks) for toks in proposals_tokens]
            with torch.no_grad():
                losses = self.model.compute_loss_from_texts(
                    candidate_trigger_strs=batch_prompts,
                    loss_func=self.loss_func,
                )
                min_loss, min_idx = torch.min(losses, dim=0)
                prop_best_score = -min_loss.item()
                best_idx = min_idx.item()

            # Update only if loss improved
            if prop_best_score > curr_score:
                curr_score = prop_best_score
                best_tokens = proposals_tokens[best_idx]
                current_prompt = batch_prompts[best_idx]
                no_improve = 0
            else:
                no_improve += 1

            self._update_history(
                pbar=pbar,
                history=history,
                best_score=curr_score,
                trigger_str=current_prompt,
                trigger_ids=best_tokens,
            )

            # Early stopping
            if (
                self.square_early_stop_patience
                and no_improve >= self.square_early_stop_patience
            ):
                break
            if best_sim is not None and curr_score > best_sim:
                break

    def optimize_trigger(
        self,
        templates: TextTemplates,
        initial_trigger: Optional[str] = DEFAULT_INIT_TRIGGER,
        targets: Optional[Targets] = None,
    ) -> OptimizerResult:
        """
        Runs the full optimization process combining random search and Square Attack.

        1. Prepares inputs and optionally initializes with a 'hot start' trigger generated via LLM.
        2. Runs `random_attack` to greedily build an initial trigger token by token.
        3. Runs `square_attack` to refine the trigger by perturbing blocks of tokens.

        Args:
           templates (List[str]): The input texts/prompts to attack.
           initial_trigger (Optional[str]): A starting trigger string (used if hot_start is False or fails).
           targets (TargetsDict): Target configuration for the loss calculation.

        Returns:
            OptimizerResult: An object containing the best loss, best trigger string, and history of optimization.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model.set_inputs_from_texts(templates=templates, targets=targets)

        # TODO hot start without target texts?
        hot_start_str = None
        if (
            targets.target_texts is not None
            and len(targets.target_texts) > 0
            and self.hot_start
        ):
            _, _, hot_start_str = self._generate_hot_start(targets.target_texts[0])

        best_sim = None
        if (
            targets.target_similarities is not None
            and len(targets.target_similarities) > 0
        ):
            best_sim = targets.target_similarities[0]

        pbar = tqdm(total=self.total_tokens + self.square_num_iters, unit=" steps")
        history = []

        self._random_attack(
            pbar=pbar, history=history, hot_start_str=hot_start_str, best_sim=best_sim
        )

        if best_sim is None or history[-1]["best_score"] <= best_sim:
            self._square_attack(pbar=pbar, history=history, best_sim=best_sim)

        result = OptimizerResult(
            best_loss=-history[-1]["best_score"],
            best_trigger_str=history[-1]["trigger_str"],
            best_trigger_ids=history[-1]["trigger_ids"],
            trigger_strs=[x["trigger_str"] for x in history],
            losses=[-x["best_score"] for x in history],
            full_prompt=[
                t.replace(OPTIMIZED_TRIGGER_PLACEHOLDER, history[-1]["trigger_str"])
                for t in templates
            ],
        )
        return result

    def _p_selection(self, it: int) -> float:
        """
        Piece-wise constant schedule for p (re-used from original Square Attack).

        Calculates the fraction of the sequence to mutate based on the current iteration number.
        The mutation size decreases as the attack progresses (fine-tuning phase).
        """
        p_init = self.square_p_init
        scaled = int(it / self.square_num_iters * 10000)
        # Mirrors original schedule thresholds.
        if 10 < scaled <= 50:
            return p_init / 2
        elif 50 < scaled <= 200:
            return p_init / 4
        elif 200 < scaled <= 500:
            return p_init / 8
        elif 500 < scaled <= 1000:
            return p_init / 16
        elif 1000 < scaled <= 2000:
            return p_init / 32
        elif 2000 < scaled <= 4000:
            return p_init / 64
        elif 4000 < scaled <= 6000:
            return p_init / 128
        elif 6000 < scaled <= 8000:
            return p_init / 256
        elif 8000 < scaled <= 10000:
            return p_init / 512
        else:
            return p_init

    def _get_valid_vocab_ids(self):
        if hasattr(self, "_valid_vocab_ids"):
            return self._valid_vocab_ids

        vocab = self.model.tokenizer.get_vocab()
        valid_ids = []
        english_pattern = re.compile(r"^[a-zA-Z]+$")

        for token, token_id in vocab.items():
            # 1. Check the regex pattern
            if english_pattern.match(token):
                # 2. Decode to ensure it's not an empty/whitespace or special token
                if self.model.tokenizer.decode(
                    [token_id], skip_special_tokens=True
                ).strip():
                    valid_ids.append(token_id)

        self._valid_vocab_ids = np.array(valid_ids)
        return self._valid_vocab_ids

    def _generate_hot_start(self, target_text: str):
        """
        Generates an initial trigger candidates using an external LLM (OpenAI).

        It queries the LLM to generate a short sentence related to the `target_text`.
        This serves as a better starting point than random tokens.
        Results are cached from `cached_responses.json` to avoid redundant API calls.

        Returns:
            tuple: (text, tokens, decoded_str)
        """
        # Check for already-calculated response to save LLM API costs and time
        text = None
        try:
            with open("cached_responses.json", "r") as f:
                cached = json.load(f)
            if target_text in cached:
                text = cached[target_text]
        except FileNotFoundError:
            pass

        if text is None:
            if self.openai_client is None:
                self.openai_client = OpenAI()

            messages: list[ChatCompletionUserMessageParam] = [
                {
                    "role": "user",
                    "content": f"Return a short sentence, up to 15 words, related to the following passage: {target_text}",
                },
            ]
            response = self.openai_client.chat.completions.create(
                model="gpt-5-nano",
                messages=messages,
                # max_completion_tokens=100,  # This seems to mess up the completion
            )
            text = response.choices[0].message.content.rstrip(".")

        tokens = self.model.tokenizer.encode(text)
        decoded: str = self.model.tokenizer.decode(tokens, skip_special_tokens=True)
        return text, tokens, decoded

    def _update_history(
        self,
        pbar: tqdm,
        history: List[dict],
        best_score: float,
        trigger_str: str,
        trigger_ids: List[int],
        step: int = None,
    ):
        """
        Logs the current step's best results to the history list and updates the progress bar.

        Args:
            best_score (float): The current best similarity/loss score.
            trigger_str (str): The current best trigger string.
            trigger_ids (List[int]): The token IDs of the trigger.
            step (int, optional): Explicit step number. Defaults to incrementing previous step.
        """
        last_step = history[-1]["step"] if len(history) else -1
        history.append(
            {
                "step": step if step is not None else last_step + 1,
                "best_score": best_score,
                "num_tokens": self.model.get_usage_stats()["total_tokens"],
                "num_api": self.model.get_usage_stats()["forward_calls"],
                "trigger_str": trigger_str,
                "trigger_ids": trigger_ids,
            }
        )

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                {
                    "similarity": f"{history[-1]["best_score"]:.5f}",
                    "num_tokens": f"{history[-1]["num_tokens"]}",
                }
            )

        if self.tracker is not None:
            self.tracker.log(
                {
                    "loss": -history[-1]["best_score"],
                    **self.model.get_usage_stats(),
                }
            )
