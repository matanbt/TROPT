import logging
import re
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from tqdm.auto import tqdm

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER
from tropt.loss.base import BaseLoss
from tropt.models import BaseModel, TargetsDict, LossTextAccessMixin, TokenAccessMixin
from tropt.optimizer.base import BaseOptimizer, OptimizerResult
from tropt.tracker.base import BaseTracker

logger = logging.getLogger(__name__)


class CombiOptimizer(BaseOptimizer):
    """
    Implements the Combination Attack optimizer

    """

    model_requirements = (LossTextAccessMixin, TokenAccessMixin)

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # attack parameters:
        hot_start: bool = True,
        batch_size: int = 128,
        best_sim: Optional[float] = 0.9,
        total_tokens: int = 100,
        random_num_pool: int = 500,
        random_early_stop_patience: int = 5,
        square_p_init: float = 0.5,
        square_num_iters: int = 2000,
        square_random_pool_per_pos: int = 300,
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
            batch_size (int): Batch size for embedding.
            sim (str): Similarity function to employ.
            best_sim (float, optional): Similarity of best passage to try and bypass. If reached, optimizer stops.
            total_tokens (int): How many tokens should be in the prefix.

            random_num_pool (int): How many tokens to generate on each iteration of random attack.
            random_early_stop_patience (int): Stop random attack and move on to square attack if no improvement for this many iterations.

            square_p_init (float): Initial fraction of tokens replaced in one update (analogous to image pixel fraction).
            square_num_iters (int): Maximum number of iterations.
            square_random_pool_per_pos (int): For every position in the chosen block we sample uniformly that many candidate tokens and pick 1 (simple random draw). Higher => more diversity.
            square_early_stop_patience (int): Stop if no improvement for this many iterations.

        """
        super().__init__(model, loss=loss, tracker=tracker, seed=seed)
        self.model = model
        self.loss = loss
        self.tracker = tracker
        self.seed = seed

        # save params:
        self.hot_start = hot_start
        self.hot_start_str = None
        self.batch_size = batch_size
        self.best_sim = best_sim
        self.total_tokens = total_tokens

        self.random_num_pool = random_num_pool
        self.random_early_stop_patience = random_early_stop_patience

        self.square_p_init = square_p_init
        self.square_num_iters = square_num_iters
        self.square_random_pool_per_pos = square_random_pool_per_pos
        self.square_early_stop_patience = square_early_stop_patience

        self.openai_client = None
        self.target_text = None

        self.pbar = None
        self.history = []

    def random_attack(
        self,
        inputs,
    ):
        curr_p = ""
        tokens = []

        if self.hot_start_str is not None:
            curr_p = self.hot_start_str
            tokens = self.model.tokenizer.encode(
                self.hot_start_str, add_special_tokens=False
            )

        base_sim = float(
            -self.model.compute_loss_from_texts(
                candidate_trigger_strs=[curr_p],
                inputs=inputs,
                loss_func=self.loss_func,
            )[0]
        )
        no_improve = 0
        # print(f"initial similarity: {base_sim}")

        iter_best_score = 0
        valid_vocab_ids = self._get_valid_vocab_ids()

        self._update_history(best_score=base_sim, trigger_str=curr_p, trigger=tokens)

        if len(tokens):
            self.pbar.update(len(tokens))

        for n in range(self.total_tokens - len(tokens)):
            # print(f"iteration {n + 1}")
            pool = np.random.choice(valid_vocab_ids, size=(self.random_num_pool,))

            # compute current baseline similarity for this iteration
            iter_best_score = float(
                -self.model.compute_loss_from_texts(
                    candidate_trigger_strs=[curr_p],
                    inputs=inputs,
                    loss_func=self.loss_func,
                )[0]
            )
            best_id = None
            best_token = None

            # evaluate candidates in parallel by batching
            for i in range(0, len(pool), self.batch_size):
                batch_ids = [
                    [int(pool[i])]
                    for i in range(i, min(i + self.batch_size, len(pool)))
                ]
                batch_tokens = self.model.tokenizer.batch_decode(batch_ids)

                # build candidate prompts
                check_ps = [curr_p + " " + t for t in batch_tokens]

                # vectorized cosine similarity against q_emb
                losses = self.model.compute_loss_from_texts(
                    candidate_trigger_strs=check_ps,
                    inputs=inputs,
                    loss_func=self.loss_func,
                )
                min_loss, min_idx = torch.min(losses, dim=0)
                prop_best_sim = float(-min_loss.item())
                best_idx = int(min_idx.item())
                if prop_best_sim > iter_best_score:
                    iter_best_score = prop_best_sim
                    best_id = int(batch_ids[best_idx][0])
                    best_token = batch_tokens[best_idx]

            if best_token is not None:
                tokens.append(best_id)
                curr_p += " " + best_token
                no_improve = 0
            else:
                no_improve += 1

            self._update_history(
                best_score=iter_best_score,
                trigger_str=curr_p,
                trigger=tokens,
            )

            if (
                self.random_early_stop_patience is not None
                and no_improve >= self.random_early_stop_patience
            ):
                self.pbar.update(self.total_tokens - len(tokens))
                break
            if self.best_sim is not None and iter_best_score > self.best_sim:
                break

        # print(f"final similarity: {iter_best_score}")

    def square_attack(
        self,
        inputs,
    ):
        """A 1D adaptation of the image Square Attack for token sequence (prompt) optimization.

        Instead of square patches over image pixels, we sample contiguous blocks ("windows")
        over the sequence of appended tokens and refresh the entire block with randomly
        sampled vocabulary tokens. A proposal is accepted if it increases cosine similarity.

        """
        initial_tokens = self.history[-1]["trigger"] if len(self.history) else []

        valid_vocab_ids = self._get_valid_vocab_ids()
        if self.total_tokens <= 0:
            raise ValueError("total_tokens must be > 0")

        # Initialize appended tokens randomly.
        appended_tokens = []
        if initial_tokens:
            appended_tokens = list(initial_tokens)

        for i in range(len(appended_tokens), self.total_tokens):
            if initial_tokens and len(initial_tokens) != 0:
                appended_tokens.append(initial_tokens[i % len(initial_tokens)])
            else:
                tok_id = np.random.choice(valid_vocab_ids)
                tok = self.model.tokenizer.decode(int(tok_id))
                if not tok.strip():  # ensure non-empty
                    tok = self.model.tokenizer.encode("the", add_special_tokens=False)[
                        0
                    ]  # fallback harmless common token
                appended_tokens.append(tok)

        def build_prompt(tokens_list):
            return self.model.tokenizer.decode(tokens_list, skip_special_tokens=True)

        current_prompt = build_prompt(appended_tokens)
        with torch.no_grad():
            best_sim = float(
                -self.model.compute_loss_from_texts(
                    candidate_trigger_strs=[current_prompt],
                    inputs=inputs,
                    loss_func=self.loss_func,
                )[0]
            )

        self._update_history(
            best_score=best_sim,
            trigger_str=current_prompt,
            trigger=appended_tokens,
        )
        best_tokens = list(appended_tokens)
        no_improve = 0

        for it in range(1, self.square_num_iters + 1):
            # Schedule analogous to square attack p-schedule.
            p = self._p_selection(self.square_p_init, it - 1, self.square_num_iters)
            block_size = max(1, int(round(p * self.total_tokens)))
            block_size = min(block_size, self.total_tokens)

            start = np.random.randint(0, self.total_tokens - block_size + 1)
            end = start + block_size

            # Generate multiple proposals by refreshing the same contiguous block with random tokens.
            proposals_tokens = []
            for _ in range(max(1, self.batch_size)):
                proposal = list(best_tokens)
                for pos in range(start, end):
                    # Sample a small pool then choose one at random for this position
                    pool_ids = np.random.choice(
                        valid_vocab_ids, size=(self.square_random_pool_per_pos,)
                    )
                    chosen_id = np.random.choice(pool_ids)
                    new_tok = int(chosen_id)
                    if not self.model.tokenizer.decode(
                        new_tok, skip_special_tokens=True
                    ).strip():
                        # skip empty; retain old token
                        continue
                    proposal[pos] = new_tok
                proposals_tokens.append(proposal)

            # Build all candidate prompts and evaluate in batches
            batch_prompts = [build_prompt(toks) for toks in proposals_tokens]
            with torch.no_grad():
                losses = self.model.compute_loss_from_texts(
                    candidate_trigger_strs=batch_prompts,
                    inputs=inputs,
                    loss_func=self.loss_func,
                )
                min_loss, min_idx = torch.min(losses, dim=0)
                prop_best_sim = float(-min_loss.item())
                best_idx = int(min_idx.item())

            if prop_best_sim > best_sim:
                best_sim = prop_best_sim
                best_tokens = proposals_tokens[best_idx]
                current_prompt = batch_prompts[best_idx]
                no_improve = 0
            else:
                no_improve += 1

            self._update_history(
                best_score=best_sim,
                trigger_str=current_prompt,
                trigger=best_tokens,
            )

            # Early stopping
            if (
                self.square_early_stop_patience
                and no_improve >= self.square_early_stop_patience
            ):
                break
            if self.best_sim is not None and best_sim > self.best_sim:
                break

    def optimize_trigger(
        self,
        texts: List[str],
        initial_trigger: Optional[str] = "! " * 20,
        targets: TargetsDict = None,
        target_text: Optional[str] = None,
    ) -> OptimizerResult:
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        inputs, _ = self.model.prepare_text_inputs(
            texts=texts,
            initial_trigger=initial_trigger,
            targets=targets,
        )
        # TODO use tracker

        # TODO better way to pass hot_start
        if target_text is not None and self.hot_start:
            self.target_text = target_text
            self._get_hot_start()

        self.pbar = tqdm(total=self.total_tokens + self.square_num_iters, unit=" steps")
        self.history = []

        self.random_attack(inputs=inputs)

        if self.best_sim is None or self.history[-1]["best_score"] <= self.best_sim:
            self.square_attack(inputs=inputs)

        result = OptimizerResult(
            best_loss=self.history[-1]["best_score"],
            best_trigger_str=self.history[-1]["trigger_str"],
            best_trigger=self.history[-1]["trigger"],
            trigger_strs=[x["trigger_str"] for x in self.history],
            losses=[x["best_score"] for x in self.history],
            full_prompt=[
                t.replace(
                    OPTIMIZED_TRIGGER_PLACEHOLDER, self.history[-1]["trigger_str"]
                )
                for t in texts
            ],
        )
        return result

    def _p_selection(self, p_init: float, it: int, n_iters: int) -> float:
        """Piece-wise constant schedule for p (re-used from original Square Attack)."""
        scaled = int(it / n_iters * 10000)
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
            if english_pattern.match(token):
                valid_ids.append(token_id)

        self._valid_vocab_ids = np.array(valid_ids)
        return self._valid_vocab_ids

    def _get_hot_start(self):
        if self.hot_start_str is not None:
            return self.hot_start_str

        if self.openai_client is None:
            self.openai_client = OpenAI()

        messages: list[ChatCompletionUserMessageParam] = [
            {
                "role": "user",
                "content": f"Return a short sentence, up to 15 words, related to the following passage: {self.target_text}",
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
        self.hot_start_str = decoded
        return text, tokens, decoded

    def _update_history(
        self,
        best_score: float,
        trigger_str: str,
        trigger: List[int],
        step: int = None,
    ):
        last_step = self.history[-1]["step"] if len(self.history) else -1
        self.history.append(
            {
                "step": step if step is not None else last_step + 1,
                "best_score": best_score,
                "num_tokens": self.model.get_usage_stats()["total_tokens"],
                "num_api": self.model.get_usage_stats()["forward_calls"],
                "trigger_str": trigger_str,
                "trigger": trigger,
            }
        )

        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix(
                {
                    "similarity": f"{self.history[-1]["best_score"]:.5f}",
                    "num_tokens": f"{self.history[-1]["num_tokens"]}",
                }
            )
