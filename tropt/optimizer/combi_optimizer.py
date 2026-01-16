import logging
import math
import re
import time
from typing import Any, List, Optional

import numpy as np
import torch
import transformers
from openai import OpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from torch import Tensor
from tqdm import tqdm

from sentence_transformers import util
from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER
from tropt.loss.base import BaseLoss
from tropt.models.base import (
    BaseModel,
    GradientTokenAccessMixin,
    LossTokenAccessMixin,
    TargetsDict,
)
from tropt.optimizer.base import BaseOptimizer, OptimizerResult
from tropt.optimizer.utils.buffer import TriggerBuffer
from tropt.optimizer.utils.retokenization import retokenize_filtering
from tropt.optimizer.utils.scheduler import (
    ConstantScheduler,
    LinearScheduler,
    NFlipScheduler,
)
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.optimizer.utils.token_initializers import get_printable_random_trigger
from tropt.tracker.base import BaseTracker

logger = logging.getLogger(__name__)


class CombiOptimizer(BaseOptimizer):
    """
    Implements the Combination Attack optimizer

    """

    model_requirements = (LossTokenAccessMixin, GradientTokenAccessMixin)  # TODO what do I need?

    def __init__(
        self,
        model: BaseModel,
        loss: BaseLoss,
        tracker: Optional[BaseTracker] = None,
        seed: Optional[int] = None,
        # attack parameters:
        hot_start: bool = True,
        batch_size=128,
        sim: str = "cos",
        best_sim: Optional[float] = None,
        total_tokens: int = 100,

        random_num_pool: int = 500,
        random_early_stop_patience: int = 5,

        square_p_init: float = 0.5,
        square_num_iters: int = 2000,
        square_random_pool_per_pos: int = 300,
        square_early_stop_patience: int = 100,
        **kwargs
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

        # save params:
        self.hot_start = hot_start
        self.hot_start_str = None
        self.batch_size = batch_size
        self.sim = util.cos_sim if sim == "cos" else util.dot_score
        self.best_sim = best_sim
        self.total_tokens = total_tokens

        self.random_num_pool = random_num_pool
        self.random_early_stop_patience = random_early_stop_patience

        self.square_p_init = square_p_init
        self.square_num_iters = square_num_iters
        self.square_random_pool_per_pos = square_random_pool_per_pos
        self.square_early_stop_patience = square_early_stop_patience

        self.tokenizer = None
        self.openai_client = OpenAI()


    def _get_trigger_variations(
        self,
        trigger_ids: Float[Tensor, "trigger_seq_len"],
        vocab_size: int,
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
            tok_to_flip_to = torch.randint(0, vocab_size, (1,), device=device).item()
            # apply the flip
            trigger_vars_ids[idx, pos_to_flip] = tok_to_flip_to

        return trigger_vars_ids

    def random_attack(
        self,
        p_adv: str,
        pbar: tqdm = None,
    ):
        curr_p = p_adv
        tokens = []
        token_count = 0
        api_calls = 0

        if self.hot_start_str is not None:
            curr_p += " " + self.hot_start_str
            tokens = self.tokenizer.encode(self.hot_start_str, add_special_tokens=False)
            # TODO change this to use tokens instead of words
            # tokens += hot_start.split(" ")

        device = self.model.device

        p_adv_emb = self.model(curr_p).to(device)
        token_count += self._get_token_count(curr_p)
        api_calls += 1
        base_sim = self.sim(self.q_emb, p_adv_emb).item()
        no_improve = 0
        # print(f"initial similarity: {base_sim}")

        iter_best_score = 0
        valid_vocab_ids = self._get_valid_vocab_ids()

        history = [
            {
                "step": 0,
                "best_score": base_sim,
                "num_tokens": token_count,
                "num_api": api_calls,
            }
        ]
        if pbar is not None:
            pbar.update(len(tokens))
            pbar.set_postfix(
                {
                    "similarity": f"{iter_best_score:.5f}",
                    "num_tokens": f"{token_count}",
                }
            )

        for n in range(total_tokens - len(tokens)):
            # print(f"iteration {n + 1}")
            pool = np.random.choice(valid_vocab_ids, size=(num_pool,))

            # compute current baseline similarity for this iteration
            p_adv_emb = self.model(curr_p).to(device)
            token_count += self._get_token_count(curr_p)
            api_calls += 1
            iter_best_score = self.sim(self.q_emb, p_adv_emb).item()
            best_token = None

            # evaluate candidates in parallel by batching
            for i in range(0, len(pool), self.batch_size):
                batch_ids = [
                    pool[i] for i in range(i, min(i + self.batch_size, len(pool)))
                ]
                batch_tokens = self.tokenizer.batch_decode(batch_ids)

                # build candidate prompts
                check_ps = [curr_p + " " + t for t in batch_tokens]

                # encode all candidates as a single batch
                embs = self.model(check_ps).to(device)
                token_count += self._get_token_count(check_ps)
                api_calls += 1

                # vectorized cosine similarity against q_emb
                scores = self.sim(self.q_emb, embs).squeeze(0)  # shape: [batch]
                max_score, max_idx = torch.max(scores, dim=0)
                ms = max_score.item()
                if ms > iter_best_score:
                    iter_best_score = ms
                    best_token = batch_tokens[max_idx.item()]

            if best_token is not None:
                tokens.append(best_token)
                curr_p += " " + best_token
                no_improve = 0
            else:
                no_improve += 1

            history.append(
                {
                    "step": n + 1,
                    "best_score": iter_best_score,
                    "num_tokens": token_count,
                    "num_api": api_calls,
                }
            )
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "similarity": f"{iter_best_score:.5f}",
                        "num_tokens": f"{token_count}",
                    }
                )

            if early_stop_patience and no_improve >= early_stop_patience:
                pbar.update(total_tokens - len(tokens) - n)
                break
            if self.best_sim is not None and iter_best_score > self.best_sim:
                break

        # print(f"final similarity: {iter_best_score}")
        return tokens, iter_best_score, history

    def square_attack(
        self,
        base_prompt: str,
        total_tokens: int = 20,
        p_init: float = 0.5,
        num_iters: int = 500,
        random_pool_per_pos: int = 50,
        early_stop_patience: int = 100,
        seed: Optional[int] = None,
        initial_tokens: Optional[List[str]] = None,
        pbar: tqdm = None,
    ):
        """A 1D adaptation of the image Square Attack for token sequence (prompt) optimization.

        Instead of square patches over image pixels, we sample contiguous blocks ("windows")
        over the sequence of appended tokens and refresh the entire block with randomly
        sampled vocabulary tokens. A proposal is accepted if it increases cosine similarity.

        Args:
            base_prompt: The initial (fixed) part of the prompt we optimize around.
            total_tokens: Number of optimizable tokens appended to base_prompt.
            p_init: Initial fraction of tokens replaced in one update (analogous to image pixel fraction).
            num_iters: Maximum number of iterations.
            random_pool_per_pos: For every position in the chosen block we sample uniformly that many candidate tokens and pick 1 (simple random draw). Higher => more diversity.
            early_stop_patience: Stop if no improvement for this many iterations.
            seed: Optional RNG seed for reproducibility.
            initial_tokens: Optional list of tokens to initialize the appended tokens (if shorter than total_tokens, repeated as needed).

        Returns:
            appended_tokens: Final list of appended tokens (length = total_tokens).
            final_prompt: The resulting full prompt string.
            history: List of (iteration, best_similarity) tracking progress (iteration 0 is initialization).
        """
        token_count = 0
        api_calls = 0

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        valid_vocab_ids = self._get_valid_vocab_ids()
        if total_tokens <= 0:
            raise ValueError("total_tokens must be > 0")

        # Initialize appended tokens randomly.
        appended_tokens = []
        if initial_tokens:
            appended_tokens = list(initial_tokens)

        for i in range(len(appended_tokens), total_tokens):
            if initial_tokens and len(initial_tokens) != 0:
                appended_tokens.append(initial_tokens[i % len(initial_tokens)])
            else:
                tok_id = np.random.choice(valid_vocab_ids)
                tok = self.tokenizer.decode(int(tok_id))
                if not tok.strip():  # ensure non-empty
                    tok = "the"  # fallback harmless common token
                appended_tokens.append(tok)

        def build_prompt(tokens_list):
            if tokens_list:
                return base_prompt + " " + " ".join(tokens_list)
            return base_prompt

        current_prompt = build_prompt(appended_tokens)
        with torch.no_grad():
            emb = self.model(current_prompt).to(device)
            token_count += self._get_token_count(current_prompt)
            api_calls += 1
            best_sim = self.sim(self.q_emb, emb).item()

        history = [
            {
                "step": 0,
                "best_score": best_sim,
                "num_tokens": token_count,
                "num_api": api_calls,
            }
        ]
        best_tokens = list(appended_tokens)
        no_improve = 0

        for it in range(1, num_iters + 1):
            # Schedule analogous to square attack p-schedule.
            p = self._p_selection(p_init, it - 1, num_iters)
            block_size = max(1, int(round(p * total_tokens)))
            block_size = min(block_size, total_tokens)

            start = np.random.randint(0, total_tokens - block_size + 1)
            end = start + block_size

            # Generate multiple proposals by refreshing the same contiguous block with random tokens.
            proposals_tokens = []
            for _ in range(max(1, self.batch_size)):
                proposal = list(best_tokens)
                for pos in range(start, end):
                    # Sample a small pool then choose one at random for this position
                    pool_ids = np.random.choice(
                        valid_vocab_ids, size=(random_pool_per_pos,)
                    )
                    chosen_id = np.random.choice(pool_ids)
                    new_tok = self.tokenizer.decode(int(chosen_id))
                    if not new_tok.strip():
                        # skip empty; retain old token
                        continue
                    proposal[pos] = new_tok
                proposals_tokens.append(proposal)

            # Build all candidate prompts and evaluate in batches
            batch_prompts = [build_prompt(toks) for toks in proposals_tokens]
            with torch.no_grad():
                embs = self.model(batch_prompts).to(device)
                token_count += sum(self._get_token_count(bp) for bp in batch_prompts)
                api_calls += 1
                scores = self.sim(self.q_emb, embs).squeeze(0)  # [batch]
                max_score, max_idx = torch.max(scores, dim=0)
                prop_best_sim = max_score.item()
                best_idx = int(max_idx.item())

            if prop_best_sim > best_sim:
                best_sim = prop_best_sim
                best_tokens = proposals_tokens[best_idx]
                current_prompt = batch_prompts[best_idx]
                no_improve = 0
            else:
                no_improve += 1

            history.append(
                {
                    "step": it + 1,
                    "best_score": best_sim,
                    "num_tokens": token_count,
                    "num_api": api_calls,
                }
            )
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "similarity": f"{best_sim:.5f}",
                        "num_tokens": f"{token_count}",
                    }
                )

            # Early stopping
            if early_stop_patience and no_improve >= early_stop_patience:
                break
            if self.best_sim is not None and best_sim > self.best_sim:
                break

        return best_tokens, current_prompt, history

    def optimize_trigger(
        self,
        texts: List[str],
        initial_trigger: Optional[str] = "! " * 20,
        targets: TargetsDict = None,
    ) -> OptimizerResult:
        inputs, _ = self.model.prepare_text_inputs(
            texts=texts,
            initial_trigger=initial_trigger,
            targets=targets,
        )
        vocab_size, tokenizer = inputs.vocab_size, inputs.tokenizer
        # TODO is this the correct tokenizer?
        self.tokenizer = tokenizer

        if self.hot_start:
            self._get_hot_start()

        # TODO accept multiple texts
        base_prompt = texts[0]

        pbar = tqdm(total=self.total_tokens + self.square_num_iters, unit=" steps")

        tokens, random_sim, random_history = self.random_attack(
            p_adv=base_prompt,
            pbar=pbar
        )

        # self.square_start = random_history[-1]["num_tokens"]

        history = []
        history.extend(random_history)
        if self.best_sim is not None and random_sim > self.best_sim:
            # return tokens, random_sim, history
            result = OptimizerResult(
                best_loss=random_sim,
                best_trigger_str=best_trigger_str,
                best_trigger=best_trigger_ids,
                trigger_strs=trigger_strings,
            )
            return result

        s_tokens, _, square_history = self.square_attack(
            base_prompt,
            total_tokens,
            p_init,
            num_iters,
            random_pool_per_pos,
            early_stop_patience,
            seed,
            tokens,
            pbar=pbar,
        )
        history.extend(
            [
                {
                    "step": item["step"] + len(random_history),
                    "best_score": item["best_score"],
                    "num_tokens": item["num_tokens"] + history[-1]["num_tokens"],
                    "num_api": item["num_api"] + history[-1]["num_api"],
                }
                for item in square_history[1:]
            ]
        )

        result = OptimizerResult(
            best_loss=loss_per_step[best_loss_idx],
            best_trigger_str=best_trigger_str,
            best_trigger=best_trigger_ids,
            losses=loss_per_step,
            trigger_strs=trigger_strings,
            full_prompt=full_prompt,
        )
        self.tracker.log({"best_loss": result.best_loss, "best_trigger_str": result.best_trigger_str})
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

        vocab = self.tokenizer.get_vocab()
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

        messages: list[ChatCompletionUserMessageParam] = [
            {
                "role": "user",
                "content": f"Return a short sentence, up to 15 words, related to the following passage: {self.query}",
            },
        ]

        response = self.openai_client.chat.completions.create(
            model="gpt-5-nano",
            messages=messages,
            # max_completion_tokens=100,  # This seems to mess up the completion
        )

        text = response.choices[0].message.content.rstrip(".")
        tokens = self.tokenizer.encode(text)
        decoded: str = self.tokenizer.decode(tokens, skip_special_tokens=True)
        self.hot_start_str = decoded
        return text, tokens, decoded

    def _get_token_count(self, text: str | list[str]) -> int:
        if isinstance(text, str):
            text = [text]
        return sum(len(x) for x in self.tokenizer(text)["input_ids"])
