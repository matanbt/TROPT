import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Annotated, Any, List, Set

import torch
from accelerate.utils.memory import find_executable_batch_size
from jaxtyping import Float
from transformers import AutoModelForCausalLM, AutoTokenizer

from tropt.loss.base import TextBasedLoss

logger = logging.getLogger(__name__)


@dataclass
class BinaryLMJudgeLoss(TextBasedLoss):
    """
    Generic loss based on LLM binary judgment (Yes/No questions). Goal is to maximize the "yes" score.

    Computes a soft score by prompting an LLM with a binary question,
    then uses the logit ratio between affirmative and negative responses.

    Subclasses should override `_create_prompt` to define the specific question.

    Args:
        model_name_or_path: HuggingFace model name/path
        positive_words: Set of words indicating positive response (default: {"Yes", "yes"})
        negative_words: Set of words indicating negative response (default: {"No", "no"})
    """

    model_name_or_path: str = "HuggingFaceTB/SmolLM2-135M"
    positive_words: Set[str] = field(default_factory=lambda: {"Yes", "yes", " Yes", " yes"})
    negative_words: Set[str] = field(default_factory=lambda: {"No", "no", " No", " no"})
    judge_lm_batch_size: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # the loaded model and tokenizer
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _positive_token_ids: Set[int] = field(default=None, init=False, repr=False)
    _negative_token_ids: Set[int] = field(default=None, init=False, repr=False)
    # TODO are the `field` required?

    def __post_init__(self):
        logger.info(f"Loading LM judge model for loss: {self.model_name_or_path}")
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16
        ).eval().to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Initialize token ID sets
        self._positive_token_ids = self._get_token_ids(self.positive_words)
        self._negative_token_ids = self._get_token_ids(self.negative_words)

    def _get_token_ids(self, words: Set[str]) -> Set[int]:
        """Convert a set of words to their token IDs."""
        token_ids = set()
        for word in words:
            tokens = self._tokenizer.encode(word, add_special_tokens=False)
            if tokens:
                token_ids.add(tokens[-1])  # Use last token if multiple
        return token_ids

    @abstractmethod
    def _create_prompt(self, text: str) -> str:
        """
        Create the evaluation prompt. Override in subclasses.
        E.g., f'Is this text readable? {text} Just answer Yes or No.'
        """
        raise NotImplementedError("Subclasses must implement _create_prompt")

    def _compute_batch_scores(
        self,
        texts: List[str]
    ) -> Float[torch.Tensor, "bsz"]:
        """
        Compute scores for a batch of texts.

        Returns:
            Tensor of shape (bsz,) with scores for each text
        """
        # Create prompts for all texts
        prompts = [self._create_prompt(text) for text in texts]
        messages_batch = [[{"role": "user", "content": prompt}] for prompt in prompts]
        formatted_prompts = [
            self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in messages_batch
        ]

        # Tokenize and pad the batch
        inputs = self._tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            logits = self._model(**inputs).logits  # (bsz, seq_len, vocab_size)

        # Get logits for the last token in each sequence
        # Account for padding by using attention_mask
        last_token_indices = inputs.attention_mask.sum(dim=1) - 1  # (bsz,)
        batch_indices = torch.arange(logits.shape[0], device=self.device)
        last_logits = logits[batch_indices, last_token_indices, :]  # (bsz, vocab_size)

        # Sum logits over positive and negative token sets
        positive_ids = list(self._positive_token_ids)
        negative_ids = list(self._negative_token_ids)

        z_aff = last_logits[:, positive_ids].sum(dim=1)  # (bsz,)
        z_neg = last_logits[:, negative_ids].sum(dim=1)  # (bsz,)

        # Compute scores (avoid division by zero)
        denom = z_aff + z_neg
        scores = torch.where(
            denom.abs() < 1e-10,
            torch.zeros_like(denom),
            (z_aff - z_neg) / denom
        )

        return scores

    def __call__(
        self,
        texts: Annotated[List[str], "bsz"],
    ) -> Float[torch.Tensor, "bsz"]:
        """
        Compute loss for a batch of texts with dynamic batching.
        """
        n_texts = len(texts)

        @find_executable_batch_size(starting_batch_size=self.judge_batch_size)
        def _compute_scores_batched(batch_size: int) -> Float[torch.Tensor, "n_texts"]:

            # --- Update judge batch size ---
            # Automatically lower the default for future calls if this run required a downgrade
            if batch_size < self.judge_batch_size:
                self.judge_batch_size = batch_size
            # --------------------

            all_scores = []

            for idx in range(0, n_texts, batch_size):
                end_idx = min(idx + batch_size, n_texts)
                batch_texts = texts[idx:end_idx]

                batch_scores = self._compute_batch_scores(batch_texts)
                all_scores.append(batch_scores)

            return torch.cat(all_scores, dim=0)

        return _compute_scores_batched()  # TODO make sure callers `to.device & dtype`


@dataclass
class InputReadabilityLoss(BinaryLMJudgeLoss):
    """
    Loss that encourages readable/fluent trigger text.
    Minimizing this loss maximizes readability.

    https://arxiv.org/abs/2410.02163
    """

    def _create_prompt(self, prompt_text: str) -> str:
        return f'Is this text readable? `{prompt_text}` Just answer Yes or No.'
