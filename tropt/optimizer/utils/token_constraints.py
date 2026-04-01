import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional
from jaxtyping import Int

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

UNUSED_TOKEN_REGEX = r"<unused\d+>"

@dataclass
class TokenConstraints:
    disallow_non_ascii: bool = True
    """
    Disallow non-ASCII tokens, which may be escaped by a defender when the trigger is used.
    """

    disallow_special_tokens: bool = True
    """
    Disallow special tokens (e.g., bos, eos, unk), which may be escaped by a defender when the trigger is used.
    """

    disallow_unused_tokens: bool = True
    """
    disallow `<unused*>` tokens, which can be filtered by a defender.
    In many cases there are not part of the special tokens, thus require special care.
    """

    disallow_custom_token_ids: List[int] = field(default_factory=list)
    """
    Disallow any additional custom token ids.
    """

    _cache: dict = field(
        default_factory=dict, init=False, repr=False, hash=False, compare=False
    )

    def get_blacklist_ids(self, tokenizer, vocab_size: Optional[int] = None) -> List[int]:
        """
        Returns a list of token IDs that should be blacklisted based on the constraints.
        """

        cache_key = (
            tokenizer.name_or_path,
            self.disallow_non_ascii,
            self.disallow_special_tokens,
            self.disallow_unused_tokens,
            tuple(self.disallow_custom_token_ids),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build blacklist:
        # initialize with any given custom ids
        blacklist_ids = set(self.disallow_custom_token_ids)
        vocab_size = vocab_size or tokenizer.vocab_size

        if self.disallow_special_tokens:
            # Including tokens from tokenizer.special_tokens_map (e.g., bos, eos, unk)
            blacklist_ids.update(tokenizer.all_special_ids)

        if self.disallow_non_ascii or self.disallow_unused_tokens:
            # Single pass: decode each token once and apply all text-based checks together.
            def is_ascii(s):
                return s.isascii() and s.isprintable()

            unused_pattern = re.compile(UNUSED_TOKEN_REGEX) if self.disallow_unused_tokens else None

            for i in range(vocab_size):
                if i in blacklist_ids:
                    continue  # skip already blacklisted ids for efficiency
                try:
                    token_str = tokenizer.decode([i])
                except Exception as e:
                    logger.debug(f"While perfoming listing token-blacklist: failed to decode token {i}: {e}")
                    # If we can't decode the token, we can't use it, so we blacklist it
                    blacklist_ids.add(i)
                    continue

                # Check non-ASCII constraint
                if self.disallow_non_ascii and token_str and not is_ascii(token_str):
                    blacklist_ids.add(i)
                    
                # Check unused token constraint
                elif self.disallow_unused_tokens and unused_pattern and token_str and unused_pattern.match(token_str):
                    blacklist_ids.add(i)

        blacklist_ids = sorted(list(blacklist_ids))
        # filter out negative / out-of-vocab ids (in case tokenizer has weird behavior)
        blacklist_ids = [tid for tid in blacklist_ids if 0 <= tid < vocab_size]
        self._cache[cache_key] = blacklist_ids
        logger.info(
            "Black-lising {}% of the vocabulary ({} tokens / {} vocab)".format(
                round(100 * len(blacklist_ids) / vocab_size, 2),
                len(blacklist_ids),
                vocab_size,
            )
        )
        return blacklist_ids

    def get_whitelist_ids(
        self,
        tokenizer,
        vocab_size: int,
        device=None,
        return_tensor: bool = False,
    ) -> List[int] | Int[Tensor, "n_valid"]:
        """Returns valid (non-blacklisted) token ids. Reuses the cached blacklist for efficiency.

        Args:
            return_tensor: If True, return a 1-D int tensor on `device` instead of a list.
            device: Required when ``return_tensor=True``.
        """
        blacklist_ids = self.get_blacklist_ids(tokenizer, vocab_size)
        blacklist_set = set(blacklist_ids)
        whitelist_ids = [i for i in range(vocab_size) if i not in blacklist_set]
        if return_tensor:
            return torch.tensor(whitelist_ids, dtype=torch.long, device=device)
        return whitelist_ids
