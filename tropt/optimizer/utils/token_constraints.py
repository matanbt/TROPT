import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

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

        if self.disallow_non_ascii:

            def is_ascii(s):
                return s.isascii() and s.isprintable()

            # Iterate through the vocabulary to find non-ASCII tokens.
            for i in range(vocab_size):
                if i in blacklist_ids:
                    continue  # skip already blacklisted ids for efficiency
                try:
                    decoded_token = tokenizer.decode([i])
                    if decoded_token and not is_ascii(decoded_token):
                        blacklist_ids.add(i)
                except Exception as e:
                    logger.warning(f"While perfoming listing token-blacklist: failed to decode token {i}: {e}")
                    # If we can't decode the token, we can't use it, so we blacklist it
                    blacklist_ids.add(i)

        if self.disallow_unused_tokens:
            # Iterate through the vocabulary to find tokens matching <unused\d+> pattern
            unused_pattern = re.compile(UNUSED_TOKEN_REGEX)
            for i in range(vocab_size):
                if i in blacklist_ids:
                    continue  # skip already blacklisted ids fr efficiency
                # Get the raw token string from the vocabulary (not decode(), which may post-process)
                token_str = tokenizer.convert_ids_to_tokens([i])[0]
                if token_str and unused_pattern.match(token_str):
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

    def get_valid_token_ids(self, tokenizer, vocab_size: int, device) -> Tensor:
        """
        Returns a tensor of valid token ids (reverse of blacklist) based on the constraints.
        Reuses the potentially cached blacklist for efficiency.
        """
        blacklist_ids = self.get_blacklist_ids(tokenizer, vocab_size)
        token_mask = torch.ones(vocab_size, device=device, dtype=torch.bool)
        token_mask[blacklist_ids] = False
        return token_mask.nonzero(as_tuple=False).squeeze(-1)
