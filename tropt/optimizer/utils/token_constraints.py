import logging
import re
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

UNUSED_TOKEN_REGEX = r"<unused\d+>"

@dataclass
class TokenConstraints:
    disallow_non_ascii: bool = True
    disallow_special_tokens: bool = (
        True  # it is reccomended to always disallow special tokens as these may be escaped (by defender) when trigger is used
    )
    disallow_unused_tokens: bool = False  # disallow `<unused*>` tokens, which can be filtered (by defender) when trigger is used [TODO test this on multiple models and then set to True by default]
    disallow_custom_token_ids: List[int] = field(default_factory=list)
    _cache: dict = field(
        default_factory=dict, init=False, repr=False, hash=False, compare=False
    )

    def get_blacklist_ids(self, tokenizer, vocab_size: int = None) -> List[int]:
        """
        Returns a list of token IDs that should be blacklisted based on the constraints.
        """

        # HACK for caching:
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
                # Get the token string directly from the tokenizer's vocabulary [TODO better??]
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
