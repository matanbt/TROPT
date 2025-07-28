import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class TokenConstraints:
    disallow_non_ascii: bool = True
    disallow_special_tokens: bool = (
        True  # it is reccomended to always disallow special tokens
    )
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
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build blacklist:
        blacklist_ids = set()
        vocab_size = vocab_size or tokenizer.vocab_size

        if self.disallow_special_tokens:
            # Including tokens from tokenizer.special_tokens_map (e.g., bos, eos, unk)
            blacklist_ids.update(tokenizer.all_special_ids)

        if self.disallow_non_ascii:

            def is_ascii(s):
                return s.isascii() and s.isprintable()

            # Iterate through the vocabulary to find non-ASCII tokens.
            for i in range(vocab_size):
                decoded_token = tokenizer.decode([i])
                if decoded_token and not is_ascii(decoded_token):
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
