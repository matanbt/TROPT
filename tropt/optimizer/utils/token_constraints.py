import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from jaxtyping import Int
from torch import Tensor

logger = logging.getLogger(__name__)

UNUSED_TOKEN_REGEX = r"(\[unused\d+\]|<unused\d+>)"

# Common shapes of special tokens
SPECIAL_TOKEN_REGEX = (
    r"^(<\|.+\|>"
    r"|\|\|\|.+\|\|\|"                                                    # OLMo `|||EMAIL_ADDRESS|||`
    r"|</?(s|pad|unk|mask|bos|eos|cls|sep|think|tool_call|tool_response)>"
    r"|\[[^\]\s]+\])$"
)

# Byte-fallback tokens, e.g. `<0x41>`.
BYTE_FALLBACK_REGEX = r"^<0x[0-9A-Fa-f]{2}>$"

@dataclass
class TokenConstraints:
    disallow_non_ascii: bool = True
    """
    Disallow non-ASCII tokens, which may be escaped by a defender when the trigger is used.
    """

    disallow_special_tokens: bool = True
    """
    Disallow special tokens (e.g., bos, eos, unk, `<|start_header_id|>`), which may be escaped by a defender when the
    trigger is used. Covers orphan copies in the plain vocab that no text reproduces.
    """

    disallow_unused_tokens: bool = True
    """
    disallow `<unused*>` tokens, which can be filtered by a defender.
    In many cases there are not part of the special tokens, thus require special care.
    """

    disallow_untypeable_tokens: bool = True
    """
    Disallow tokens no text can produce: byte-fallback pieces that text never re-encodes
    to (`<0x41>`), and ids no token maps to (padding rows of an oversized embedding).
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
        vocab_size = vocab_size or _resolve_vocab_size(tokenizer)

        cache_key = (
            tokenizer.name_or_path,
            vocab_size,
            self.disallow_non_ascii,
            self.disallow_special_tokens,
            self.disallow_unused_tokens,
            self.disallow_untypeable_tokens,
            tuple(self.disallow_custom_token_ids),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build blacklist:
        # initialize with any given custom ids
        blacklist_ids = set(self.disallow_custom_token_ids)

        if self.disallow_special_tokens:
            blacklist_ids.update(_get_special_token_ids(tokenizer))

        if self.disallow_untypeable_tokens:
            # These are identified by their vocabulary *piece*, not by decoding it
            vocab: Optional[Dict[str, int]] = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None
            if vocab is None:
                # Backends without a piece -> id mapping (e.g. OpenAI/tiktoken) can't be checked
                logger.warning(
                    "Tokenizer %s exposes no vocabulary; cannot filter untypeable tokens "
                    "(byte-fallback pieces, unmapped ids). The trigger may not be reproducible.",
                    tokenizer.name_or_path,
                )
            else:
                # Some bytes are typeable: `<0x0A>` is how Llama-2 writes a newline
                blacklist_ids.update(
                    i for i in _get_byte_fallback_ids(vocab, vocab_size)
                    if not _is_reproducible(tokenizer, i)
                )
                blacklist_ids.update(_get_tokenless_ids(vocab, vocab_size))

        if self.disallow_non_ascii or self.disallow_unused_tokens or self.disallow_special_tokens:
            # Single pass: decode each token once and apply all text-based checks together.
            def is_ascii(s):
                return s.isascii() and s.isprintable()

            unused_pattern = re.compile(UNUSED_TOKEN_REGEX) if self.disallow_unused_tokens else None
            special_pattern = re.compile(SPECIAL_TOKEN_REGEX) if self.disallow_special_tokens else None

            for i in range(vocab_size):
                if i in blacklist_ids:
                    continue  # skip already blacklisted ids for efficiency
                try:
                    token_str = tokenizer.decode([i])
                except Exception as e:
                    logger.debug(f"While performing listing token-blacklist: failed to decode token {i}: {e}")
                    # If we can't decode the token, we can't use it, so we blacklist it
                    blacklist_ids.add(i)
                    continue

                # Check non-ASCII constraint
                if self.disallow_non_ascii and token_str and not is_ascii(token_str):
                    blacklist_ids.add(i)

                # Check unused token constraint
                elif self.disallow_unused_tokens and unused_pattern and token_str and unused_pattern.match(token_str):
                    blacklist_ids.add(i)

                # Orphan specials in the plain vocab, which `_get_special_token_ids` cannot
                # see; the reproducibility check spares lookalikes such as BLOOM's `[INFO]`.
                elif (special_pattern and token_str and special_pattern.match(token_str)
                        and not _is_reproducible(tokenizer, i)):
                    blacklist_ids.add(i)

        blacklist_ids = sorted(list(blacklist_ids))
        # filter out negative / out-of-vocab ids (in case tokenizer has weird behavior)
        blacklist_ids = [tid for tid in blacklist_ids if 0 <= tid < vocab_size]
        self._cache[cache_key] = blacklist_ids
        logger.info(
            "Black-listing {}% of the vocabulary ({} tokens / {} vocab)".format(
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


## ------- Tokenizer-introspection helpers ------- ##

def _is_reproducible(tokenizer, token_id: int) -> bool:
    """True if typing the token's text yields `token_id` again. Probed word-initial and
    mid-word, as BPE/SentencePiece keep those apart (XLM-R's `[1]` needs a word before it)."""
    try:
        text = tokenizer.decode([token_id])
    except Exception:
        return False
    for probe in (text, f"a{text}"):
        try:
            if token_id in tokenizer.encode(probe, add_special_tokens=False):
                return True
        except Exception as e:
            logger.debug(f"While performing listing token-blacklist: failed to encode token {token_id}: {e}")
    return False


def _get_special_token_ids(tokenizer) -> set[int]:
    """All ids the tokenizer marks as special.

    Notes:
    - `all_special_ids` holds only the `special_tokens_map` ones (Llama-3: bos and eos), so added tokens
    flagged special (Llama-3's other 254, Gemma's `<start_of_turn>`) are collected too.
    - Added tokens are matched on shape too, as some vendors ship special tokens with
    `special=False` (Qwen `<|fim_prefix|>`, GPT-J `<|extratoken_N|>`, Gemma `<s>`).
    - Shape alone decides here, with no reproducibility check: HF's `encode` matches added
    tokens in raw text, so every added token "reproduces" and the check can't tell them apart.
    """
    ids = {int(i) for i in (getattr(tokenizer, "all_special_ids", None) or [])}
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    special_tok_pattern = re.compile(SPECIAL_TOKEN_REGEX)
    for tid, tok in added.items():
        content = getattr(tok, "content", "") or ""
        if getattr(tok, "special", False) or special_tok_pattern.match(content):
            ids.add(int(tid))
    return ids


def _get_tokenless_ids(vocab: Dict[str, int], vocab_size: int) -> set[int]:
    """Ids in `range(vocab_size)` that no token maps to.

    Models often pad the embedding matrix past the tokenizer (e.g, Qwen3: 151936 rows vs
    151669 tokens); those rows decode to nothing, so no text can ever produce them.
    """
    return set(range(vocab_size)) - {int(i) for i in vocab.values()}


def _get_byte_fallback_ids(vocab: Dict[str, int], vocab_size: int) -> set[int]:
    """SentencePiece's byte-fallback ids (`<0x41>`)."""
    pattern = re.compile(BYTE_FALLBACK_REGEX)
    return {
        int(i) for piece, i in vocab.items()
        if int(i) < vocab_size and isinstance(piece, str) and pattern.match(piece)
    }


def _resolve_vocab_size(tokenizer) -> int:
    """Calculate the effective vocab size

    E.g., `tokenizer.vocab_size` excludes added tokens (128000 vs 128256 on Llama-3),
    which would both hide those ids from the scan and truncate them out of the final blacklist.
    """
    added = getattr(tokenizer, "added_tokens_decoder", None) or {}
    return max([tokenizer.vocab_size, *(int(tid) + 1 for tid in added)])
