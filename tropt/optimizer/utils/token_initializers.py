import random
import string
from typing import List, Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor

from tropt.model.model_base import BaseTokenizer


def random_single_flips(
    trigger_ids: Int[Tensor, "trigger_seq_len"],
    n_variations: int,
    valid_token_ids: Optional[Int[Tensor, "n_valid"]] = None,
    vocab_size: Optional[int] = None,
) -> Int[Tensor, "n_variations trigger_seq_len"]:
    """`n_variations` copies of `trigger_ids`, the first left intact and each of the
    rest given one random single-token flip.

    Used by gradient-averaging optimizers (GASLITE, GASLITE+) and by logit-sampling
    ones (RASLITE+). Sampling is restricted to `valid_token_ids` when given,
    otherwise uniform over `vocab_size`.
    """
    assert (valid_token_ids is None) ^ (vocab_size is None), (
        "Pass exactly one of `valid_token_ids` or `vocab_size`."
    )
    device = trigger_ids.device
    variations = trigger_ids.repeat(n_variations, 1)

    n_flips = n_variations - 1
    if n_flips <= 0:
        return variations

    pos = torch.randint(0, variations.shape[1], (n_flips, 1), device=device)
    if valid_token_ids is not None:
        tok = valid_token_ids[
            torch.randint(0, len(valid_token_ids), (n_flips, 1), device=device)
        ]
    else:
        assert vocab_size is not None  # guaranteed by the XOR assert above
        tok = torch.randint(0, vocab_size, (n_flips, 1), device=device)
    # `variations[1:]` is a view, so this writes through to `variations`.
    variations[1:].scatter_(1, pos, tok.to(variations.dtype))
    return variations


def get_printable_random_trigger(
    trigger_len: int,
    return_ids: bool = False,
    blacklist_ids: Optional[List[int]] = None,
    tokenizer: Optional[BaseTokenizer] = None,
    token_constraints: Optional["TokenConstraints"] = None,  # ty:ignore[unresolved-reference] (to avoid circular imports)
) -> str | Float[Tensor, "trigger_seq_len"]:
    """
    Generates a random initial trigger consisting of printable ASCII english letters.
    - If the tokenizer is provided, the trigger is tokenized and truncated to ensure it fits within the specified length. Otherwise, the length stands for the number of characters.
    - Tokens whose IDs appear in `blacklist_ids` are resampled until a clean sequence is found.
    - `return_ids`: If True, returns a 1-D LongTensor of token IDs (requires `tokenizer`), instead of the string.
    - `token_constraints`: If provided (and tokenizer is set), extracts blacklist_ids automatically. Overrides `blacklist_ids`.
    """
    assert not (token_constraints is not None and blacklist_ids is not None), (
        "Pass either `token_constraints` or `blacklist_ids`, not both."
    )
    if token_constraints is not None and tokenizer is not None:
        blacklist_ids = token_constraints.get_blacklist_ids(tokenizer)

    _chars = string.ascii_letters + string.digits + ' '  # + string.punctuation
    _chars += ' ' * 10  # adding more spaces to increase their appearance

    if tokenizer is not None:
        blacklist_ids: set[int] = set(blacklist_ids) if blacklist_ids else set()
        _token_ids: list[int] = []
        while len(_token_ids) < trigger_len:
            candidate = ''.join(random.choices(_chars, k=trigger_len * 4))
            candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
            clean_ids = [t for t in candidate_ids if t not in blacklist_ids]
            needed = trigger_len - len(_token_ids)
            _token_ids.extend(clean_ids[:needed])
        initial_trigger: str = tokenizer.decode(_token_ids)
    else:
        initial_trigger: str = ''.join(random.choices(_chars, k=trigger_len * 4))
        initial_trigger = initial_trigger[:trigger_len]

    if return_ids:
        assert tokenizer is not None, "Tokenizer must be provided to return token IDs."
        return torch.tensor(tokenizer.encode(initial_trigger, add_special_tokens=False))  # shape: (trigger_seq_len,)

    return initial_trigger
