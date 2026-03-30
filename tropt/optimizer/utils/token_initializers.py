from tropt.model.model_base import BaseTokenizer
from ty_extensions import Unknown
import random
import string
from typing import List, Optional

import transformers
from jaxtyping import Float
from torch import Tensor


def get_printable_random_trigger(
    trigger_len: int,
    return_ids: bool = False,
    blacklist_ids: Optional[List[int]] = None,
    tokenizer: Optional[BaseTokenizer] = None,
) -> str | Float[Tensor, "trigger_seq_len"]:
    """
    Generates a random initial trigger consisting of printable ASCII english letters.
    If the tokenizer is provided, the trigger is tokenized and truncated to ensure it fits within the specified length.
    Tokens whose IDs appear in blacklist_ids are resampled until a clean sequence is found.
    Otherwise, the trigger is generated as a string of the specified length.
    """
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
        return tokenizer.encode(initial_trigger, add_special_tokens=False)  # shape: (trigger_seq_len,)

    return initial_trigger