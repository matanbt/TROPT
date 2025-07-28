import random
import string

import transformers
import torch
from typing import List
from jaxtyping import Float
from torch import Tensor

def get_printable_random_trigger(
    trigger_len: int,
    return_ids: bool = False,
    tokenizer: transformers.PreTrainedTokenizer = None,
) -> str | Float[Tensor, "trigger_seq_len"]:
    """
    Generates a random initial trigger consisting of printable ASCII characters.
    If the tokenizer is provided, the trigger is tokenized and truncated to ensure it fits within the specified length.
    Otherwise, the trigger is generated as a string of the specified length.
    """
    _chars = string.ascii_letters + string.digits + string.punctuation
    initial_trigger = ''.join(random.choices(_chars, k=trigger_len * 4))

    if tokenizer is not None:
        # Encode, slice to strict token length, and decode back
        _token_ids = tokenizer.encode(initial_trigger, add_special_tokens=False)
        initial_trigger = tokenizer.decode(_token_ids[:trigger_len])
    else:
        initial_trigger = initial_trigger[:trigger_len]

    if return_ids:
        assert tokenizer is not None, "Tokenizer must be provided to return token IDs."
        return tokenizer(
            initial_trigger, add_special_tokens=False, return_tensors="pt"
        ).input_ids.squeeze(0)  # shape: (trigger_seq_len,)

    return initial_trigger