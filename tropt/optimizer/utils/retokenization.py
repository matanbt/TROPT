"""
Retokenization utilities for token-level optimization.

Provides both filtering (keep only candidates that survive round-trip)
and transformation (produce the retokenized version of a token sequence).
"""

import logging
from typing import Any, List, Tuple

import torch
import transformers
from jaxtyping import Float
from torch import Tensor

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER, TextTemplates

logger = logging.getLogger(__name__)

# TODO need to profile whether these functions are bottlenecks, and if so parallelize them. (e.g., `retokenize_filtering` loop over the vocab can be parallelized)

def retokenize_filtering(
        ids: Float[Tensor, "bsz n_ids"],
        tokenizer: transformers.PreTrainedTokenizer
    ) -> Float[Tensor, "new_search_width n_optim_ids"]:
    """
    Filters out sequeneces of token ids that change after retokenization.
    It is a common practice for discrete token optimizations to ensure alignment between the optimized
    token sequences and the ones that will be eventually provided to the model. It was shown
    to improve performance.

    Args:
        ids : Tensor, shape = (bsz, n_ids)
            batch of token ids to be filtered
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer

    Returns:
        filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
            all token ids that are the same after retokenization
    """
    ids_decoded = tokenizer.batch_decode(ids)
    filtered_ids = []

    for i in range(len(ids_decoded)):
        # Retokenize the decoded token ids
        ids_encoded = tokenizer(
            ids_decoded[i], return_tensors="pt", add_special_tokens=False
        ).to(ids.device)["input_ids"][0]


        if torch.equal(ids[i], ids_encoded):
            # trigger is the same after retokenization
            filtered_ids.append(ids[i])

    if not filtered_ids:
        # This occurs in some cases, e.g. using the Llama-3 tokenizer with a bad initialization
        raise RuntimeError(
            "No token sequences are the same after decoding and re-encoding. "
            "Consider disabling retokenization filtering by setting `use_retokenize=False` or trying a different initial trigger string."
        )
    logger.debug(f"Retokenization filtering: {len(filtered_ids)}/{len(ids)} = {100 * len(filtered_ids) / len(ids):.2f}% candidates kept.")

    return torch.stack(filtered_ids)


def retokenize_transform(
    ids: Float[Tensor, "n_ids"],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Float[Tensor, "n_ids"]:
    """Retokenize a token sequence: decode then re-encode.

    Handles length mismatches by truncating or padding (with original ids)
    to preserve the original sequence length.

    Args:
        ids: Token ids to retokenize, shape (n_ids,).
        tokenizer: The model's tokenizer.

    Returns:
        Retokenized ids with the same length as input.
    """
    text = tokenizer.decode_trigger(ids)
    retok_ids = tokenizer.encode_trigger(text).to(ids.device, dtype=ids.dtype)

    trigger_len = ids.shape[0]
    if retok_ids.shape[0] >= trigger_len:
        return retok_ids[:trigger_len]
    else:
        # Pad with original ids if retokenization shortened the sequence
        result = ids.clone()
        result[: retok_ids.shape[0]] = retok_ids
        return result


def full_messages_retokenize_filtering(
    candidate_trigger_ids: Float[Tensor, "n_candidates trigger_seq_len"],
    tokenizer: transformers.PreTrainedTokenizer,
    templates: TextTemplates,
    trigger_placeholder: str = OPTIMIZED_TRIGGER_PLACEHOLDER,
):
    """
    Filters out candidate triggers that change after retokenization
    in the *full* trigger-combined message context
    (as opposed to just retokenizing the trigger, handled by `retokenize_filtering`).

    Some context:
        The idea is that we want full alignment between: (i) the token ids the model "sees"
        during optimization, and (ii) the token ids the model "sees" at inference time (which
        will be an artifact of retokenization). Crucially, this restriction is much more strict
        than the one in `retokenize_filtering` (i.e. the following function also enforces the
        former condition), which only requires successful retokenization of the trigger.
        Subsequenctly, for some tokenizers, this function may leave very few to no valid
        candidates, in which case the user should consider disabling. Notably, empirically,
        optimizations were shown to perform well with the `retokenize_filtering` alone.

    Args:
        candidate_trigger_ids : Tensor, shape = (n_candidates, trigger_seq_len)
            candidate trigger token ids
        tokenizer : ~transformers.PreTrainedTokenizer
            the model's tokenizer
        templates : List[str] (length = n_templates)
            list of user message templates, each containing the `trigger_placeholder`
            where the trigger will be inserted.
        trigger_placeholder : str
            the placeholder string in `templates` to be replaced by the trigger

    Returns:
        filtered_ids : Tensor, shape = (new_n_candidates, trigger_seq_len)
            all token ids that are the same after retokenization in full context
    """
    candidate_trigger_texts = tokenizer.batch_decode(candidate_trigger_ids)
    filtered_ids = []

    # Split the user templates into before/after the trigger parts
    before_texts, after_texts = [], []
    for template in templates:
        bef, aft = template.split(trigger_placeholder)
        before_texts.append(bef)
        after_texts.append(aft)

    # Tokenize the before/after the trigger
    # We save the tensor ids in lists, as they may have different lengths
    before_ids: List[List[int]] = tokenizer(before_texts, add_special_tokens=False).input_ids
    after_ids: List[List[int]] = tokenizer(after_texts, add_special_tokens=False).input_ids

    for cand_trigger_text, cand_trigger_ids in zip(candidate_trigger_texts, candidate_trigger_ids):
        # We take each trigger, combine it with each user template, and check if retokenization matches
        is_curr_trigger_valid = True

        for template, curr_before_ids, curr_after_ids in zip(templates, before_ids, after_ids):
            # 1. Build the triggeted template text:
            triggered_template: str = template.replace(trigger_placeholder, cand_trigger_text)

            # 2.a. Build the concat of the original ids:
            # (this is what the optimization sees)
            triggered_template_ids: List[int] = curr_before_ids + cand_trigger_ids.tolist() + curr_after_ids
            # 2.b. Get the (re)tokenization of the triggered text template:
            # (this is what the model input will see at inference time)
            triggered_template_new_ids: List[int] = tokenizer(
                triggered_template, add_special_tokens=False
            ).input_ids

            print("old texts:", tokenizer.decode(triggered_template_ids))
            print("new texts:", tokenizer.decode(triggered_template_new_ids))

            # 3. We want the original to match the (re)tokenization:
            if triggered_template_ids != triggered_template_new_ids:
                is_curr_trigger_valid = False
                break

        if is_curr_trigger_valid:
            filtered_ids.append(cand_trigger_ids)

    if not filtered_ids:
        raise RuntimeError(
            "[Full Message Retokenization] No token sequences are the same after retokenization "
            "in full context. Consider disabling retokenization filtering by setting `use_retokenize=False` or trying a different initial trigger string."
        )
    logger.debug(f"Full-template Retokenization filtering: {len(filtered_ids)}/{len(candidate_trigger_ids)} = {100 * len(filtered_ids) / len(candidate_trigger_ids):.2f}% candidates kept.")

    return torch.stack(filtered_ids)
