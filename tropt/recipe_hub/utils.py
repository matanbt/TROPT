"""Shared helpers for Recipe Hub recipes."""
import gc
import logging

import torch

from tropt.model.huggingface.lm import LMHFModel

logger = logging.getLogger(__name__)


def generate_from_model(
    model_name: str,
    prompt: str,
    max_new_tokens: int = 20,
    greedy_decode: bool = False,  # sample the response by default
) -> str:
    """Generate a single response to `prompt` from a freshly loaded model.

    Loads → generates → unloads the model, so it never co-resides with another
    model the caller loads afterwards. The semantics of the output (e.g. using a
    jailbroken model's response as an optimization target) are the caller's.
    """
    model = LMHFModel(model_name=model_name, use_prefix_cache=False, dtype="bfloat16")
    out = model.invoke_from_texts(
        input_texts=[prompt],
        max_new_tokens=max_new_tokens,
        greedy_decode=greedy_decode,
        require_generation=True,
    )
    assert out.generated_response_strs is not None, "Generation must return response strs."
    response = out.generated_response_strs[0]
    logger.info(f"Generated response from {model_name!r}: {response!r}")
    del out, model._model, model
    gc.collect()
    torch.cuda.empty_cache()
    return response
