"""
RS Encoder attack recipe — black-box Random Search on embedding models.

Works with both HuggingFace and OpenAI encoder models. Optimizes a discrete
trigger to align a passage's embedding with a target vector using cosine
similarity loss.
"""

from typing import Optional, Union

import torch
from jaxtyping import Float

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER, Targets
from tropt.loss import SimilarityLoss
from tropt.model.huggingface.encoder import EncoderHFModel
from tropt.model.openai.encoder import EncoderOpenAIModel
from tropt.optimizer import OptimizerResult
from tropt.optimizer.rs_optimizer import RandomSearchOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.optimizer.utils.token_initializers import get_printable_random_trigger
from tropt.tracker import BaseTracker


def run_rs_encoder(
    template: str = "Malicious passage. {{OPTIMIZED_TRIGGER}}",
    target_vector: Float[torch.Tensor, "1 d_model"] = torch.randn(1, 768),
    # --- model ---
    model_name: str = "intfloat/e5-base-v2",
    use_openai: bool = False,
    model_obj: Optional[Union[EncoderHFModel, EncoderOpenAIModel]] = None,
    # --- misc ---
    tracker: Optional[BaseTracker] = None,
    seed: Optional[int] = None,
) -> OptimizerResult:
    """Run a black-box Random Search attack on an embedding model.

    Args:
        template: Template string with ``{{OPTIMIZED_TRIGGER}}`` placeholder.
        target_vector: Target embedding to align toward (shape ``1 x d_model``).
        model_name: HuggingFace model ID or OpenAI model name (e.g. ``"text-embedding-3-small"``).
        use_openai: If True, load ``EncoderOpenAIModel`` instead of ``EncoderHFModel``.
        model_obj: Pre-loaded model to use instead of creating from ``model_name``.
    """
    assert OPTIMIZED_TRIGGER_PLACEHOLDER in template, (
        f"Template must contain {OPTIMIZED_TRIGGER_PLACEHOLDER}"
    )

    if model_obj is None:
        if use_openai:
            model_obj = EncoderOpenAIModel(model_name=model_name)
        else:
            model_obj = EncoderHFModel(model_name=model_name)

    tc = TokenConstraints()
    initial_trigger = get_printable_random_trigger(
        trigger_len=50, tokenizer=model_obj.tokenizer, token_constraints=tc,
    )

    optimizer = RandomSearchOptimizer(
        model=model_obj,
        loss=SimilarityLoss(),
        tracker=tracker,
        seed=seed,
        num_steps=500,
        n_candidates=128,
        token_constraints=tc,
        mutation_mode="block_random",
        schedule="fixed",
        initial_block_len=4,
        patience=25,
    )

    return optimizer.optimize_trigger(
        templates=[template],
        targets=Targets(target_vectors=target_vector),
        initial_trigger=initial_trigger,
    )
