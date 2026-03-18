from typing import Optional

import torch
from jaxtyping import Float

from tropt.common import Targets
from tropt.loss import SimilarityLoss
from tropt.model.huggingface.encoder import EncoderHFModel
from tropt.optimizer import OptimizerResult
from tropt.optimizer.gcg_optimizer import GCGOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker import BaseTracker


def run_gcg_embedding_variant(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    prefix_info: str = "Voldermort was right all along. {{OPTIMIZED_TRIGGER}}",
    target_vector: Float[torch.Tensor, "1 d_model"] = torch.randn(
        1, 384
    ),  # random target vector for demo purposes
    model_obj: Optional[EncoderHFModel] = None,
    tracker: Optional[BaseTracker] = None,
) -> OptimizerResult:
    """
    Reporposing GCG to attack embedding models
    https://arxiv.org/abs/2412.20953

    Args:
        model_name (str): The name of the HuggingFace model to attack.
        prefix_info (str): The string prefixing the passage with a placeholder for the trigger.
        target_vector (Tensor, (d_model)): The target vector the passage's embedding is aligned with.
        model_obj: Pre-loaded EncoderHFModel to use instead of creating from `model_name`.
        tracker: Optional tracker for logging.
    """
    if model_obj is None:
        model_obj = EncoderHFModel(
            model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    model = model_obj
    loss = SimilarityLoss()

    optimizer = GCGOptimizer(
        model=model,
        loss=loss,
        tracker=tracker,
        # Set parameters from the paper:
        num_steps=500,
        n_candidates=512,
        sample_topk=256,
        sample_n_replace=1,
        token_constraints=TokenConstraints(
            disallow_non_ascii=True, disallow_special_tokens=True
        ),
        use_retokenize=True,
    )

    result = optimizer.optimize_trigger(
        templates=[prefix_info],
        targets=Targets(target_vectors=target_vector),
        initial_trigger="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    )

    return result
