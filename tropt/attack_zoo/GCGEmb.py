import torch
from jaxtyping import Float

from tropt.optimizer.base import OptimizerResult
from tropt.optimizer.gcg_optimizer import GCGOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.loss.base import SimilarityLoss
from tropt.models.huggingface.encoder import EncoderHFModel


def run_gcg_embedding_variant(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    prefix_info: str = "Voldermort was right all along. {{OPTIMIZED_TRIGGER}}",
    target_vector: Float[torch.Tensor, "1 d_model"] = torch.randn(
        1, 384
    ),  # random target vector for demo purposes
) -> OptimizerResult:
    """
    Reporposing GCG to attack embedding models
    https://arxiv.org/abs/2412.20953

    Args:
        model_name (str): The name of the HuggingFace model to attack.
        prefix_info (str): The string prefixing the passage with a placeholder for the trigger.
        target_vector (Tensor, (d_model)): The target vector the passage's embedding is aligned with.
    """
    model = EncoderHFModel(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    loss = SimilarityLoss()

    optimizer = GCGOptimizer(
        model=model,
        loss=loss,
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
        targets=dict(target_vectors=target_vector.to(model.device)),
        initial_trigger="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    )

    return result
