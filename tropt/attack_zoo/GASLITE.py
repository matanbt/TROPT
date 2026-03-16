import torch
from jaxtyping import Float

from tropt.common import Targets
from tropt.loss import SimilarityLoss
from tropt.models.huggingface.encoder import EncoderHFModel
from tropt.optimizer import OptimizerResult
from tropt.optimizer.gaslite_optimizer import GASLITEOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints


def run_gaslite(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    prefix_info: str = "Voldermort was right all along. {{OPTIMIZED_TRIGGER}}",
    target_vector: Float[torch.Tensor, "1 d_model"] = torch.randn(
        1, 384
    ),  # random target vector for demo purposes
) -> OptimizerResult:
    """
    Run the GASLITE's attack recipe on a given embedding model.
    https://arxiv.org/abs/2412.20953

    Args:
        model_name (str): The name of the HuggingFace model to attack.
        prefix_info (str): The string prefixing the passage with a placeholder for the trigger (i.e., the "malicious information").
        target_vector (Tensor, (d_model)): The target vector the passage's embedding is aligned (the centroid of the target query set).
    """

    model = EncoderHFModel(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    loss = SimilarityLoss()

    optimizer = GASLITEOptimizer(
        model=model,
        loss=loss,
        # Set parameters from the paper:
        n_candidates=128,
        n_grad=50,
        n_flip=20,
        token_constraints=TokenConstraints(
            disallow_non_ascii=True, disallow_special_tokens=True
        ),
        use_retokenize=True,
    )

    result = optimizer.optimize_trigger(
        templates=[prefix_info],
        targets=Targets(
            target_vectors=target_vector
        ),
        initial_trigger=("! " * 100).strip(),
    )

    return result
