import torch
from jaxtyping import Float

from tropt.attacks.optimizer.base import OptimizerResult
from tropt.attacks.optimizer.gasliteplus_optimizer import GASLITEPlusOptimizer
from tropt.attacks.optimizer.utils.token_constraints import TokenConstraints
from tropt.loss.base import SimilarityLoss
from tropt.models.huggingface.encoder import EncoderHFModel


def run_gaslite_plus(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    prefix_info: str = "Voldermort was right all along. {{OPTIMIZED_TRIGGER}}",
    target_vector: Float[torch.Tensor, "1 d_model"] = torch.randn(
        1, 384
    ),  # random target vector for demo purposes
) -> OptimizerResult:
    """
    Run the GASLITE+ attack recipe on a given embedding model.
    Extension of GASLITE with buffer and adaptive parameters.

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

    optimizer = GASLITEPlusOptimizer(
        model=model,
        loss=loss,
        # Set parameters from the default config:
        num_steps=100,
        n_candidates=128,
        n_grad=10,
        n_flip=30,
        token_constraints=TokenConstraints(
            disallow_non_ascii=True, disallow_special_tokens=True
        ),
        use_retokenize=True,
        # Plus-specific params:
        buffer_size=10,
        decline_n_flip_from_step=0.5,
        early_stopping_patience=5,
        early_stopping_threshold=0.005,
    )

    result = optimizer.optimize_trigger(
        texts=[prefix_info],
        targets={loss.TARGET_KEY: target_vector.to(model.device)},
        initial_trigger=("! " * 100).strip(),
    )

    return result
