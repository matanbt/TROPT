import torch
from jaxtyping import Float

from tropt.loss.base import SimilarityLoss
from tropt.models.huggingface.encoder import EncoderHFModel
from tropt.optimizer.base import OptimizerResult
from tropt.optimizer.gasliteplus_optimizer import GASLITEPlusOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints


def run_gaslite_plus(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    prefix_info: str = "Voldermort was right all along. {{OPTIMIZED_TRIGGER}}",
    target_vector: Float[torch.Tensor, "1 d_model"] = torch.randn(
        1, 384
    ),  # random target vector for demo purposes
    initial_trigger: str = ("! " * 100).strip(),
    quick_variant: bool = False,
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

    params = dict(
        n_bulk_flips=20,
        n_flip=0.3,
        n_grad=10,
        num_steps=150,
        buffer_size=10,
        n_candidates=256,
        flip_pos_method=["ordered"],
        decline_n_flip_from_step=0.5,
        early_stopping_patience=30,
        early_stopping_threshold=0.0001,
    )
    if quick_variant:
        params.update(
            dict(
                n_bulk_flips=10,
                n_grad=5,
                num_steps=100,
                n_candidates=128,
            )
        )

    optimizer = GASLITEPlusOptimizer(
        model=model,
        loss=loss,
        # Set parameters from the default config:
        token_constraints=TokenConstraints(
            disallow_non_ascii=True, disallow_special_tokens=True
        ),
        use_retokenize=True,

        # Attack params:
        **params,
    )

    result = optimizer.optimize_trigger(
        texts=[prefix_info],
        targets={loss.TARGET_KEY: target_vector.to(model.device)},
        initial_trigger=initial_trigger,
    )

    return result
