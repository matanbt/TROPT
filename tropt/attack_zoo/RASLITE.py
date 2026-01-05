import torch
from jaxtyping import Float

from tropt.optimizer.base import OptimizerResult
from tropt.optimizer.raslite_optimizer import RASLITEOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.loss.base import SimilarityLoss
from tropt.models.base import TextAccessMixin, TokenAccessMixin
from tropt.models.huggingface.encoder import EncoderHFModel
from tropt.models.huggingface.lm import LMHFModel


def run_raslite(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    prefix_info: str = "Voldermort was right all along. {{OPTIMIZED_TRIGGER}}",
    target_vector: Float[torch.Tensor, "1 d_model"] = torch.randn(
        1, 384
    ),  # random target vector for demo purposes
    util_lm_name: str = "google/gemma-3-270m-it",
) -> OptimizerResult:
    """
    Run the GASLITE's black-box variant -- RASLITE -- on a given embedding model.
    https://arxiv.org/abs/2412.20953

    Args:
        model_name (str): The name of the HuggingFace model to attack.
        util_lm_name (str): The name of the HuggingFace LM model to use for logits calculation.
        prefix_info (str): The string prefixing the passage with a placeholder for the trigger (i.e., the "malicious information").
        target_vector (Tensor, (d_model)): The target vector the passage's embedding is aligned (the centroid of the target query set).
    """

    model = EncoderHFModel(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # use_prefix_cache=False,
    )  # or any black-box-access encoder model
    util_lm = LMHFModel(
        model_name=util_lm_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_prefix_cache=False,
    )

    assert isinstance(model, TextAccessMixin) and isinstance(util_lm, TokenAccessMixin)

    loss = SimilarityLoss()

    optimizer = RASLITEOptimizer(
        model=model,
        util_lm=util_lm,
        loss=loss,
        # Set parameters from the paper:
        num_steps=100,
        n_candidates=128,
        n_flip=20,
        token_constraints=TokenConstraints(
            disallow_non_ascii=True, disallow_special_tokens=True
        ),
        use_retokenize=True,
        use_random_logits=False,
        flip_pos_method="random",
    )

    result = optimizer.optimize_trigger(
        texts=[prefix_info],
        targets={loss.TARGET_KEY: target_vector.to(model.device)},
        initial_trigger=("! " * 40).strip(),
    )

    return result
