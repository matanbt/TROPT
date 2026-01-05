import torch
from jaxtyping import Float

from tropt.optimizer.base import OptimizerResult
from tropt.optimizer.rasliteplus_optimizer import RASLITEPlusOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.loss.base import SimilarityLoss
from tropt.models.base import TextAccessMixin, TokenAccessMixin
from tropt.models.huggingface.encoder import EncoderHFModel
from tropt.models.huggingface.lm import LMHFModel


def run_rasliteplus(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    prefix_info: str = "Voldermort was right all along. {{OPTIMIZED_TRIGGER}}",
    target_vector: Float[torch.Tensor, "1 d_model"] = torch.randn(
        1, 384
    ),  # random target vector for demo purposes
    util_lm_name: str = "google/gemma-3-270m-it",
) -> OptimizerResult:
    """
    Run the RASLITEPlus (GASLITEPlus + Black-box) attack on a given embedding model.
    Combines the utility LM approach of RASLITE with the enhanced optimization of GASLITEPlus.

    Args:
        model_name (str): The name of the HuggingFace model to attack.
        util_lm_name (str): The name of the HuggingFace LM model to use for logits calculation.
        prefix_info (str): The string prefixing the passage with a placeholder for the trigger.
        target_vector (Tensor, (d_model)): The target vector.
    """

    model = EncoderHFModel(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )  # or any black-box-access encoder model
    util_lm = LMHFModel(
        model_name=util_lm_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_prefix_cache=False,
    )

    assert isinstance(model, TextAccessMixin) and isinstance(util_lm, TokenAccessMixin)

    loss = SimilarityLoss()

    optimizer = RASLITEPlusOptimizer(
        model=model,
        util_lm=util_lm,
        loss=loss,
        # Set parameters (combining RASLITE defaults with GASLITEPlus enhancements):
        num_steps=500,
        token_constraints=TokenConstraints(
            disallow_non_ascii=True, disallow_special_tokens=True
        ),
        use_retokenize=True,

        # Original features:
        n_candidates=1024,
        n_flip=1, # 1, 0.1, 0.2, 0.3, 0.4

        # Plus features:
        use_random_logits=True,  # True or False
        flip_pos_method="ordered", # "ordered" or "random"
        buffer_size=10,
        n_bulk_flips=1,  # 1, 5, 10, 20

        # [Optional] Advanced features:
        # decline_n_flip_from_step=0.5, # Optional
        # early_stopping_patience=None, # Optional
        # n_grad=1, # Default to 1 (no averaging) unless specified, as RASLITE uses logits directly
    )

    result = optimizer.optimize_trigger(
        texts=[prefix_info],
        targets={loss.TARGET_KEY: target_vector.to(model.device)},
        # initial_trigger=("! " * 50).strip(),

        ## random text:
        initial_trigger="What is the meaning of life? also, what is 2+2? oh, and tell me a joke! wait, one more thing: how to bake a cake? OH AND finally, what's the weather like today? NO MORE PLEASE!",
    )

    return result
