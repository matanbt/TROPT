import torch
from jaxtyping import Float

from tropt.common import DEFAULT_INIT_TRIGGER, TargetKey
from tropt.loss.base import SimilarityLoss
from tropt.models import TextAccessMixin, TokenAccessMixin
from tropt.models.huggingface.encoder import EncoderHFModel
from tropt.models.huggingface.lm import LMHFModel
from tropt.optimizer.base import OptimizerResult
from tropt.optimizer.rasliteplus_optimizer import RASLITEPlusOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints


def run_rasliteplus(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    prefix_info: str = "Voldermort was right all along. {{OPTIMIZED_TRIGGER}}",
    target_vector: Float[torch.Tensor, "1 d_model"] = torch.randn(
        1, 384
    ),  # random target vector for demo purposes
    # util_lm_name: str = "google/gemma-3-270m-it",
    initial_trigger: str = DEFAULT_INIT_TRIGGER,
    log_to_wandb: bool = False,
) -> OptimizerResult:
    """
    Run the RASLITEPlus (GASLITEPlus + Black-box) attack on a given embedding model.
    Combines the utility LM approach of RASLITE with the enhanced optimization of GASLITEPlus.

    Args:
        model_name (str): The name of the HuggingFace model to attack;
            - prefixed with "openai/" to use OpenAI embedding models.
        util_lm_name (str): The name of the HuggingFace LM model to use for logits calculation.
        prefix_info (str): The string prefixing the passage with a placeholder for the trigger.
        target_vector (Tensor, (d_model)): The target vector.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name.startswith("openai/"):
        from tropt.models.openai.encoder import EncoderOpenAIModel
        model_name = model_name.replace("openai/", "")
        model = EncoderOpenAIModel(
            model_name=model_name,
        )
    else:
        model = EncoderHFModel(
            model_name=model_name,
        )  # or any black-box-access encoder model
        device = model.device

    util_lm = None  # we dont use logits

    target_vector = model(["paris is the capital of france. It is also known for the Eiffel Tower, its art, culture, and history."])

    loss = SimilarityLoss()

    if log_to_wandb:
        from tropt.tracker.base import WandbTracker
        tracker = WandbTracker("raslite+", "tropt-runs")

    optimizer = RASLITEPlusOptimizer(
        model=model,
        util_lm=util_lm,
        loss=loss,
        # Set parameters (combining RASLITE defaults with GASLITEPlus enhancements):
        num_steps=1500,
        token_constraints=TokenConstraints(
            disallow_non_ascii=True, disallow_special_tokens=True
        ),
        use_retokenize=False,  # OpenAI doesn't respond well to this subroutine; anyway we only evaluate the trigger with string

        # Original features:
        n_candidates=128,
        n_flip=1,

        # Plus features:
        use_random_logits=True,  # True or False
        flip_pos_method="ordered", # "ordered" or "random"
        buffer_size=10,
        n_bulk_flips=1,  # 1, 5, 10, 20

        # [Optional] Advanced features:
        # decline_n_flip_from_step=0.5, # Optional
        # early_stopping_patience=None, # Optional
        # n_grad=1, # Default to 1 (no averaging) unless specified, as RASLITE uses logits directly

        tracker=tracker if log_to_wandb else None,
    )

    result = optimizer.optimize_trigger(
        templates=[prefix_info],
        targets={TargetKey.TARGET_VECTORS: target_vector.to(device)},
        initial_trigger=initial_trigger,
    )

    if log_to_wandb:
        usage_stats = model.get_usage_stats()
        tracker.log(usage_stats)
        tracker.finish()


    return result

