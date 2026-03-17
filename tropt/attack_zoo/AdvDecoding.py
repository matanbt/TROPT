"""
Adversarial Decoding (AdvDecoding) Attack Implementation
https://arxiv.org/abs/2410.02163
"""

import torch
from jaxtyping import Float

from tropt.loss import CombinedLoss, InputReadabilityLoss, PrefillCELoss, SimilarityLoss
from tropt.model.huggingface.encoder import EncoderHFModel
from tropt.model.huggingface.lm import LMHFModel
from tropt.optimizer import OptimizerResult
from tropt.optimizer.beamsearch_optimizer import BeamSearchOptimizer


def run_advdecoding_encoder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    util_lm_name: str = "google/gemma-3-270m-it",
    prefix_info: str = "Voldermort was right all along. {{OPTIMIZED_TRIGGER}}",
    target_vector: Float[torch.Tensor, "1 d_model"] = torch.randn(
        1, 384
    ),  # random target vector for demo purposes
) -> OptimizerResult:
    """
    Run the AdvDecoding encoder's corpus poisoning attack.

    Args:
        model_name (str): The name of the HuggingFace model to attack.
        util_lm_name (str): The name of the utility language model for scoring readability.
        prefix_info (str): The malicious information prompt with a placeholder for the trigger.
        target_vector (Float[torch.Tensor, "1 d_model"]): The target embedding vector to align with.

    Returns:
        OptimizerResult: Optimization results containing the best trigger and loss trajectory.

    References:
        AdvDecoding paper: https://arxiv.org/abs/2410.02163
        Algorithm 1: Adversarial Decoding with Multiple Scorers

    Notes:
    - AdvDecoding is a variant of BEAST, but uses specific set of params, a combined loss,
      and a util LM to filter the beam candidates. Thus, we use BEASTOptimizer here.
    - This function can be easily extended to support jailbreak attacks (by modifying the model and losses).
    """
    # Initialize model
    model =EncoderHFModel(
        model_name=model_name,
    )
    util_lm = LMHFModel(  # for per-step logits
        model_name=util_lm_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Parameters (from the paper):
    beam_width: int = 30  # m in the paper; mostly use 30 [k1 in BEAST]
    max_length: int = 30  # Paper uses 30
    top_k: int = 10  # Paper uses top_k=10 logits filtering
    temperature: float = 1.0  # as there is no sampling anyway
    prefix_prompt = "Write a sentence with a lot of triggers. {{OPTIMIZED_TRIGGER}}"  # a prompt for util LM to compute logits of the trigger; prompt is taken from the paper

    loss = CombinedLoss(
        loss_funcs=[
            SimilarityLoss(),  # Main attack loss: align to target embedding
            InputReadabilityLoss(),
            # can add here additional "scorers" as losses
        ],
        weights=[1.0, 1.0],  # Weights for each loss component
    )

    # Initialize optimizer with AdvDecoding parameters
    optimizer = BeamSearchOptimizer(
        model=model,
        loss=loss,
        util_lm=util_lm,
        num_steps=max_length,  # num steps = length of trigger to generate
        beam_size=beam_width,
        top_k=top_k,
        branching_factor=top_k,
        temperature=temperature,
        use_model_with_token_inputs=False,  # computes the target model loss in text-level
    )

    # Run optimization
    result = optimizer.optimize_trigger(
        templates=[prefix_info],  # templates for the target model
        targets=dict(target_vectors=target_vector.to(model.device)),
        util_lm_templates=[prefix_prompt],  # templates for the LM logits
    )

    return result
