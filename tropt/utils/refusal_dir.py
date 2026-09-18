"""
Refusal direction extraction and ablation for steering-based attacks.

Implements the difference-in-means method from:
- Arditi et al. (2024): https://arxiv.org/abs/2406.11717
- Loosely based on the original implementation: https://github.com/andyrdt/refusal_direction

Useful for attacks the suppress model refusals via activation steering (e.g., IRIS attack).
"""

import logging
import random
from typing import List, Optional, Tuple

import torch
from jaxtyping import Float

from tropt.model.huggingface.lm import LMHFModel

logger = logging.getLogger(__name__)

ADVBENCH_CSV_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"


def _shuffled_split(instructions: List[str], test_split: float) -> Tuple[List[str], List[str]]:
    """Deterministic train/test split (fixed seed, so runs are reproducible)."""
    instructions = list(instructions)
    random.Random(42).shuffle(instructions)
    n_test = int(len(instructions) * test_split)
    return instructions[n_test:], instructions[:n_test]


def get_harmful_instructions(n_samples: Optional[int] = None, test_split: float = 0.2) -> Tuple[List[str], List[str]]:
    """
    Load harmful instructions from AdvBench dataset.

    Args:
        n_samples: Number of samples to return (None = all)
        test_split: Fraction of data to use for test set

    Returns:
        (train_instructions, test_instructions)
    """
    from datasets import load_dataset

    dataset = load_dataset("csv", data_files=ADVBENCH_CSV_URL, split="train")
    instructions = dataset["goal"]

    if n_samples:
        instructions = instructions[:n_samples]

    return _shuffled_split(instructions, test_split)


def get_harmless_instructions(n_samples: Optional[int] = None, test_split: float = 0.2) -> Tuple[List[str], List[str]]:
    """
    Load harmless instructions from Alpaca dataset.

    Args:
        n_samples: Number of samples to return (None = all)
        test_split: Fraction of data to use for test set

    Returns:
        (train_instructions, test_instructions)
    """
    from datasets import load_dataset

    dataset = load_dataset('tatsu-lab/alpaca')

    # Filter for instructions without inputs
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])
            if n_samples and len(instructions) >= n_samples:
                break

    return _shuffled_split(instructions, test_split)


def extract_activations(
    model: LMHFModel,
    prompts: List[str],
    position: str = "last",
) -> torch.Tensor:
    """
    Extract hidden state activations from all layers for given prompts.

    Args:
        model: LMHFModel instance
        prompts: List of prompts to extract activations from
        position: Which position to extract ("last" or "mean")

    Returns:
        Activations tensor of shape (n_prompts, n_layers, d_model)
    """
    all_activations = []

    for prompt in prompts:
        # Format with chat template
        messages = [{"role": "user", "content": prompt}]
        inputs = model.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"].to(model.device)

        # Forward pass with hidden states (off-pattern raw-model access, needed here)
        with torch.no_grad():
            outputs = model._model(
                inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract activations from all layers
        hidden_states = outputs.hidden_states[1:]  # Skip input embeddings, keep hidden layers

        if position == "last":  # the `INPUT_LAST_TOKEN` position
            # Take last token from each layer
            activations = torch.stack([h[0, -1, :] for h in hidden_states])  # (n_layers, d_model)
        elif position == "mean":
            # Average over sequence
            activations = torch.stack([h[0].mean(dim=0) for h in hidden_states])  # (n_layers, d_model)
        else:
            raise ValueError(f"Unknown position: {position}")

        all_activations.append(activations.cpu())

    return torch.stack(all_activations)  # (n_prompts, n_layers, d_model)


def compute_refusal_directions(
    model: LMHFModel,
    harmful_prompts: Optional[List[str]] = None,
    harmless_prompts: Optional[List[str]] = None,
    n_samples: int = 128,
    position: str = "last",
) -> Float[torch.Tensor, "n_layers d_model"]:
    """
    Compute refusal directions for all layers using difference-in-means.

    Args:
        model: LMHFModel instance
        harmful_prompts: List of harmful prompts (if None, loads from AdvBench)
        harmless_prompts: List of harmless prompts (if None, loads from Alpaca)
        n_samples: Number of samples to use per category; Arditi et al. (2024) used 128
        position: Which position to extract ("last" or "mean"); it's common to use the last token's activations.

    Returns:
        Refusal directions tensor of shape (n_layers, d_model), normalized per layer
    """
    # Load datasets if not provided
    if harmful_prompts is None:
        logger.info("Loading harmful prompts from AdvBench...")
        harmful_train, _ = get_harmful_instructions(n_samples=n_samples)
        harmful_prompts = harmful_train[:n_samples]

    if harmless_prompts is None:
        logger.info("Loading harmless prompts from Alpaca...")
        harmless_train, _ = get_harmless_instructions(n_samples=n_samples)
        harmless_prompts = harmless_train[:n_samples]

    # Extract activations
    logger.info(f"Extracting activations for {len(harmful_prompts)} harmful prompts...")
    harmful_acts = extract_activations(model, harmful_prompts, position)  # (n_prompts, n_layers, d_model)

    logger.info(f"Extracting activations for {len(harmless_prompts)} harmless prompts...")
    harmless_acts = extract_activations(model, harmless_prompts, position)  # (n_prompts, n_layers, d_model)

    # Compute refusal directions (mean difference per layer)
    refusal_dirs = harmful_acts.mean(dim=0) - harmless_acts.mean(dim=0)  # (n_layers, d_model)

    # Normalize per layer
    refusal_dirs = refusal_dirs / (refusal_dirs.norm(dim=1, keepdim=True) + 1e-8)

    logger.info(f"Computed refusal directions: shape={refusal_dirs.shape}")

    return refusal_dirs

