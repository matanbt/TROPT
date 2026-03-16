"""
Refusal direction extraction and ablation for steering-based attacks.

Implements the difference-in-means method from:
- Arditi et al. (2024): https://arxiv.org/abs/2406.11717
- Original implementation: https://github.com/andyrdt/refusal_direction

Useful for attacks the suppress model refusals via activation steering (e.g., IRIS attack), or for
generating jailbroken target outputs via refusal ablation (a.k.a. on thegreat abliterated model).
"""

import io
import logging
from typing import Callable, List, Optional, Tuple

import pandas as pd
import requests
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from tropt.model.huggingface.lm import LMHFModel

logger = logging.getLogger(__name__)


def get_harmful_instructions(n_samples: Optional[int] = None, test_split: float = 0.2) -> Tuple[List[str], List[str]]:
    """
    Load harmful instructions from AdvBench dataset.

    Args:
        n_samples: Number of samples to return (None = all)
        test_split: Fraction of data to use for test set

    Returns:
        (train_instructions, test_instructions)
    """
    url = 'https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv'
    response = requests.get(url)
    dataset = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    instructions = dataset['goal'].tolist()

    if n_samples:
        instructions = instructions[:n_samples]

    train, test = train_test_split(instructions, test_size=test_split, random_state=42)
    return train, test


def get_harmless_instructions(n_samples: Optional[int] = None, test_split: float = 0.2) -> Tuple[List[str], List[str]]:
    """
    Load harmless instructions from Alpaca dataset.

    Args:
        n_samples: Number of samples to return (None = all)
        test_split: Fraction of data to use for test set

    Returns:
        (train_instructions, test_instructions)
    """
    dataset = load_dataset('tatsu-lab/alpaca')

    # Filter for instructions without inputs
    instructions = []
    for i in range(len(dataset['train'])):
        if dataset['train'][i]['input'].strip() == '':
            instructions.append(dataset['train'][i]['instruction'])
            if n_samples and len(instructions) >= n_samples:
                break

    train, test = train_test_split(instructions, test_size=test_split, random_state=42)
    return train, test


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
        ).to(model.device)

        # Forward pass with hidden states
        with torch.no_grad():
            outputs = model.model(
                inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract activations from all layers
        # hidden_states is tuple: (embeddings, layer1, layer2, ..., layerN)
        hidden_states = outputs.hidden_states[1:]  # Skip embeddings, keep transformer layers

        if position == "last":
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
    n_samples: int = 400,
    position: str = "last",
) -> torch.Tensor:
    """
    Compute refusal directions for all layers using difference-in-means.

    Args:
        model: LMHFModel instance
        harmful_prompts: List of harmful prompts (if None, loads from AdvBench)
        harmless_prompts: List of harmless prompts (if None, loads from Alpaca)
        n_samples: Number of samples to use per category
        position: Which position to extract ("last" or "mean")

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


def create_ablation_hook(
    refusal_dir: torch.Tensor,
    ablate: bool = True,
    scale: float = 1.0,
    targeted_positions: Optional[slice] = None,
) -> Callable:
    """
    Create a hook function for ablating or adding refusal direction.

    Args:
        refusal_dir: Refusal direction vector (d_model,)
        ablate: If True, remove refusal direction; if False, add it
        scale: Scale factor for addition (only used if ablate=False)
        targeted_positions: Token positions to apply to (None = all positions)

    Returns:
        Hook function compatible with layer.register_forward_hook()
    """
    def hook(module, args, output):
        # Handle both tuple and non-tuple outputs
        if isinstance(output, tuple):
            act, cache = output
        else:
            act = output
            cache = None

        if ablate:
            # Project out refusal direction
            proj = (act * refusal_dir).sum(dim=-1, keepdim=True)  # (batch, seq_len, 1)
            new_act = act - refusal_dir * proj
        else:
            # Add refusal direction
            new_act = act + refusal_dir * scale

        # Apply only to targeted positions if specified
        if targeted_positions is not None:
            act = act.clone()
            act[:, targeted_positions, :] = new_act[:, targeted_positions, :]
        else:
            act = new_act

        # Return in same format as input
        if cache is not None:
            return act, cache
        return act

    return hook


def ablate_refusal_direction(
    model: LMHFModel,
    refusal_dirs: torch.Tensor,
    source_layer: int,
    targeted_layers: Optional[List[int]] = None,
    targeted_positions: Optional[slice] = None,
    ablate: bool = True,
    scale: float = 1.0,
) -> Tuple[LMHFModel, Callable]:
    """
    Apply hooks to model for ablating/adding refusal direction.

    Args:
        model: LMHFModel instance
        refusal_dirs: Refusal directions for all layers (n_layers, d_model)
        source_layer: Which layer's refusal direction to use
        targeted_layers: Which layers to apply hooks to (None = all)
        targeted_positions: Token positions to apply to (None = all)
        ablate: If True, remove refusal; if False, add it
        scale: Scale factor for addition

    Returns:
        (model, remove_hooks_fn) -- call remove_hooks_fn() to restore model
    """
    refusal_dir = refusal_dirs[source_layer].to(model.device)

    hook_fn = create_ablation_hook(
        refusal_dir=refusal_dir,
        ablate=ablate,
        scale=scale,
        targeted_positions=targeted_positions,
    )

    # Register hooks to model layers
    hook_handles = []
    for i, layer in enumerate(model.model.model.layers):
        if targeted_layers is not None and i not in targeted_layers:
            continue
        hook_handles.append(layer.register_forward_hook(hook_fn))

    # Function to remove hooks
    def remove_hooks():
        for handle in hook_handles:
            handle.remove()

    return model, remove_hooks


def generate_jailbroken_responses(
    model: LMHFModel,
    prompts: List[str],
    refusal_dirs: torch.Tensor,
    source_layer: int,
    max_new_tokens: int = 50,
    **ablate_kwargs,
) -> List[str]:
    """
    Generate responses with refusal direction ablated.

    Args:
        model: LMHFModel instance
        prompts: List of prompts to generate from
        refusal_dirs: Refusal directions (n_layers, d_model)
        source_layer: Which layer's direction to use
        max_new_tokens: Maximum tokens to generate
        **ablate_kwargs: Additional arguments for ablate_refusal_direction()

    Returns:
        List of generated responses (jailbroken)
    """
    responses = []

    for prompt in prompts:
        # Apply refusal ablation
        model, remove_hooks = ablate_refusal_direction(
            model=model,
            refusal_dirs=refusal_dirs,
            source_layer=source_layer,
            **ablate_kwargs,
        )

        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        inputs = model.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=model.tokenizer.eos_token_id,
            )

        # Decode response only (skip prompt)
        response = model.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True,
        )
        responses.append(response)

        # Remove hooks
        remove_hooks()

    return responses
