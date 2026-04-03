"""Prompt Recovery for Image Generation Models.

Based on: "Prompt Recovery for Image Generation Models: A Comparative Study
of Discrete Optimizers" (Williams et al., 2025).

Uses CLIP-like models as a proxy: optimize discrete text tokens to maximize
cosine similarity between the text embedding and a target image embedding.
Evaluation follows the paper's protocol: CLIP similarity (text vs. image)
and Text Embedding Similarity (inverted prompt vs. ground-truth prompt).
"""

from dataclasses import dataclass
from typing import Optional

import torch

from tropt.common import Targets
from tropt.loss import SimilarityLoss
from tropt.model.huggingface.clip_encoder import CLIPEncoderHFModel
from tropt.optimizer import OptimizerResult
from tropt.optimizer.gcg_optimizer import GCGOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker import BaseTracker

# Paper uses 8-20 free tokens; 8 is the default
_DEFAULT_INITIAL_TRIGGER = "! ! ! ! ! ! ! !"


def run_prompt_recovery(
    image=None,
    image_path: Optional[str] = None,
    model_name: str = "openai/clip-vit-large-patch14",
    template: str = "{{OPTIMIZED_TRIGGER}}",
    initial_trigger: str = _DEFAULT_INITIAL_TRIGGER,
    num_steps: int = 3000,
    n_candidates: int = 512,
    model_obj: Optional[CLIPEncoderHFModel] = None,
    tracker: Optional[BaseTracker] = None,
) -> OptimizerResult:
    """Recover the prompt that generated a given image using GCG + CLIP.

    Args:
        image: A PIL Image to invert. If None, loads from `image_path`.
        image_path: Path to an image file (used if `image` is None).
        model_name: CLIP-like model to use as proxy.
        template: Text template with trigger placeholder.
        initial_trigger: Starting trigger tokens.
        num_steps: Number of GCG optimization steps (paper uses 3000).
        n_candidates: Candidate batch size per step (paper uses 512).
        model_obj: Pre-loaded CLIPEncoderHFModel.
        tracker: Optional experiment tracker.

    Returns:
        OptimizerResult with `best_trigger_str` as the recovered prompt.
    """
    if image is None:
        if image_path is None:
            raise ValueError("Either `image` or `image_path` must be provided.")
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

    if model_obj is None:
        model_obj = CLIPEncoderHFModel(
            model_name=model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    # Encode the target image into CLIP space
    image_embedding = model_obj.encode_images(image)  # (1, d_model)

    optimizer = GCGOptimizer(
        model=model_obj,
        loss=SimilarityLoss(),
        tracker=tracker,
        num_steps=num_steps,
        n_candidates=n_candidates,
        sample_topk=256,
        sample_n_replace=1,
        token_constraints=TokenConstraints(
            disallow_non_ascii=True, disallow_special_tokens=True
        ),
        use_retokenize=True,
    )

    result = optimizer.optimize_trigger(
        templates=[template],
        targets=Targets(target_vectors=image_embedding),
        initial_trigger=initial_trigger,
    )

    return result


# ======================= Evaluation =======================


@dataclass
class PromptRecoveryEvaluation:
    """Evaluation results for prompt recovery."""
    clip_similarity: float
    text_embedding_similarity: Optional[float] = None


def evaluate_prompt_recovery(
    inverted_prompt: str,
    image=None,
    image_path: Optional[str] = None,
    original_prompt: Optional[str] = None,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    text_sim_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    clip_model_obj: Optional[CLIPEncoderHFModel] = None,
) -> PromptRecoveryEvaluation:
    """Evaluate a recovered prompt following the paper's protocol.

    Metrics:
        1. CLIP similarity: cosine similarity between the inverted prompt's
           text embedding and the original image embedding in CLIP space.
        2. Text Embedding Similarity (optional, requires `original_prompt`):
           cosine similarity between sentence embeddings of the inverted and
           original prompts using all-MiniLM-L6-v2.

    Note: The paper also uses FID/KID (image-to-image), which requires a
    text-to-image generation pipeline and is not included here.

    Args:
        inverted_prompt: The recovered/inverted prompt text.
        image: Target PIL Image (or provide `image_path`).
        image_path: Path to the target image.
        original_prompt: Ground-truth prompt (if available) for text similarity.
        clip_model_name: CLIP model for computing CLIP similarity.
        text_sim_model_name: Sentence encoder for text embedding similarity.
        clip_model_obj: Pre-loaded CLIPEncoderHFModel (reuse from optimization).

    Returns:
        PromptRecoveryEvaluation with computed metrics.
    """
    if image is None:
        if image_path is None:
            raise ValueError("Either `image` or `image_path` must be provided.")
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

    # --- Metric 1: CLIP Similarity (text vs. image) ---
    if clip_model_obj is None:
        clip_model_obj = CLIPEncoderHFModel(
            model_name=clip_model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    with torch.no_grad():
        text_emb = clip_model_obj.invoke_from_texts([inverted_prompt]).output_embeddings  # (1, d)
        image_emb = clip_model_obj.encode_images(image)  # (1, d)

        # Cosine similarity
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        clip_sim = (text_emb * image_emb).sum(dim=-1).item()

    # --- Metric 2: Text Embedding Similarity (optional) ---
    text_sim = None
    if original_prompt is not None:
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(text_sim_model_name)
        embeddings = st_model.encode(
            [inverted_prompt, original_prompt], convert_to_tensor=True
        )
        inv_emb = embeddings[0]
        orig_emb = embeddings[1]
        inv_emb = inv_emb / inv_emb.norm()
        orig_emb = orig_emb / orig_emb.norm()
        text_sim = (inv_emb * orig_emb).sum().item()

    return PromptRecoveryEvaluation(
        clip_similarity=clip_sim,
        text_embedding_similarity=text_sim,
    )
