"""Prompt Recovery for Image Generation Models.

Based on: "Prompt Recovery for Image Generation Models: A Comparative Study
of Discrete Optimizers" (Williams et al., 2025).

Uses CLIP-like models as a proxy: optimize discrete text tokens to maximize
cosine similarity between the text embedding and a target image embedding.
Evaluation follows the paper's protocol: CLIP similarity (text vs. image)
and Text Embedding Similarity (inverted prompt vs. ground-truth prompt).
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from jaxtyping import Float
from torch import Tensor

from tropt.common import Targets
from tropt.loss import SimilarityLoss
from tropt.model.huggingface.clip_encoder import CLIPTextEncoderHFModel
from tropt.optimizer import OptimizerResult
from tropt.optimizer.gcgplus_optimizer import GCGPlusOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.optimizer.utils.token_initializers import get_printable_random_trigger
from tropt.tracker import BaseTracker

# Paper uses 8-20 free tokens; 8 is the default
_DEFAULT_INITIAL_TRIGGER = "! ! ! ! ! ! ! !"


def get_image_embedding_for_clip_model(
    image_path: Optional[str] = None,
    image=None,
    model_name: str = "openai/clip-vit-large-patch14",
) -> Float[Tensor, "1 d_model"]:
    """Encode an image into CLIP's shared embedding space using the vision encoder.

    Loads only the vision side of the full CLIP model, encodes the image,
    and returns the projected image embedding.

    Args:
        image_path: Path to an image file (used if `image` is None).
        image: A PIL Image. If None, loads from `image_path`.
        model_name: CLIP model whose vision encoder to use.

    Returns:
        Image embedding tensor of shape (1, d_model).
    """
    from PIL import Image as PILImage
    from transformers import CLIPModel, CLIPProcessor

    if image is None:
        if image_path is None:
            raise ValueError("Either `image` or `image_path` must be provided.")
        image = PILImage.open(image_path).convert("RGB")

    processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = CLIPModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = clip_model.to(device)

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_emb = clip_model.get_image_features(**inputs)  # (1, d_model)

    # Clean up: we only needed the vision encoder
    del clip_model, processor

    return image_emb.pooler_output


def run_prompt_recovery(
    image=None,
    model_name: str = "openai/clip-vit-large-patch14",
    template: str = "{{OPTIMIZED_TRIGGER}}",
    initial_trigger: str = _DEFAULT_INITIAL_TRIGGER,
    num_steps: int = 3000,
    n_candidates: int = 512,
    tracker: Optional[BaseTracker] = None,
    target_image_path: Optional[str] = None,
    target_image_emb: Optional[Float[Tensor, "d_model"]] = None,
    seed: Optional[int] = None,
) -> OptimizerResult:
    """Recover the prompt that generated a given image using GCG + CLIP.

    Args:
        image: A PIL Image to invert. If None, loads from `target_image_path`.
        model_name: CLIP-like model to use as proxy.
        template: Text template with trigger placeholder.
        initial_trigger: Starting trigger tokens.
        num_steps: Number of GCG optimization steps (paper uses 3000).
        n_candidates: Candidate batch size per step (paper uses 512).
        tracker: Optional experiment tracker.
        target_image_path: Path to an image file (used if `image` is None).
        target_image_emb: Pre-computed image embedding (skips encoding).

    Returns:
        OptimizerResult with `best_trigger_str` as the recovered prompt.
    """
    model_obj = CLIPTextEncoderHFModel(
        model_name=model_name,
    )

    if target_image_emb is None:
        target_image_emb = get_image_embedding_for_clip_model(
            image_path=target_image_path,
            image=image,
            model_name=model_name,
        )

    # Uses MAC optimizer:
    optimizer = GCGPlusOptimizer(
        model=model_obj,
        loss=SimilarityLoss(),
        proxy_model=model_obj,
        tracker=tracker,
        candidate_selection="gradient",
        num_steps=num_steps,
        n_candidates=n_candidates,
        sample_topk=256,
        sample_n_replace=(1, 1),
        momentum=0.6,  # MAC paper's optimal mu
        candidate_oversample_factor=1.1,
        token_constraints=TokenConstraints(),
        use_retokenize=True,
        seed=seed,
    )

    result = optimizer.optimize_trigger(
        templates=[template],
        targets=Targets(target_vectors=target_image_emb),
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
    clip_text_model_obj: Optional[CLIPTextEncoderHFModel] = None,
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
    """
    if image is None:
        if image_path is None:
            raise ValueError("Either `image` or `image_path` must be provided.")
        from PIL import Image
        image = Image.open(image_path).convert("RGB")

    # --- Metric 1: CLIP Similarity (text vs. image) ---
    if clip_text_model_obj is None:
        clip_text_model_obj = CLIPTextEncoderHFModel(
            model_name=clip_model_name,
        )

    image_emb = get_image_embedding_for_clip_model(
        image=image,
        model_name=clip_model_name,
    )

    with torch.no_grad():
        text_emb = clip_text_model_obj.invoke_from_texts(
            [inverted_prompt]
        ).output_embeddings  # (1, d)

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


# ======================= Image Generation =======================


def generate_image_from_prompt(
    prompt: str,
    model_name: str = "black-forest-labs/FLUX.1-dev",
    num_inference_steps: int = 28,
    height: int = 512,
    width: int = 512,
    seed: Optional[int] = None,
):
    """Generate an image from a text prompt using a diffusers pipeline.

    Args:
        prompt: Text prompt to generate from.
        model_name: Diffusers model to use.
        num_inference_steps: Number of denoising steps.
        height: Output image height.
        width: Output image width.
        seed: Random seed for reproducibility.

    Returns:
        PIL Image.
    """
    is_flux = "flux" in model_name.lower()

    if is_flux:
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
    else:
        # Stable Diffusion (e.g. sd2-community/stable-diffusion-2-1)
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )

    pipe.enable_model_cpu_offload()

    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        generator=generator,
    ).images[0]

    del pipe
    return image


# ======================= End-to-end =======================


@dataclass
class PromptRecoveryQuadruple:
    """Prompt → image → recovered prompt → regenerated image."""
    original_prompt: str
    original_image: Any  # PIL.Image.Image
    recovered_prompt: str
    recovered_image: Any  # PIL.Image.Image
    best_loss: float


def recover_prompt_end_to_end(
    prompt: str,
    sd_model_name: str = "sd2-community/stable-diffusion-2-1",
    clip_model_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    num_steps: int = 1000,
    n_candidates: int = 512,
    n_initial_tokens: int = 20,
    seed: int = 0,
    height: int = 768,
    width: int = 768,
    num_inference_steps: int = 50,
    tracker: Optional[BaseTracker] = None,
    initial_trigger: Optional[str] = None,
) -> PromptRecoveryQuadruple:
    """Generate an image from `prompt`, recover the prompt from the image, regenerate.

    Defaults mirror the exp3-promrec reproduction of Williams et al. 2024:
    SD-2.1 + OpenCLIP H/14 (laion2B), 1K GCG steps, random 20-token init.
    """
    import gc
    import random as _random

    original_image = generate_image_from_prompt(
        prompt=prompt,
        model_name=sd_model_name,
        num_inference_steps=num_inference_steps,
        height=height, width=width,
        seed=seed,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if initial_trigger is None:
        from transformers import AutoTokenizer
        # Re-seed `random` immediately before the initializer — SD pipeline init
        # above may have consumed global random state.
        _random.seed(seed)
        initial_trigger = get_printable_random_trigger(
            trigger_len=n_initial_tokens,
            tokenizer=AutoTokenizer.from_pretrained(clip_model_name),
        )

    result = run_prompt_recovery(
        image=original_image,
        model_name=clip_model_name,
        initial_trigger=initial_trigger,
        num_steps=num_steps,
        n_candidates=n_candidates,
        tracker=tracker,
        seed=seed,
    )
    recovered_prompt = result.best_trigger_str
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    recovered_image = generate_image_from_prompt(
        prompt=recovered_prompt,
        model_name=sd_model_name,
        num_inference_steps=num_inference_steps,
        height=height, width=width,
        seed=seed,
    )

    return PromptRecoveryQuadruple(
        original_prompt=prompt,
        original_image=original_image,
        recovered_prompt=recovered_prompt,
        recovered_image=recovered_image,
        best_loss=float(result.best_loss),
    )
