"""
Prompt Recovery reproduction (Williams et al., 2024).

Samples N prompts from DiffusionDB (text-only, streamed), generates reference
images with SD-2.1, inverts each with the PromptRecovery recipe (MAC over
OpenCLIP H/14), then regenerates from the recovered prompt. Writes original
prompt, original image, recovered prompt, and recovered image per sample so
the companion notebook can build a paper-style qualitative figure.

Usage
-----
  python scripts/opt-bench/exp3-promrec.py
"""

import gc
import json
import random
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset

from tropt.recipe_hub.PromptRecovery import (
    generate_image_from_prompt,
    run_prompt_recovery,
)
from tropt.tracker import WandbTracker

# ─── Constants ───────────────────────────────────────────────────────────────
WANDB_ENTITY = "matanbt"
WANDB_PROJECT = "tropt-prompt-recovery"

# Paper setup (Williams et al., 2024, p.5–7)
SD_MODEL = "sd2-community/stable-diffusion-2-1"  # mirror of stabilityai/stable-diffusion-2-1 (fn.2)
CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"  # SD-2.1's text encoder
DATASET_NAME = "poloclub/diffusiondb"
DATASET_CONFIG = "2m_text_only"  # prompts-only; streamed (avoids full image dataset download)

N_PROMPTS = 5
POOL_SIZE = 1000      # rows streamed from head before uniform sampling
NUM_STEPS = 3000      # paper p.5: convergence point
N_CANDIDATES = 512    # paper p.7
SEED = 42

# SD generation
GEN_HEIGHT = 768
GEN_WIDTH = 768
NUM_INFERENCE_STEPS = 50

OUTPUT_DIR = Path("scripts/opt-bench/results/exp3_promrec")


def _sample_prompts(n: int, pool_size: int, seed: int) -> list[str]:
    """Stream a head pool from DiffusionDB text-only and uniform-sample n prompts."""
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)
    pool = [row["prompt"] for row in ds.take(pool_size)]
    return random.Random(seed).sample(pool, n)


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {N_PROMPTS} prompts from {DATASET_NAME}/{DATASET_CONFIG} (pool={POOL_SIZE})...")
    prompts = _sample_prompts(N_PROMPTS, POOL_SIZE, SEED)
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p[:100]}")

    all_results = []

    for i, original_prompt in enumerate(prompts):
        print(f"\n{'='*60}\n  Prompt {i+1}/{N_PROMPTS}\n{'='*60}")
        run_dir = OUTPUT_DIR / f"prompt_{i:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Generate the original image ---
        print(f"[1/3] Generating original image ({SD_MODEL})...")
        original_image = generate_image_from_prompt(
            prompt=original_prompt,
            model_name=SD_MODEL,
            num_inference_steps=NUM_INFERENCE_STEPS,
            height=GEN_HEIGHT, width=GEN_WIDTH,
            seed=SEED,
        )
        original_image.save(run_dir / "original_image.png")
        (run_dir / "original_prompt.txt").write_text(original_prompt, encoding="utf-8")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- 2. Prompt recovery (MAC + CLIP H/14) ---
        run_name = f"recovery[p={i},steps={NUM_STEPS}]"
        tracker = WandbTracker(
            run_name,
            tags=["prompt-recovery", "mac", CLIP_MODEL.split("/")[-1]],
            project_name=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            experiment_config={
                "attack": "mac",
                "task": "prompt_recovery",
                "sd_model": SD_MODEL,
                "clip_model": CLIP_MODEL,
                "prompt_idx": i,
                "original_prompt": original_prompt,
                "num_steps": NUM_STEPS,
                "n_candidates": N_CANDIDATES,
                "seed": SEED,
            },
        )

        print(f"[2/3] Recovering prompt ({NUM_STEPS} steps)...")
        result = run_prompt_recovery(
            image=original_image,
            model_name=CLIP_MODEL,
            num_steps=NUM_STEPS,
            n_candidates=N_CANDIDATES,
            tracker=tracker,
        )
        recovered_prompt = result.best_trigger_str
        print(f"  Original:  {original_prompt!r}")
        print(f"  Recovered: {recovered_prompt!r}")
        print(f"  Best loss: {result.best_loss:.4f}")

        (run_dir / "recovered_prompt.txt").write_text(recovered_prompt, encoding="utf-8")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- 3. Regenerate image from recovered prompt ---
        print("[3/3] Regenerating image from recovered prompt...")
        recovered_image = generate_image_from_prompt(
            prompt=recovered_prompt,
            model_name=SD_MODEL,
            num_inference_steps=NUM_INFERENCE_STEPS,
            height=GEN_HEIGHT, width=GEN_WIDTH,
            seed=SEED,
        )
        recovered_image.save(run_dir / "recovered_image.png")

        metadata = {
            "prompt_idx": i,
            "original_prompt": original_prompt,
            "recovered_prompt": recovered_prompt,
            "best_loss": float(result.best_loss),
            "num_steps": NUM_STEPS,
            "n_candidates": N_CANDIDATES,
            "seed": SEED,
        }
        with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        all_results.append(metadata)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "experiment": "prompt_recovery",
        "sd_model": SD_MODEL,
        "clip_model": CLIP_MODEL,
        "dataset": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "n_prompts": N_PROMPTS,
        "pool_size": POOL_SIZE,
        "num_steps": NUM_STEPS,
        "n_candidates": N_CANDIDATES,
        "seed": SEED,
        "gen_height": GEN_HEIGHT,
        "gen_width": GEN_WIDTH,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to {summary_path}")
    print(f"All {N_PROMPTS} runs complete.")


if __name__ == "__main__":
    run()
