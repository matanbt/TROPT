"""
Prompt Recovery reproduction (Williams et al., 2024).

Samples N prompts from DiffusionDB (text-only, streamed), then for each prompt
runs the full recovery pipeline across multiple seeds. Each run generates a
reference image with SD-2.1, inverts it with the PromptRecovery recipe (MAC
over OpenCLIP H/14) from a random 20-token initial trigger, and regenerates
the image from the recovered prompt. Writes the quadruple (original prompt,
original image, recovered prompt, recovered image) per (prompt, seed) so the
companion notebook can build a paper-style qualitative figure.

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

from tropt.recipe_hub.PromptRecovery import recover_prompt_end_to_end
from tropt.tracker import WandbTracker

# ─── Constants ───────────────────────────────────────────────────────────────
WANDB_ENTITY = "matanbt"
WANDB_PROJECT = "tropt-prompt-recovery"

# Paper setup (Williams et al., 2024, p.5–7)
SD_MODEL = "sd2-community/stable-diffusion-2-1"  # mirror of stabilityai/stable-diffusion-2-1 (fn.2)
CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"  # SD-2.1's text encoder
DATASET_NAME = "poloclub/diffusiondb"
# Text-only metadata parquet (2M prompts); load directly to avoid the
# legacy dataset script that newer `datasets` versions no longer support.
DATASET_PARQUET_URL = f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/main/metadata.parquet"

N_PROMPTS = 5
POOL_SIZE = 1000       # rows streamed from head before uniform sampling
NUM_STEPS = 500       # truncated from paper's 3000 for faster iteration
N_INITIAL_TOKENS = 20  # paper uses 8-20; we pick the upper bound with random init
PROMPT_SAMPLING_SEED = 42
SEEDS = [0, 1, 2]      # three independent runs per prompt

# Hand-picked prompts appended to the DiffusionDB sample for a paper-friendly
# qualitative figure: one cool, one funny, one cute.
EXTRA_PROMPTS = [
    # cool
    "a lone astronaut standing on a cliff watching two suns set over a crystalline desert, cinematic lighting, ultra detailed",
    # funny
    "a T-rex trying to eat spaghetti with tiny arms, frustration, photorealistic",
    # cute
    "a tiny dragon curled up asleep in a porcelain teacup, soft pastel lighting",
]

# SD generation
GEN_HEIGHT = 768
GEN_WIDTH = 768
NUM_INFERENCE_STEPS = 50

OUTPUT_DIR = Path("scripts/opt-bench/results/exp3_promrec")


def _sample_prompts(n: int, pool_size: int, seed: int) -> list[str]:
    """Stream a head pool from DiffusionDB text-only and uniform-sample n prompts."""
    ds = load_dataset(
        "parquet", data_files=DATASET_PARQUET_URL, split="train", streaming=True,
    )
    pool = [row["prompt"] for row in ds.take(pool_size)]
    return random.Random(seed).sample(pool, n)


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Sampling {N_PROMPTS} prompts from {DATASET_NAME} metadata.parquet (pool={POOL_SIZE})...")
    prompts = _sample_prompts(N_PROMPTS, POOL_SIZE, PROMPT_SAMPLING_SEED)
    prompts = prompts + EXTRA_PROMPTS
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p[:100]}")

    all_results = []

    for i, original_prompt in enumerate(prompts):
        prompt_dir = OUTPUT_DIR / f"prompt_{i:02d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "original_prompt.txt").write_text(original_prompt, encoding="utf-8")

        for seed in SEEDS:
            print(f"\n{'='*60}\n  Prompt {i+1}/{len(prompts)}  |  seed={seed}\n{'='*60}")
            run_dir = prompt_dir / f"seed_{seed:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            run_name = f"recovery[p={i},seed={seed},steps={NUM_STEPS}]"
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
                    "n_initial_tokens": N_INITIAL_TOKENS,
                    "seed": seed,
                },
            )

            quad = recover_prompt_end_to_end(
                prompt=original_prompt,
                sd_model_name=SD_MODEL,
                clip_model_name=CLIP_MODEL,
                num_steps=NUM_STEPS,
                n_initial_tokens=N_INITIAL_TOKENS,
                seed=seed,
                height=GEN_HEIGHT, width=GEN_WIDTH,
                num_inference_steps=NUM_INFERENCE_STEPS,
                tracker=tracker,
            )

            quad.original_image.save(run_dir / "original_image.png")
            quad.recovered_image.save(run_dir / "recovered_image.png")
            (run_dir / "recovered_prompt.txt").write_text(quad.recovered_prompt, encoding="utf-8")

            print(f"  Original:  {original_prompt!r}")
            print(f"  Recovered: {quad.recovered_prompt!r}")
            print(f"  Best loss: {quad.best_loss:.4f}")

            metadata = {
                "prompt_idx": i,
                "seed": seed,
                "original_prompt": original_prompt,
                "recovered_prompt": quad.recovered_prompt,
                "best_loss": quad.best_loss,
                "num_steps": NUM_STEPS,
                "n_candidates": N_CANDIDATES,
                "n_initial_tokens": N_INITIAL_TOKENS,
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
        "dataset_file": "metadata.parquet",
        "n_prompts": len(prompts),
        "n_sampled_prompts": N_PROMPTS,
        "n_extra_prompts": len(EXTRA_PROMPTS),
        "extra_prompts": EXTRA_PROMPTS,
        "pool_size": POOL_SIZE,
        "num_steps": NUM_STEPS,
        "n_candidates": N_CANDIDATES,
        "n_initial_tokens": N_INITIAL_TOKENS,
        "prompt_sampling_seed": PROMPT_SAMPLING_SEED,
        "seeds": SEEDS,
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
    print(f"All {len(prompts)}×{len(SEEDS)} runs complete.")


if __name__ == "__main__":
    run()
