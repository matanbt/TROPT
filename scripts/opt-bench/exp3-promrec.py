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
from datetime import datetime
from pathlib import Path

import torch

from tropt.recipe_hub.PromptRecovery import recover_prompt_end_to_end
from tropt.tracker import WandbTracker

# ─── Constants ───────────────────────────────────────────────────────────────
WANDB_ENTITY = "matanbt"
WANDB_PROJECT = "tropt-prompt-recovery"

# Paper setup (Williams et al., 2024, p.5–7)
SD_MODEL = "sd2-community/stable-diffusion-2-1"  # mirror of stabilityai/stable-diffusion-2-1 (fn.2)
CLIP_MODEL = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"  # SD-2.1's text encoder
DATASET_NAME = "poloclub/diffusiondb"
# Prompts pre-sampled once from DiffusionDB's metadata.parquet to avoid
# fetching from HF on compute nodes with no internet access. Sampling
# procedure: stream first POOL_SIZE=1000 rows from text-only metadata.parquet,
# uniform-sample N_PROMPTS=5 with random.Random(PROMPT_SAMPLING_SEED=42).
# See scripts/opt-bench/sample_diffusiondb_prompts.py to re-sample.
SAMPLED_PROMPTS = [
    # 'anthropomorphic ferret animal, dressed as cute raver girl dancer, dance club in tropical paradise, many woodland creatures dancing, concept design, contrast, hot toy, kim jung gi, greg rutkowski, zabrocki, karlkka, jayison devadas, trending on artstation, 8 k, ultra wide angle, pincushion lens effect ',
    # 'render of dreamy beautiful landscape, fantasy dreamy, dreams beach, artger, large scale, details vintage photo hyper realistic ultra realistic photo realistic photography, unreal engine, high detailed, 8 k ',
    # 'a painting by edward hopper of scenes from the mad max movie universe. ',
    # 'overcook videogame in the style of hr giger ',
    # 'a beautiful painting of the heart of pripyat by nekro and pascal blanche and syd mead and greg rutkowski and sin jong hun and victo ngai and simon stalenhag and chris voy. in style of cg art. ray tracing, cel shading, 3 d. ue 5. hyper detailed. realistic. maya. octane render. ',
]

OPTIMIZER_TYPES = ["gcg", "adv_decoding"]  # MAC and beam-search-decoding
TRIGGER_LENS = [5, 10, 15, 20]             # paper uses 8-20; sweep for ablation
SEEDS = [0, 1]                             # two independent runs per config

# Hand-picked prompts appended to the DiffusionDB sample for a paper-friendly
# qualitative figure: one cool, one funny, one cute.
EXTRA_PROMPTS = [
    # cool
    "a lone astronaut standing on a cliff watching two suns set over a crystalline desert, cinematic lighting, ultra detailed",
    # funny
    "a T-rex trying to eat spaghetti with tiny arms, frustration, photorealistic",
    # cute
    "a tiny dragon curled up asleep in a porcelain teacup, soft pastel lighting",

    # epic
    "a colossal whale drifting silently through a sky of swirling nebulae, bioluminescent fins, dreamlike atmosphere, cinematic scale",
    # funny
    "a raccoon in a tiny lab coat presenting a chaotic equation on a chalkboard, chalk dust floating, dramatic lighting",
    # cute
    "a family of hedgehogs having a candlelit dinner on a mushroom table in a mossy forest, golden hour, storybook style",
    # cool
    "a samurai silhouetted against a giant red moon, cherry blossoms mid-swirl in the wind, cinematic wide shot, painterly detail",
    # surreal
    "an enormous library where the bookshelves curve into a spiral staircase reaching into the clouds, warm dust-lit beams, highly detailed",

    # from PEZ paper:
    "teddy bear on skateboard, city street, shallow depth of field, photorealistic, cinematic lighting, low angle",

    # beautiful
    "a jellyfish drifting through a moonlit aurora sky, translucent glow, ethereal atmosphere",
    # funny
    "a grumpy cat wearing a tiny crown, sitting on a throne of yarn balls, regal lighting",
]

# SD generation
GEN_HEIGHT = 768
GEN_WIDTH = 768
NUM_INFERENCE_STEPS = 50

OUTPUT_DIR = Path("scripts/opt-bench/results/exp3_promrec")


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prompts = list(SAMPLED_PROMPTS) + EXTRA_PROMPTS
    print(f"Using {len(SAMPLED_PROMPTS)} pre-sampled {DATASET_NAME} prompts + {len(EXTRA_PROMPTS)} extras")
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p[:100]}")

    all_results = []

    n_configs = len(OPTIMIZER_TYPES) * len(TRIGGER_LENS) * len(SEEDS)

    for i, original_prompt in enumerate(prompts):
        prompt_dir = OUTPUT_DIR / f"prompt_{i:02d}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "original_prompt.txt").write_text(original_prompt, encoding="utf-8")

        for optimizer_type in OPTIMIZER_TYPES:
            for trigger_len in TRIGGER_LENS:
                for seed in SEEDS:
                    cfg_tag = f"{optimizer_type}_tl{trigger_len:02d}_seed{seed:02d}"
                    print(f"\n{'='*60}\n  Prompt {i+1}/{len(prompts)}  |  {cfg_tag}\n{'='*60}")
                    run_dir = prompt_dir / cfg_tag
                    run_dir.mkdir(parents=True, exist_ok=True)

                    run_name = f"recovery[p={i},opt={optimizer_type},tl={trigger_len},seed={seed}]"
                    tracker = WandbTracker(
                        run_name,
                        tags=["prompt-recovery", optimizer_type, CLIP_MODEL.split("/")[-1]],
                        project_name=WANDB_PROJECT,
                        entity=WANDB_ENTITY,
                        experiment_config={
                            "attack": optimizer_type,
                            "task": "prompt_recovery",
                            "sd_model": SD_MODEL,
                            "clip_model": CLIP_MODEL,
                            "prompt_idx": i,
                            "original_prompt": original_prompt,
                            "optimizer_type": optimizer_type,
                            "trigger_len": trigger_len,
                            "seed": seed,
                        },
                    )

                    quad = recover_prompt_end_to_end(
                        prompt=original_prompt,
                        sd_model_name=SD_MODEL,
                        clip_model_name=CLIP_MODEL,
                        optimizer_type=optimizer_type,
                        trigger_len=trigger_len,
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
                        "optimizer_type": optimizer_type,
                        "trigger_len": trigger_len,
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
        "n_sampled_prompts": len(SAMPLED_PROMPTS),
        "n_extra_prompts": len(EXTRA_PROMPTS),
        "sampled_prompts": list(SAMPLED_PROMPTS),
        "extra_prompts": EXTRA_PROMPTS,
        "optimizer_types": OPTIMIZER_TYPES,
        "trigger_lens": TRIGGER_LENS,
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
    print(f"All {len(prompts)}×{n_configs} runs complete.")


if __name__ == "__main__":
    run()
