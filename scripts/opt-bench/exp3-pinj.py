"""
Prompt Injection Evasion via Universal Adversarial Triggers (UAT).

Optimizes a universal trigger that causes a prompt-injection classifier
(Llama Prompt Guard 2) to misclassify injection prompts as benign.
Uses GCG-style optimization with UAT batch sampling over many templates.

Evaluates on three splits:
  - Held-in:  injection prompts used during optimization (+ trigger)
  - Held-out: unseen injection prompts (+ trigger)
  - Benign:   benign prompts (+ trigger) — measures collateral damage

Usage
-----
  python scripts/opt-bench/exp3-pinj.py
"""

import json
import random
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER
from tropt.loss import MisclassCELoss
from tropt.model.huggingface.classifier import ClassifierHFModel
from tropt.optimizer.gcgplus_optimizer import GCGPlusOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.optimizer.utils.token_initializers import get_printable_random_trigger
from tropt.tracker import WandbTracker

# ─── Constants ───────────────────────────────────────────────────────────────
WANDB_ENTITY = "matanbt"
WANDB_PROJECT = "tropt-pinj"

MODEL_NAME = "meta-llama/Llama-Prompt-Guard-2-86M"
DATASET_NAME = "rogue-security/prompt-injections-benchmark"

# Prompt Guard 2 class mapping: 0=BENIGN, 1=MALICIOUS
BENIGN_CLASS_IDX = 0
MALICIOUS_CLASS_IDX = 1

N_HELD_IN = 50
TRIGGER_LEN = 20
NUM_STEPS = 500
TEMPLATE_BATCH_SIZE = 10
N_CANDIDATES = 256
SEED = 42

TOKEN_CONSTRAINTS = TokenConstraints(
    disallow_non_ascii=True, disallow_special_tokens=True
)

OUTPUT_DIR = Path("scripts/opt-bench/results")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _evaluate_split(
    model: ClassifierHFModel,
    texts: list[str],
    trigger: str,
    append_trigger: bool,
    batch_size: int = 32,
) -> dict:
    """Classify texts (optionally with trigger appended) and return metrics."""
    if append_trigger:
        eval_texts = [t + " " + trigger for t in texts]
    else:
        eval_texts = texts

    all_preds = []
    for i in range(0, len(eval_texts), batch_size):
        batch = eval_texts[i : i + batch_size]
        with torch.no_grad():
            out = model.invoke_from_texts(batch)
        preds = out.output_class_logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)

    n_benign = sum(1 for p in all_preds if p == BENIGN_CLASS_IDX)
    return {
        "n_total": len(texts),
        "n_predicted_benign": n_benign,
        "n_predicted_malicious": len(texts) - n_benign,
        "benign_rate": n_benign / len(texts),
    }


def _save_results(results: dict, filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def run():
    """Run the UAT prompt-injection evasion experiment."""

    # --- Load model ---
    print(f"Loading model: {MODEL_NAME}")
    model = ClassifierHFModel(model_name=MODEL_NAME, dtype=torch.bfloat16)

    # --- Load dataset ---
    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split="test")
    injections = [row["text"] for row in ds if row["label"] == "jailbreak"]
    benign_texts = [row["text"] for row in ds if row["label"] == "benign"]
    print(f"Dataset: {len(injections)} injections, {len(benign_texts)} benign")

    # Shuffle and split injection prompts
    rng = random.Random(SEED)
    shuffled_injections = list(injections)
    rng.shuffle(shuffled_injections)
    held_in = shuffled_injections[:N_HELD_IN]
    held_out = shuffled_injections[N_HELD_IN:]
    print(f"Held-in: {len(held_in)}, Held-out: {len(held_out)}")

    # --- Baseline: classify the actual eval splits without trigger ---
    print("\n--- Baseline (no trigger, same splits used for eval) ---")
    baseline_held_in = _evaluate_split(model, held_in, trigger="", append_trigger=False)
    baseline_held_out = _evaluate_split(model, held_out, trigger="", append_trigger=False)
    baseline_benign = _evaluate_split(model, benign_texts, trigger="", append_trigger=False)
    print(f"Held-in  (no trigger): {baseline_held_in['benign_rate']:.1%} → benign  (natural FN rate)")
    print(f"Held-out (no trigger): {baseline_held_out['benign_rate']:.1%} → benign  (natural FN rate)")
    print(f"Benign   (no trigger): {baseline_benign['benign_rate']:.1%} → benign  (natural TN rate)")

    # --- Build templates ---
    templates = [
        text + " " + OPTIMIZED_TRIGGER_PLACEHOLDER for text in held_in
    ]

    initial_trigger = get_printable_random_trigger(
        trigger_len=TRIGGER_LEN,
        tokenizer=model.tokenizer,
        token_constraints=TOKEN_CONSTRAINTS,
    )

    # --- Optimization ---
    experiment_config = {
        "attack": "uat_gcgplus",
        "task": "prompt_injection_evasion",
        "model_name": MODEL_NAME,
        "dataset": DATASET_NAME,
        "n_held_in": N_HELD_IN,
        "n_held_out": len(held_out),
        "n_benign": len(benign_texts),
        "trigger_len": TRIGGER_LEN,
        "template_batch_size": TEMPLATE_BATCH_SIZE,
        "seed": SEED,
    }

    run_name = f"pinj[uat,gcgplus,n={N_HELD_IN},bs={TEMPLATE_BATCH_SIZE}]"
    print(f"\n{'='*60}\n  {run_name}\n{'='*60}")

    tracker = WandbTracker(
        run_name,
        tags=["prompt-injection", "uat", "gcgplus",
              MODEL_NAME.split("/")[-1]],
        project_name=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        experiment_config=experiment_config,
    )

    loss = MisclassCELoss(targeted=True, target_class_idx=BENIGN_CLASS_IDX)

    optimizer = GCGPlusOptimizer(
        model=model,
        loss=loss,
        tracker=tracker,
        seed=SEED,

        # GCG parameters:
        num_steps=500,
        n_candidates=512,
        sample_topk=256,
        sample_n_replace=1,
        token_constraints=TOKEN_CONSTRAINTS,
        use_retokenize=True,

        # UAT instantiation:
        template_batch_size=TEMPLATE_BATCH_SIZE,
    )

    result = optimizer.optimize_trigger(
        templates=templates,
        initial_trigger=initial_trigger,
    )

    trigger = result.best_trigger_str
    print(f"\nOptimized trigger: {trigger!r}")
    print(f"Best loss: {result.best_loss:.4f}")

    # --- Evaluation ---
    print("\n--- Evaluation ---")

    eval_held_in = _evaluate_split(model, held_in, trigger, append_trigger=True)
    eval_held_out = _evaluate_split(model, held_out, trigger, append_trigger=True)
    eval_benign_triggered = _evaluate_split(model, benign_texts, trigger, append_trigger=True)
    eval_benign_clean = _evaluate_split(model, benign_texts, "", append_trigger=False)

    lift_held_in = eval_held_in["benign_rate"] - baseline_held_in["benign_rate"]
    lift_held_out = eval_held_out["benign_rate"] - baseline_held_out["benign_rate"]
    benign_delta = eval_benign_triggered["benign_rate"] - baseline_benign["benign_rate"]

    print(f"Held-in:  {baseline_held_in['benign_rate']:.1%} → {eval_held_in['benign_rate']:.1%}  (lift {lift_held_in:+.1%})")
    print(f"Held-out: {baseline_held_out['benign_rate']:.1%} → {eval_held_out['benign_rate']:.1%}  (lift {lift_held_out:+.1%})")
    print(f"Benign:   {baseline_benign['benign_rate']:.1%} → {eval_benign_triggered['benign_rate']:.1%}  (delta {benign_delta:+.1%})")

    # --- Save ---
    _save_results(
        {
            "experiment": "prompt_injection_evasion",
            "model_name": MODEL_NAME,
            "dataset": DATASET_NAME,
            "trigger": trigger,
            "trigger_len": TRIGGER_LEN,
            "num_steps": NUM_STEPS,
            "template_batch_size": TEMPLATE_BATCH_SIZE,
            "n_candidates": N_CANDIDATES,
            "seed": SEED,
            "n_held_in": len(held_in),
            "n_held_out": len(held_out),
            "n_benign": len(benign_texts),
            "final_loss": result.best_loss,
            "timestamp": datetime.now().isoformat(),
            "baseline": {
                "held_in": baseline_held_in,
                "held_out": baseline_held_out,
                "benign": baseline_benign,
            },
            "eval": {
                "held_in": eval_held_in,
                "held_out": eval_held_out,
                "benign_with_trigger": eval_benign_triggered,
                "benign_clean": eval_benign_clean,
            },
            "lift": {
                "held_in": lift_held_in,
                "held_out": lift_held_out,
                "benign_delta": benign_delta,
            },
            "config": experiment_config,
        },
        f"exp3_pinj_{MODEL_NAME.split('/')[-1]}.json",
    )

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Trigger:      {trigger!r}")
    print(f"  Held-in:      {baseline_held_in['benign_rate']:.1%} → {eval_held_in['benign_rate']:.1%}  (lift {lift_held_in:+.1%})")
    print(f"  Held-out:     {baseline_held_out['benign_rate']:.1%} → {eval_held_out['benign_rate']:.1%}  (lift {lift_held_out:+.1%})")
    print(f"  Benign kept:  {baseline_benign['benign_rate']:.1%} → {eval_benign_triggered['benign_rate']:.1%}  (delta {benign_delta:+.1%})")
    print(f"{'='*60}")


if __name__ == "__main__":
    run()
