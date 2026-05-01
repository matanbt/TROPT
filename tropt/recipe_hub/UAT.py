"""Universal Adversarial Triggers (UAT) for classifier evasion.

Adapts the UAT framework (Wallace et al., 2019) using GCG-style optimization
instead of the original HotFlip variant. Optimizes a single universal trigger
that causes misclassification across many inputs via batch sampling.

Reference: https://arxiv.org/abs/1908.07125
"""

from typing import List, Optional

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER, Targets
from tropt.loss import MisclassCELoss
from tropt.model.huggingface.classifier import ClassifierHFModel
from tropt.optimizer import OptimizerResult
from tropt.optimizer.gcgplus_optimizer import GCGPlusOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.optimizer.utils.token_initializers import get_printable_random_trigger
from tropt.tracker import BaseTracker

_UAT_TOKEN_CONSTRAINTS = TokenConstraints(
    disallow_non_ascii=True, disallow_special_tokens=True
)


def uat_classifier(
    templates: List[str],
    target_class_idx: int,
    model_obj: ClassifierHFModel,
    tracker: Optional[BaseTracker] = None,
    seed: int = 42,
    trigger_len: int = 5,
    num_steps: int = 500,
    template_batch_size: int = 5,
    n_candidates: int = 256,
) -> OptimizerResult:
    """UAT on a classifier model using GCG optimization.

    Args:
        templates: Input texts with ``{{OPTIMIZED_TRIGGER}}`` placeholder.
        target_class_idx: Class index to steer predictions toward (targeted attack).
        model_obj: Pre-loaded ClassifierHFModel.
        trigger_len: Number of trigger tokens to optimize.
        template_batch_size: Templates sampled per optimization step (UAT-style).
    """
    loss = MisclassCELoss(targeted=True)
    initial_trigger = get_printable_random_trigger(
        trigger_len=trigger_len,
        tokenizer=model_obj.tokenizer,
        token_constraints=_UAT_TOKEN_CONSTRAINTS,
    )

    optimizer = GCGPlusOptimizer(
        model=model_obj,
        loss=loss,
        tracker=tracker,
        seed=seed,
        num_steps=num_steps,
        n_candidates=n_candidates,
        sample_topk=256,
        sample_n_replace=1,
        token_constraints=_UAT_TOKEN_CONSTRAINTS,
        use_retokenize=False,
        template_batch_size=template_batch_size,
    )

    return optimizer.optimize_trigger(
        templates=templates,
        targets=Targets(target_class_idx=[target_class_idx] * len(templates)),
        initial_trigger=initial_trigger,
    )


def uat_prompt_injection(
    model_name: str = "meta-llama/Llama-Prompt-Guard-2-86M",
    n_samples: int = 50,
    trigger_len: int = 5,
    num_steps: int = 500,
    template_batch_size: int = 5,
    tracker: Optional[BaseTracker] = None,
    seed: int = 42,
) -> dict:
    """UAT for prompt-injection evasion on Llama Prompt Guard 2.

    Loads the classifier and the rogue-security/prompt-injections-benchmark
    dataset, optimizes a universal trigger to make injection prompts be
    classified as BENIGN, then evaluates on held-in, held-out, and benign splits.

    Returns a dict with the optimization result and evaluation metrics.
    """
    import random as stdlib_random

    import torch
    from datasets import load_dataset

    # --- Load model ---
    model = ClassifierHFModel(model_name=model_name)
    # Prompt Guard 2: 0=BENIGN, 1=MALICIOUS
    benign_class_idx = 0

    # --- Load dataset ---
    ds = load_dataset("rogue-security/prompt-injections-benchmark", split="test")
    injections = [row["text"] for row in ds if row["label"] == "jailbreak"]
    benign_texts = [row["text"] for row in ds if row["label"] == "benign"]

    # Shuffle and sample injection prompts
    rng = stdlib_random.Random(seed)
    rng.shuffle(injections)
    held_in = injections[:n_samples]
    held_out = injections[n_samples:]

    print(f"Dataset: {len(injections)} injections, {len(benign_texts)} benign")
    print(f"Held-in: {len(held_in)}, Held-out: {len(held_out)}")

    # Build templates: append trigger placeholder to each injection prompt
    templates = [
        text + " " + OPTIMIZED_TRIGGER_PLACEHOLDER for text in held_in
    ]

    # --- Optimize ---
    result = uat_classifier(
        templates=templates,
        target_class_idx=benign_class_idx,
        model_obj=model,
        tracker=tracker,
        seed=seed,
        trigger_len=trigger_len,
        num_steps=num_steps,
        template_batch_size=template_batch_size,
    )

    assert result.best_trigger_str is not None
    trigger: str = result.best_trigger_str
    print(f"\nOptimized trigger: {trigger!r}")
    print(f"Best loss: {result.best_loss:.4f}")

    # --- Evaluate ---
    def evaluate_split(texts: list[str], append_trigger: bool) -> dict:
        if append_trigger:
            eval_texts = [t + " " + trigger for t in texts]
        else:
            eval_texts = texts
        with torch.no_grad():
            out = model.invoke_from_texts(eval_texts)
        preds = out.output_class_logits.argmax(dim=-1).cpu().tolist()
        n_benign = sum(1 for p in preds if p == benign_class_idx)
        return {
            "n_total": len(texts),
            "n_predicted_benign": n_benign,
            "attack_success_rate": n_benign / len(texts),
        }

    eval_held_in = evaluate_split(held_in, append_trigger=True)
    eval_held_out = evaluate_split(held_out, append_trigger=True)
    eval_benign = evaluate_split(benign_texts, append_trigger=True)
    eval_benign_clean = evaluate_split(benign_texts, append_trigger=False)

    print("\n--- Evaluation ---")
    print(f"Held-in  (injections + trigger): {eval_held_in['attack_success_rate']:.1%} classified as benign")
    print(f"Held-out (injections + trigger): {eval_held_out['attack_success_rate']:.1%} classified as benign")
    print(f"Benign   (+ trigger):            {eval_benign['attack_success_rate']:.1%} classified as benign")
    print(f"Benign   (clean, no trigger):    {eval_benign_clean['attack_success_rate']:.1%} classified as benign")

    return {
        "result": result,
        "trigger": trigger,
        "eval_held_in": eval_held_in,
        "eval_held_out": eval_held_out,
        "eval_benign_with_trigger": eval_benign,
        "eval_benign_clean": eval_benign_clean,
    }
