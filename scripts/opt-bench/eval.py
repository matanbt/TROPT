"""
Evaluation pipeline for optimizer benchmarks.

Builds three CSVs from wandb run data:
  I.   Run metadata and best results (model-agnostic)
  II.  + triggered generation and BLEU score
  III. + universality evaluation via strongreject

Usage
-----
  python -m scripts.opt-bench.eval build-csv-i --wandb-project tropt-optbench
  python -m scripts.opt-bench.eval build-csv-ii --model-name google/gemma-2-2b-it
  python -m scripts.opt-bench.eval build-csv-ii --model-name openai/gpt-5-nano --use-litellm
  python -m scripts.opt-bench.eval build-csv-iii --model-name google/gemma-2-2b-it
"""
import os
from typing import List

import pandas as pd
import torch
import typer
import wandb
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from scripts.attack_evaluate.evaluate_jailbreakness import evaluate_triggers
from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER

# ─── Defaults ───────────────────────────────────────────────────────────────
WANDB_ENTITY = "matanbt"
DEFAULT_WANDB_PROJECT = "tropt-optbench"
DEFAULT_RUN_TYPE = "optbench_whitebox"

app = typer.Typer()


# ─── Helpers ────────────────────────────────────────────────────────────────
def _generate_with_hf(triggered_messages: List[str], model_name: str,
                      max_new_tokens: int, batch_size: int) -> List[str]:
    """Generate responses using a local HuggingFace pipeline."""
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    messages = [[{"role": "user", "content": t}] for t in triggered_messages]
    outputs = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False,
                   batch_size=batch_size, return_full_text=False)
    return [out[0]["generated_text"] for out in outputs]


def _generate_with_litellm(triggered_messages: List[str], model_name: str,
                           max_new_tokens: int) -> List[str]:
    """Generate responses via LiteLLM (for API-based models like OpenAI)."""
    from tropt.model.litellm_proxy.lm import LiteLLMModel

    model = LiteLLMModel(model_name=model_name)
    output = model.invoke_from_texts(
        input_texts=triggered_messages,
        require_generation=True,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    return output.generated_response_strs


# ─── CSV I: Wandb metadata ─────────────────────────────────────────────────
@app.command()
def build_csv_i(
    wandb_project: str = typer.Option(DEFAULT_WANDB_PROJECT, help="Wandb project name"),
    run_type: str = typer.Option(DEFAULT_RUN_TYPE, help="Filter by run_type config value"),
    output_path: str = typer.Option("scripts/opt-bench/results/csv_i.csv"),
):
    """Pull essential run metadata and results from Wandb -> CSV I."""
    api = wandb.Api()
    runs = api.runs(
        f"{WANDB_ENTITY}/{wandb_project}",
        filters={"state": "finished", "config.run_type": run_type},
    )

    rows = []
    for run in runs:
        c, s = run.config, run.summary
        rows.append({
            "run_id": run.id,
            "run_name": run.name,
            "model_name": c.get("model_name"),
            "optimizer_name": c.get("optimizer_name"),
            "variant_name": c.get("variant_name"),
            "loss_name": c.get("loss_name"),
            "is_soft": c.get("is_soft", False),
            "msg_id": c.get("msg_id"),
            "seed": c.get("seed"),
            "optimized_instruction": c.get("optimized_instruction"),
            "optimized_target": c.get("optimized_target"),
            "best_loss": s.get("best_loss"),
            "best_trigger_str": s.get("best_trigger_str"),
            "total_flops": s.get("total_models_stats/total_flops"),
            "total_tokens": s.get("total_models_stats/total_input_tokens"),
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    csv_i = pd.DataFrame(rows)
    csv_i.to_csv(output_path, index=False)
    print(f"CSV I -> {output_path}  ({len(csv_i)} rows)")


# ─── CSV II: + generation and BLEU ─────────────────────────────────────────
@app.command()
def build_csv_ii(
    csv_i_path: str = typer.Option("scripts/opt-bench/results/csv_i.csv"),
    model_name: str = typer.Option("google/gemma-2-2b-it", help="Model for generation"),
    output_path: str = typer.Option("scripts/opt-bench/results/csv_ii.csv"),
    max_new_tokens: int = typer.Option(128),
    batch_size: int = typer.Option(32),
    use_litellm: bool = typer.Option(False, help="Use LiteLLM for generation (API models)"),
):
    """Add triggered-message generation and BLEU score to CSV I -> CSV II."""
    df = pd.read_csv(csv_i_path)

    df["triggered_message"] = df.apply(
        lambda r: str(r["optimized_instruction"]).replace(
            OPTIMIZED_TRIGGER_PLACEHOLDER, str(r["best_trigger_str"])
        ),
        axis=1,
    )

    triggered = df["triggered_message"].tolist()
    if use_litellm:
        df["generated_response"] = _generate_with_litellm(
            triggered, model_name, max_new_tokens,
        )
    else:
        df["generated_response"] = _generate_with_hf(
            triggered, model_name, max_new_tokens, batch_size,
        )

    smooth = SmoothingFunction().method1

    def _bleu(row) -> float:
        target = str(row["optimized_target"])
        generated = str(row["generated_response"])[:len(target)]
        return sentence_bleu([target.split()], generated.split(), smoothing_function=smooth)

    df["bleu"] = df.apply(_bleu, axis=1)
    df["eval_model"] = model_name

    df.to_csv(output_path, index=False)
    print(f"CSV II -> {output_path}  ({len(df)} rows)")


# ─── CSV III: universality evaluation ───────────────────────────────────────
@app.command()
def build_csv_iii(
    csv_i_path: str = typer.Option("scripts/opt-bench/results/csv_i.csv"),
    model_name: str = typer.Option("google/gemma-2-2b-it", help="Model for generation"),
    output_path: str = typer.Option("scripts/opt-bench/results/csv_iii.csv"),
    batch_size: int = typer.Option(32),
    use_litellm: bool = typer.Option(False, help="Use LiteLLM for generation (API models)"),
):
    """Run full universality evaluation on all triggers -> CSV III."""
    df = pd.read_csv(csv_i_path)
    df["trigger_id"] = range(len(df))

    eval_df = evaluate_triggers(
        model_name=model_name,
        trigger_strs=df["best_trigger_str"].tolist(),
        trigger_ids=df["trigger_id"].tolist(),
        harmful_dataset="clearharm[:50]",
        batch_size=batch_size,
        model_backend="litellm" if use_litellm else "hf_pipeline",
    )
    eval_df["eval_model"] = model_name

    meta_cols = ["trigger_id", "model_name", "optimizer_name", "variant_name",
                 "loss_name", "is_soft", "msg_id", "seed"]
    final_df = pd.merge(eval_df, df[meta_cols], on="trigger_id", how="left")
    final_df.to_csv(output_path, index=False)
    print(f"CSV III -> {output_path}  ({len(final_df)} rows)")


if __name__ == "__main__":
    app()
