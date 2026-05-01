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
from ty_extensions import Unknown
import ast
import os
from typing import List

import pandas as pd
import torch
import typer
import wandb
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from importlib import import_module

from scripts.attack_evaluate.evaluate_jailbreakness import evaluate_triggers
from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER

# `scripts.opt-bench` has a hyphen, so it can't be imported via `from ... import`;
# use string-based import to grab the variant→template_fn lookup.
get_variant_template_fn = import_module("scripts.opt-bench.exp2").get_variant_template_fn

# ─── Defaults ───────────────────────────────────────────────────────────────
WANDB_ENTITY = "matanbt"
DEFAULT_WANDB_PROJECT = "tropt-optbench"
DEFAULT_RUN_TYPE = "optbench_whitebox"

app = typer.Typer()


# ─── Helpers ────────────────────────────────────────────────────────────────
def _maybe_parse_list(val):
    """Parse a stringified Python list back to a list; pass through otherwise.

    `enhancebench_multi` runs log `optimized_instruction` / `optimized_target`
    as a Python list (one entry per template), which `pd.read_csv` reads back
    as the string repr `"['a', 'b', ...]"`. This recovers the list.
    """
    if isinstance(val, str) and val.startswith("[") and val.endswith("]"):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return val


_MULTI_RUN_TYPES = {"enhancebench_multi"}


def _explode_multi_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Explode multi-instruction rows into one row per (instruction, target).

    **CSV-II only.** Multi-instruction runs (`run_type` in `_MULTI_RUN_TYPES`)
    log `optimized_instruction` / `optimized_target` as Python lists; CSV-II
    needs per-template BLEU, so each multi-row is exploded into N rows that
    share the same universal trigger but differ in the training template.

    Do NOT use for CSV-III: there, each universal trigger should be evaluated
    once against the held-out universality dataset, not N times against the N
    training templates (the trigger is identical across the exploded rows).
    """
    df = df.copy()
    is_multi_row = df["run_type"].isin(_MULTI_RUN_TYPES)
    # Only parse the cells we actually intend to explode — avoids
    # mis-parsing single-instruction strings that happen to look list-like.
    df.loc[is_multi_row, "optimized_instruction"] = (
        df.loc[is_multi_row, "optimized_instruction"].map(_maybe_parse_list)
    )
    df.loc[is_multi_row, "optimized_target"] = (
        df.loc[is_multi_row, "optimized_target"].map(_maybe_parse_list)
    )

    is_multi = df["optimized_instruction"].map(lambda v: isinstance(v, list))
    if not is_multi.any():
        return df.reset_index(drop=True)
    # Sanity-check pairing before exploding.
    bad = df[is_multi].apply(
        lambda r: not isinstance(r["optimized_target"], list)
        or len(r["optimized_target"]) != len(r["optimized_instruction"]),
        axis=1,
    )
    if bad.any():
        raise ValueError(
            f"{bad.sum()} multi-row(s) have mismatched instruction/target list lengths"
        )
    df = df.explode(["optimized_instruction", "optimized_target"], ignore_index=True)
    return df


def _generate_with_hf(triggered_messages: List[str], model_name: str,
                      max_new_tokens: int, batch_size: int) -> List[str]:
    """Generate responses using a local HuggingFace pipeline."""
    from transformers import pipeline

    pipe = pipeline(
        "text-generation",
        model=model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    tok = pipe.tokenizer
    if tok.pad_token_id is None:
        eos = pipe.model.config.eos_token_id
        tok.pad_token_id = eos[0] if isinstance(eos, list) else eos
    tok.padding_side = "left"
    messages = [[{"role": "user", "content": t}] for t in triggered_messages]
    outputs = pipe(messages, max_new_tokens=max_new_tokens, max_length=None,
                   do_sample=False, batch_size=batch_size, return_full_text=False)
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
    model_name: str = typer.Option(None, help="Filter by config.model_name (optional)"),
    output_path: str = typer.Option("scripts/opt-bench/results/csv_i.csv"),
):
    """Pull essential run metadata and results from Wandb -> CSV I."""
    api = wandb.Api()
    filters = {"state": "finished", "config.run_type": run_type}
    if model_name:
        filters["config.model_name"] = model_name
    # filters["config.msg_id"] = 0  # Only include runs with msg_id=0 (initial optimization)
    runs = api.runs(f"{WANDB_ENTITY}/{wandb_project}", filters=filters)
    rows = []
    for run in runs:
        c, s = run.config, run.summary
        rows.append({
            "run_id": run.id,
            "run_name": run.name,
            "run_type": c.get("run_type"),
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
            "total_tokens": s.get("total_models_stats/total_tokens"),
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
    max_new_tokens: int = typer.Option(256),
    batch_size: int = typer.Option(8),
    use_litellm: bool = typer.Option(False, help="Use LiteLLM for generation (API models)"),
):
    """Add triggered-message generation and BLEU score to CSV I -> CSV II."""
    df = pd.read_csv(csv_i_path)
    df = _explode_multi_rows(df)

    # Drop rows missing the fields we need; otherwise `str(NaN) == "nan"` and
    # the placeholder substitution silently produces a malformed prompt.
    required = ["optimized_instruction", "optimized_target", "best_trigger_str"]
    n_before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  dropped {n_before - len(df)} row(s) with missing required fields")

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
        # Truncate generated to the same number of *words* as target — char
        # truncation can cut a word in half and produce noisy BLEU.
        # target_words = str(row["optimized_target"]).split()
        # generated_words = str(row["generated_response"]).split()[: len(target_words)]
        # return sentence_bleu(
        #     [target_words], generated_words, smoothing_function=smooth
        # )
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
    batch_size: int = typer.Option(8),
    use_litellm: bool = typer.Option(False, help="Use LiteLLM for generation (API models)"),
    judge_openai_model: str = typer.Option(
        None,
        help="If set (e.g. 'openai/gpt-4o-mini'), use strongreject_rubric with this "
             "OpenAI judge instead of strongreject_finetuned. Requires OPENAI_API_KEY.",
    ),
):
    """Run full universality evaluation on all triggers -> CSV III.

    Each trigger is evaluated under the same prompt-wrapping `template_fn`
    the variant used during training (e.g. PRS for `gcg_prs_template`).
    Variants without a `template_fn` fall through identity wrapping.

    NOTE: deliberately consumes raw CSV-I (one row per run, including multi
    runs as a single row with a universal trigger) — does NOT call
    `_explode_multi_rows`. Each universal trigger is evaluated once on the
    held-out universality dataset; exploding here would generate the same
    trigger × dataset N times redundantly.
    """
    df = pd.read_csv(csv_i_path)
    df["trigger_id"] = range(len(df))
    df["best_trigger_str"] = df["best_trigger_str"].fillna("").astype(str)

    eval_kwargs = {}
    if judge_openai_model:
        eval_kwargs["evaluators"] = ["strongreject_rubric"]
        eval_kwargs["judge_models"] = [judge_openai_model]

    template_fns = [get_variant_template_fn(v) for v in df["variant_name"].tolist()]

    eval_df = evaluate_triggers(
        model_name=model_name,
        trigger_strs=df["best_trigger_str"].tolist(),
        trigger_ids=df["trigger_id"].tolist(),
        harmful_dataset="clearharm[:100]",
        batch_size=batch_size,
        model_backend="litellm" if use_litellm else "hf_pipeline",
        template_fns=template_fns,
        **eval_kwargs,
    )
    eval_df["eval_model"] = model_name

    meta_cols = ["trigger_id", "model_name", "optimizer_name", "variant_name",
                 "loss_name", "is_soft", "msg_id", "seed"]
    final_df = pd.merge(eval_df, df[meta_cols], on="trigger_id", how="left")
    final_df.to_csv(output_path, index=False)
    print(f"CSV III -> {output_path}  ({len(final_df)} rows)")


if __name__ == "__main__":
    app()
