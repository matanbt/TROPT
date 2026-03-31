
@app.command()
def build_csv_i(
    output_path: str = typer.Option("scripts/opt-bench/results/csv_i.csv"),
):
    """Pull essential run metadata and results from Wandb → CSV I."""
    api = wandb.Api()
    runs = api.runs(
        f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"state": "finished", "config.run_type": _RUN_TYPE},
    )

    rows = []
    for run in runs:
        c, s = run.config, run.summary
        rows.append({
            "run_id": run.id,
            "run_name": run.name,
            "model_name": c.get("model_name"),
            "optimizer_name": c.get("optimizer_name"),
            "loss_name": c.get("loss_name"),
            "is_soft": c.get("is_soft", False),
            "msg_id": c.get("msg_id"),
            "seed": c.get("seed"),
            "optimized_instruction": c.get("optimized_instruction"),
            "optimized_target": c.get("optimized_target"),
            "best_loss": s.get("best_loss"),
            "best_trigger_str": s.get("best_trigger_str"),
            "total_flops": s.get("final/total_flops"),
            "total_tokens": s.get("final/total_tokens"),
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    csv_i = pd.DataFrame(rows)
    csv_i.to_csv(output_path, index=False)
    print(f"CSV I → {output_path}  ({len(csv_i)} rows)")


@app.command()
def build_csv_ii(
    csv_i_path: str = typer.Option("scripts/opt-bench/results/csv_i.csv"),
    model_name: str = typer.Option("google/gemma-2-2b-it", help="Model used for generation"),
    output_path: str = typer.Option("scripts/opt-bench/results/csv_ii.csv"),
    max_new_tokens: int = typer.Option(128),
    batch_size: int = typer.Option(32),
):
    """Add triggered-message generation and BLEU score to CSV I → CSV II."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    df = pd.read_csv(csv_i_path)

    df["triggered_message"] = df.apply(
        lambda r: str(r["optimized_instruction"]).replace(
            OPTIMIZED_TRIGGER_PLACEHOLDER, str(r["best_trigger_str"])
        ),
        axis=1,
    )

    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    messages = [[{"role": "user", "content": t}] for t in df["triggered_message"]]
    outputs = pipe(messages, max_new_tokens=max_new_tokens, do_sample=False,
                   batch_size=batch_size, return_full_text=False)
    df["generated_response"] = [out[0]["generated_text"] for out in outputs]

    smooth = SmoothingFunction().method1

    def _bleu(row) -> float:
        target = str(row["optimized_target"])
        generated = str(row["generated_response"])[:len(target)]
        return sentence_bleu([target.split()], generated.split(), smoothing_function=smooth)

    df["bleu"] = df.apply(_bleu, axis=1)
    df["eval_model"] = model_name

    df.to_csv(output_path, index=False)
    print(f"CSV II → {output_path}  ({len(df)} rows)")


@app.command()
def build_csv_iii(
    csv_i_path: str = typer.Option("scripts/opt-bench/results/csv_i.csv"),
    model_name: str = typer.Option("google/gemma-2-2b-it", help="Model used for generation"),
    output_path: str = typer.Option("scripts/opt-bench/results/csv_iii.csv"),
    batch_size: int = typer.Option(32),
):
    """Run full universality evaluation on all triggers → CSV III.

    Evaluates each trigger across all ClearHarm messages via strongreject_finetuned.
    """
    df = pd.read_csv(csv_i_path)
    df["trigger_id"] = range(len(df))

    eval_df = evaluate_triggers(
        model_name=model_name,
        trigger_strs=df["best_trigger_str"].tolist(),
        trigger_ids=df["trigger_id"].tolist(),
        harmful_dataset="clearharm",
        batch_size=batch_size,
    )
    eval_df["eval_model"] = model_name

    meta_cols = ["trigger_id", "model_name", "optimizer_name", "loss_name",
                 "is_soft", "msg_id", "seed"]
    final_df = pd.merge(eval_df, df[meta_cols], on="trigger_id", how="left")
    final_df.to_csv(output_path, index=False)
    print(f"CSV III → {output_path}  ({len(final_df)} rows)")
