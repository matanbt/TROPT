"""
Run GCG on specific advbench samples, using two different implementations.
"""
import math
import os
import pandas as pd
import wandb
from scripts.attack_evaluate.evaluate_jailbreakness import evaluate_triggers
from scripts.reprod_experiment.nanogcg import GCGConfig, GCG
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import typer
from typing import List
from tropt.loss.base import PrefillCELoss
from tropt.models.huggingface.lm import LMHFModel
from tropt.optimizer.base import OptimizerResult
from tropt.optimizer.gcg_optimizer import GCGOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker.base import WandbTracker
from tropt.common import Targets

import numpy as np
import matplotlib.pyplot as plt

# Default list from original script
DEFAULT_INDICES = [ 
    225, 
    # 89, 155, 78, 283, 162, 236, 389, 568, 695,
    # 128, 342, 456, 509, 634, 721, 
]
INITIAL_TRIGGER = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
RANDOM_SEED = 42
WANDB_ENTITY = "matanbt"
WANDB_PROJECT = "reprod_gcg"
DATASET_PATH = "scripts/reprod_experiment/data/advbench_plus.csv"
app = typer.Typer()

# TODO these runs should be:
# {google/gemma-2-2b-it} x {3 seeds} x {15 samples}

@app.command()
def nanogcg(
    model_name: str = typer.Option("google/gemma-2-2b-it", help="HuggingFace model identifier"),
    sample_indices: List[int] = typer.Option(
        DEFAULT_INDICES, 
        help="Specific indices to run. Usage: --sample-indices 225 --sample-indices 89"
    ),
    seed: int = typer.Option(RANDOM_SEED, help="Random seed for reproducibility"),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CSV
    df = pd.read_csv(DATASET_PATH)
    df['message_id'] = range(len(df))

    # Load model and tokenizer
    print(f"Loading model {model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype='bfloat16',
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Loaded model {model_name} with dtype {model.dtype}.")

    for message_id in sample_indices:
        # Filter dataframe for the specific index
        row = df.loc[df.message_id == message_id].iloc[0]
        
        print(f"running: advbench[{model_name.split('/')[-1]}, {message_id}]")

        # Load data safely accessing the scalar values
        messages = row['message_template'].replace('{{OPTIMIZED_TRIGGER}}', '{optim_str}')
        target = row['target_response_prefix']
        run_name = f'nanogcg[{model_name.split("/")[-1]},m={message_id}]'
        if seed != RANDOM_SEED:
            run_name += f'_s={seed}'
        wandb_metadata = dict(
            optimized_message_id=message_id,
            name=run_name,
            model_name=model_name,
            implementation='nanogcg',
            method='gcg',
            random_seed=seed,
        )

        # Set GCG parameters
        config = GCGConfig(
            optim_str_init=INITIAL_TRIGGER,
            seed=seed,
            num_steps=500,
            search_width=512,
            topk=256,
            n_replace=1,
            use_prefix_cache=False,
            wandb_log=True
        )
        
        # Initialize wandb
        wandb.init(
            name=wandb_metadata.get('name', "nanogcg"),
            tags=['nanogcg', 'gcg'],
            project=WANDB_PROJECT, 
            entity=WANDB_ENTITY,
            config={**wandb_metadata},
        )

        # Run GCG
        gcg = GCG(model, tokenizer, config)
        result = gcg.run(messages, target, wandb_metadata)

        wandb.finish()


@app.command()
def tropt(
    model_name: str = typer.Option("google/gemma-2-2b-it", help="HuggingFace model identifier"),
    sample_indices: List[int] = typer.Option(
        DEFAULT_INDICES, 
        help="Specific indices to run. Usage: --sample-indices 225 --sample-indices 89"
    ),
    method: str = typer.Option("gcg", help="Which optimier to use"),
    seed: int = typer.Option(RANDOM_SEED, help="Random seed for reproducibility"),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load CSV
    df = pd.read_csv(DATASET_PATH)
    df['message_id'] = range(len(df))

    # Load model and tokenizer
    model = LMHFModel(
        model_name=model_name,
        device=device,
        forward_pass_batch_size=1024,
        use_prefix_cache=False,
        dtype='bfloat16',
    )
    loss = PrefillCELoss()

    for message_id in sample_indices:

       # Filter dataframe for the specific index
        row = df.loc[df.message_id == message_id].iloc[0]
        
        print(f"running: advbench[{model_name.split('/')[-1]}, {message_id}]")

        # Load data safely accessing the scalar values
        messages = row['message_template']
        target = row['target_response_prefix']
        run_name = f'tropt[{method},{model_name.split("/")[-1]},m={message_id}]'
        if seed != RANDOM_SEED:
            run_name += f'_s={seed}'
        wandb_metadata = dict(
            optimized_message_id=message_id,
            name=run_name,
            model_name=model_name,
            implementation='tropt',
            method=method,
            random_seed=seed,
        )

        # Run GCG
        tracker = WandbTracker(
            wandb_metadata.get('name', "tropt"),
            tags=['tropt', method],
            project_name=WANDB_PROJECT, 
            entity=WANDB_ENTITY,
            config_dump=wandb_metadata
        )
        if method == 'gcg':
            optimizer = GCGOptimizer(
                model=model,
                loss=loss,
                seed=seed,
                tracker=tracker,
                # Set parameters from the paper:
                num_steps=500,
                n_candidates=512,
                sample_topk=256,
                sample_n_replace=1,
                token_constraints=TokenConstraints(
                    disallow_non_ascii=True, disallow_special_tokens=True
                ),
                use_retokenize=True,
            )
        elif method == 'gslt':
            # TODO 
            from tropt.optimizer.gaslite_optimizer import GASLITEOptimizer
            optimizer = GASLITEOptimizer(
                model=model,
                loss=loss,
                seed=seed,
                tracker=tracker,
                
                num_steps=100,
                n_grad=10,
                n_flip=7,  # ~0.3 of 20
                n_candidates=256,
                token_constraints=TokenConstraints(
                    disallow_non_ascii=True, disallow_special_tokens=True
                ),
                use_retokenize=True,
            )
        elif method == 'gslt+':
            from tropt.optimizer.gasliteplus_optimizer import GASLITEPlusOptimizer
            optimizer = GASLITEPlusOptimizer(
                model=model,
                loss=loss,
                seed=seed,
                tracker=tracker,
                
                num_steps=100,
                n_grad=10,
                n_flip=7,  # ~0.3 of 20
                n_bulk_flips=7,  # similar to GASLITE (1 flip at a time)
                buffer_size=10,
                n_candidates=256,
                flip_pos_method="ordered",
                token_constraints=TokenConstraints(
                    disallow_non_ascii=True, disallow_special_tokens=True
                ),
                use_retokenize=True,
            )
        elif method == 'qgslt+2':
            from tropt.optimizer.gasliteplus_optimizer import GASLITEPlusOptimizer
            optimizer = GASLITEPlusOptimizer(
                model=model,
                loss=loss,
                seed=seed,
                tracker=tracker,
                
                num_steps=100,
                n_grad=5,
                n_flip=7,  # ~0.3 of 20
                n_candidates=128,
                buffer_size=10,
                decline_n_flip_from_step=0.5,
                early_stopping_patience=100,  # effectively disabled
                n_bulk_flips=10,
                flip_pos_method="ordered",
                token_constraints=TokenConstraints(
                    disallow_non_ascii=True, disallow_special_tokens=True
                ),
                use_retokenize=True,
            )

        result = optimizer.optimize_trigger(
            templates=[messages],
            targets=Targets(target_response_strs=[target]),
            initial_trigger=INITIAL_TRIGGER,
        )
        tracker.finish()

    
@app.command()
def eval_jailbreak_results(
    model_name: str = typer.Option("google/gemma-2-2b-it", help="HuggingFace model identifier"),
    output_path: str = typer.Option("evaluation_results.csv", help="Where to save the results"),
):

    api = wandb.Api()
    runs = api.runs(
        path=f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"state": "finished"}
    )

    metadata_df = []
    
    for run in runs:
        # Extract the optimized trigger. 
        trigger = run.summary['best_trigger_str']
        metadata_df.append(dict(
            trigger=trigger,
            target_model=run.config['model_name'],
            optimized_message_id=run.config['optimized_message_id'],
            method=run.config['method'],
            implementation=run.config['implementation'],
            random_seed=run.config.get('random_seed', None),
        ))

    metadata_df = pd.DataFrame(metadata_df)
    metadata_df['trigger_id'] = range(len(metadata_df))

    # Load Model
    print(f"Loading model {model_name} for evaluation...")
    model = LMHFModel(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Run Evaluation
    eval_df = evaluate_triggers(  # TODO get from tropts
        model=model,
        trigger_strs=metadata_df['trigger'].tolist(),
        trigger_ids=metadata_df['trigger_id'].tolist(),
        eval_dataset_path=DATASET_PATH,
        batch_size=8
    )
    eval_df['eval_model'] = model_name

    # Merge the DFs by the trigger index
    final_df = pd.merge(
        eval_df, 
        metadata_df, 
        on='trigger_id', 
        how='left'
    )

    # Save
    final_df.to_csv(output_path, index=False)
    print(f"Evaluation complete. Results saved to {output_path}")

# ---------------------------------------------
# Plots
# ---------------------------------------------
@app.command()
def analyze_loss_step_progress(
    message_id: int = typer.Option(225, help="The message ID to analyze"),
    model_name: str = typer.Option("google/gemma-2-2b-it", help="Model name filter"),
    output_plot: str = typer.Option("loss_comparison.png", help="Path to save the resulting plot"),
    compare_tropt_methods: bool = typer.Option(False, help="If True, compares TROPT variants (GCG vs GASLITE) instead of NanoGCG vs Tropt"),
    x_axis: str = typer.Option("time", help="Plot against 'step' or 'time'"),
    max_val: int = typer.Option(None, help="Cutoff for x-axis (steps or minutes)"),
):
    """
    Plots Loss vs Step/Time. Can compare NanoGCG vs Tropt, or Tropt internal methods.
    """
    print(f"Fetching runs for Message ID: {message_id}, Model: {model_name}...")
    
    api = wandb.Api()
    filters = {
        "config.optimized_message_id": message_id,
        "config.model_name": model_name,
        "state": "finished"
    }
    
    # If comparing internal tropt methods, filter explicitly
    if compare_tropt_methods:
        filters["config.implementation"] = "tropt"

    runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters)
    print(f"Found {len(runs)} runs.")
    
    # key -> list of (x_values, loss_values)
    data = {} 
    
    for run in runs:
        impl = run.config.get('implementation')
        method = run.config.get('method', 'gcg') # default to gcg if missing
        
        # Determine the grouping key
        if compare_tropt_methods:
            if impl != 'tropt': continue
            group_key = f"tropt-{method}"
        else:
            if impl not in ['nanogcg', 'tropt']: continue
            group_key = impl

        if group_key not in data: data[group_key] = []

        # Fetch history
        hist = run.history(keys=['loss', '_runtime', '_step'], samples=10000)
        
        if 'loss' in hist:
            hist = hist.dropna(subset=['loss'])
            losses = hist['loss'].values
            
            # Determine X-axis
            if x_axis == "time":
                # Fallback to index if runtime missing, convert to minutes
                if '_runtime' in hist:
                    xs = hist['_runtime'].values / 60.0 
                else:
                    xs = np.arange(len(losses))
            else:
                # Use wandb step or array index
                xs = hist['_step'].values if '_step' in hist else np.arange(len(losses))

            # Sort
            sort_idx = np.argsort(xs)
            data[group_key].append((xs[sort_idx], losses[sort_idx]))

    # --- Plotting ---
    plt.figure(figsize=(8, 5))
    
    # Dynamic colors
    keys = sorted(data.keys())
    cmap = plt.get_cmap("tab10")
    
    all_max_x = []
    for values in data.values():
        for x, _ in values: all_max_x.append(x.max())
            
    if not all_max_x:
        print("No valid data found.")
        return

    limit = max_val if max_val else max(all_max_x)
    common_grid = np.linspace(0, limit, 500)

    for i, group_key in enumerate(keys):
        runs_data = data[group_key]
        interpolated_losses = []
        
        for x, y in runs_data:
            interp_val = np.interp(common_grid, x, y, left=np.nan, right=np.nan)
            interpolated_losses.append(interp_val)
            
        interpolated_losses = np.array(interpolated_losses)
        mean_loss = np.nanmean(interpolated_losses, axis=0)
        std_loss = np.nanstd(interpolated_losses, axis=0)
        
        color = cmap(i)
        plt.plot(common_grid, mean_loss, label=f"{group_key}", color=color, linewidth=2)
        plt.fill_between(
            common_grid, mean_loss - std_loss, mean_loss + std_loss, 
            color=color, alpha=0.2
        )

    plt.xlabel("Step" if x_axis == "step" else "Time (Minutes)")
    plt.ylabel("Loss")
    plt.title(f"Optimization Progress ({x_axis}): Msg {message_id}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")

@app.command()
def analyze_loss_step_all_grid(
    model_name: str = typer.Option("google/gemma-2-2b-it", help="Model name filter"),
    output_plot: str = typer.Option("gcg_grid_comparison.png", help="Path to save the resulting plot"),
    max_val: int = typer.Option(None, help="Cutoff for x-axis (steps or minutes)"),
    x_axis: str = typer.Option("step", help="Plot against 'step' or 'time'"),
    compare_tropt_methods: bool = typer.Option(False, help="If True, compares Tropt variants (GCG vs GASLITE)"),
    sample_indices: List[int] = typer.Option(DEFAULT_INDICES, help="Indices to include in the grid")
):
    """
    Plots a grid of Loss vs Step/Time comparisons for all specified messages.
    Can compare NanoGCG vs Tropt, or Tropt internal methods (GCG vs GASLITE).
    """
    # 1. Setup Grid
    num_plots = len(sample_indices)
    cols = 4
    rows = math.ceil(num_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    
    # Dynamic colors for flexibility
    cmap = plt.get_cmap("tab10")
    
    api = wandb.Api()
    print(f"Generating grid for {num_plots} messages...")

    # 2. Iterate through every message ID
    for i, message_id in enumerate(sample_indices):
        ax = axes[i]
        print(f"Processing Msg ID: {message_id} ({i+1}/{num_plots})...")
        
        filters = {
            "config.optimized_message_id": int(message_id),
            "config.model_name": model_name,
        }
        if compare_tropt_methods:
            filters["config.implementation"] = "tropt"

        runs = api.runs(path=f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters=filters)
        
        data = {} 
        all_max_x = []
        
        for run in runs:
            impl = run.config.get('implementation')
            method = run.config.get('method', 'gcg')

            # Determine Grouping Key
            if compare_tropt_methods:
                if impl != 'tropt': continue
                group_key = f"tropt-{method}"
            else:
                if impl not in ['nanogcg', 'tropt']: continue
                group_key = impl
            
            if group_key not in data: data[group_key] = []
            
            # Fetch history
            hist = run.history(keys=['loss', '_runtime', '_step'], samples=5000)
            if 'loss' in hist:
                hist = hist.dropna(subset=['loss'])
                losses = hist['loss'].values
                
                # Determine X Values
                if x_axis == "time":
                    if '_runtime' in hist:
                        xs = hist['_runtime'].values / 60.0
                    else:
                        xs = np.arange(len(losses)) # Fallback
                else:
                    xs = hist['_step'].values if '_step' in hist else np.arange(len(losses))
                
                # Sort
                sort_idx = np.argsort(xs)
                xs_sorted = xs[sort_idx]
                ls_sorted = losses[sort_idx]
                
                if len(xs_sorted) > 0:
                    data[group_key].append((xs_sorted, ls_sorted))
                    all_max_x.append(xs_sorted.max())

        # --- Plotting on Subplot ---
        if not all_max_x:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Msg {message_id}")
            continue

        limit = max_val if max_val else max(all_max_x)
        common_grid = np.linspace(0, limit, 300)
        
        # Sort keys to ensure consistent colors across subplots
        sorted_keys = sorted(data.keys())

        for k_idx, group_key in enumerate(sorted_keys):
            runs_data = data[group_key]
            if not runs_data: continue
            
            interpolated_losses = []
            for x, y in runs_data:
                interp_val = np.interp(common_grid, x, y, left=np.nan, right=np.nan)
                interpolated_losses.append(interp_val)
            
            interpolated_losses = np.array(interpolated_losses)
            mean_loss = np.nanmean(interpolated_losses, axis=0)
            std_loss = np.nanstd(interpolated_losses, axis=0)
            
            # Use distinct color for each group key
            color = cmap(k_idx)
            
            ax.plot(common_grid, mean_loss, label=group_key, color=color, linewidth=1.5)
            ax.fill_between(
                common_grid, mean_loss - std_loss, mean_loss + std_loss, 
                color=color, alpha=0.2
            )

        ax.set_title(f"Msg ID: {message_id}")
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # Labels on edges only
        if i % cols == 0:
            ax.set_ylabel("Loss")
        if i >= (rows - 1) * cols:
            ax.set_xlabel("Step" if x_axis == "step" else "Time (m)")

    # 3. Cleanup
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Global Legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=len(handles), bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches='tight')
    print(f"Grid analysis complete. Saved to {output_plot}")

@app.command()
def compare_asr_csv(
    csv_path: str = typer.Option("evaluation_results.csv", help="Path to the results CSV"),
):
    """
    Analyzes the evaluation CSV and prints a Markdown table comparing NanoGCG vs TROPT.
    """
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # 1. Normalize Column Names
    # We look for common names for the loss column returned by evaluate_triggers
    loss_col = next((c for c in ['loss', 'eval_loss', 'target_loss'] if c in df.columns), None)
    success_col = next((c for c in ['success', 'is_success'] if c in df.columns), None)
    
    if not loss_col:
        print("Error: Could not find a 'loss' column in the CSV.")
        return

    print(f"Analyzing {len(df)} records using loss column: '{loss_col}'...\n")

    # 2. Global Summary Table
    # Group by implementation to get mean/std across ALL runs
    summary = df.groupby('implementation')[loss_col].agg(['mean', 'std', 'count'])
    if success_col:
        summary['success_rate'] = df.groupby('implementation')[success_col].mean() * 100

    print("### 📊 Global Summary")
    print("| Implementation | Mean Loss | Std Dev | Samples |" + (" Success Rate |" if success_col else ""))
    print("| :--- | :--- | :--- | :--- |" + (" :--- |" if success_col else ""))
    
    for impl, row in summary.iterrows():
        success_str = f" {row['success_rate']:.1f}% |" if success_col else ""
        print(f"| **{impl}** | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |{success_str}")
    print("\n" + "-"*40 + "\n")

    # 3. Per-Message Breakdown (Pivot Table)
    # We average the 3 seeds for each message_id
    grouped = df.groupby(['optimized_message_id', 'implementation'])[loss_col].mean().reset_index()
    
    # Pivot: Index=MsgID, Columns=Implementation, Values=Loss
    pivot = grouped.pivot(index='optimized_message_id', columns='implementation', values=loss_col)
    
    # Calculate Delta (assuming we have both keys)
    if 'nanogcg' in pivot.columns and 'tropt' in pivot.columns:
        pivot['delta'] = pivot['nanogcg'] - pivot['tropt'] # Positive = TROPT is better (lower loss)
        pivot['winner'] = pivot['delta'].apply(lambda x: 'TROPT' if x > 0 else 'NanoGCG')
    
    print("### 🧐 Per-Message Breakdown (Avg over seeds)")
    headers = "| Msg ID | NanoGCG Loss | TROPT Loss | Delta (Nano - TROPT) | Winner |"
    print(headers)
    print("| :--- | :--- | :--- | :--- | :--- |")

    for msg_id, row in pivot.iterrows():
        n_loss = row.get('nanogcg', float('nan'))
        t_loss = row.get('tropt', float('nan'))
        
        # Formatting for missing data
        n_str = f"{n_loss:.4f}" if not pd.isna(n_loss) else "N/A"
        t_str = f"{t_loss:.4f}" if not pd.isna(t_loss) else "N/A"
        
        delta_str = ""
        winner_str = ""
        
        if not pd.isna(n_loss) and not pd.isna(t_loss):
            delta = n_loss - t_loss
            # Bold the winner logic
            if delta > 0.001: # TROPT wins significantly
                delta_str = f"🟢 +{delta:.4f}"
                winner_str = "**TROPT**"
            elif delta < -0.001: # NanoGCG wins significantly
                delta_str = f"🔴 {delta:.4f}"
                winner_str = "NanoGCG"
            else:
                delta_str = "0.0000"
                winner_str = "Tie"
        
        print(f"| {msg_id} | {n_str} | {t_str} | {delta_str} | {winner_str} |")

    print("\n*Note: Positive Delta (🟢) means TROPT achieved lower loss.*")


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

@app.command()
def plot_universality_violin(
    input_path: str= typer.Option("evaluation_results.csv", help="Path to the evaluation results CSV"),
    output_csv_path: str= typer.Option("universality_scores.csv", help="Output CSV file path"),
    plot_path: str= typer.Option("universality_plot.png", help="Output image file path")):
    # 1. Load Data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
        return

    # 2. Calculate Universality Score
    # Group by implementation and trigger_id, then average the metric across messages
    universality_df = df.groupby(['implementation', 'method', 'trigger_id'])['strongreject_finetuned'].mean().reset_index()
    universality_df = universality_df[universality_df.implementation == 'tropt']
    universality_df.rename(columns={'strongreject_finetuned': 'universality_score'}, inplace=True)
    universality_df['name'] = universality_df['implementation'] + '--' + universality_df['method']

    # 3. Save to CSV
    universality_df.to_csv(output_csv_path, index=False)
    print(f"Universality scores saved to: {output_csv_path}")

    # 4. Generate Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Violin plot for distribution shape
    sns.violinplot(
        data=universality_df,
        x='name',
        y='universality_score',
        inner=None,  # Remove inner bars to keep it clean for the strip plot
        color='lightblue'
    )

    sns.pointplot(
        data=universality_df,
        x='name',
        y='universality_score',
        errorbar=None,     # Just the mean, no confidence intervals
        color='red',       # Distinct color
        markers='_',       # Use a line marker
        scale=2,           # Make the line wider
        join=False         # Do not connect the means
    )

    # Strip plot to show individual data points
    sns.stripplot(
        data=universality_df,
        x='name',
        y='universality_score',
        color='black',
        alpha=0.6,
        jitter=True
    )

    plt.title('Universality Score Distribution by GCG Implementation', fontsize=14)
    plt.ylabel('Universality Score (Avg StrongReject)', fontsize=12)
    plt.xlabel('Implementation', fontsize=12)
    
    # Save Plot
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Violin plot saved to: {plot_path}")

@app.command()
def plot_universality_box(
    input_path: str= typer.Option("evaluation_results.csv", help="Path to the evaluation results CSV"),
    output_csv_path: str= typer.Option("universality_scores.csv", help="Output CSV file path"),
    plot_path: str= typer.Option("universality_boxplot.png", help="Output image file path")):
    
    # 1. Load Data
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
        return

    # 2. Calculate Universality Score
    universality_df = df.groupby(['implementation', 'method', 'trigger_id'])['strongreject_finetuned'].mean().reset_index()
    universality_df.rename(columns={'strongreject_finetuned': 'universality_score'}, inplace=True)
    universality_df['name'] = universality_df['implementation'] + '--' + universality_df['method']

    # 3. Save to CSV
    universality_df.to_csv(output_csv_path, index=False)
    print(f"Universality scores saved to: {output_csv_path}")

    # 4. Generate Box Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Box plot
    sns.boxplot(
        data=universality_df,
        x='name',
        y='universality_score',
        color='lightblue',
        showfliers=False,  # Hide outliers (since we use stripplot below)
        showmeans=True,    # Add mean
        meanline=True,     # Draw mean as a line, not a point
        meanprops={'color': 'red', 'ls': '-', 'lw': 2, 'label': 'Mean'}
    )

    # Strip plot (individual points)
    sns.stripplot(
        data=universality_df,
        x='name',
        y='universality_score',
        color='black',
        alpha=0.6,
        jitter=True
    )

    plt.title('Universality Score Distribution by Implementation', fontsize=14)
    plt.ylabel('Universality Score', fontsize=12)
    plt.xlabel('Implementation', fontsize=12)
    
    # Save Plot
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Box plot saved to: {plot_path}")

@app.command()
def analyze_tropt_violin(
    input_path: str = typer.Option("evaluation_results.csv", help="Path to results CSV"),
    plot_path: str = typer.Option("tropt_variant_violin.png", help="Output path"),
    metric: str = typer.Option("strongreject_finetuned", help="Metric to plot"),
):
    """
    Generates a violin plot comparing strictly Tropt variants (GCG vs GASLITE, etc).
    """
    if not os.path.exists(input_path):
        print("Input file not found.")
        return

    df = pd.read_csv(input_path)
    
    # Filter for Tropt only
    df = df[df['implementation'] == 'tropt'].copy()
    
    if df.empty:
        print("No tropt data found in CSV.")
        return

    # Ensure method column exists (fallback to config if needed, but CSV usually has it)
    if 'method' not in df.columns:
        print("Error: 'method' column missing from CSV.")
        return

    print(f"Plotting {metric} for methods: {df['method'].unique()}")

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # Order ensures consistent comparison
    order = sorted(df['method'].unique())

    # Violin
    sns.violinplot(
        data=df, x='method', y=metric,
        order=order, inner=None, palette="muted"
    )

    # Strip (Dots)
    sns.stripplot(
        data=df, x='method', y=metric,
        order=order, color='black', alpha=0.4, jitter=True
    )

    plt.title(f'Tropt Variants Comparison: {metric}')
    plt.ylabel('Score')
    plt.xlabel('Optimization Method')
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Violin comparison saved to {plot_path}")

if __name__ == "__main__":
    app()
