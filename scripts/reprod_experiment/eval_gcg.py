"""
Run GCG on specific advbench samples, using two different implementations.

Plotting / analysis lives in analysis.ipynb (same directory).
"""
import os
from typing import List

import pandas as pd
import torch
from tropt.attack_zoo.BEAST import run_beast
from tropt.attack_zoo.GCG import run_gcg, run_gcg_perplexity
from tropt.attack_zoo.GCGHij import run_gcghij
from tropt.attack_zoo.IRIS import run_iris
from tropt.attack_zoo.RASLITEPlus import run_rasliteplus_llm
import typer
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.attack_evaluate.evaluate_jailbreakness import evaluate_triggers
from scripts.reprod_experiment.nanogcg import GCG, GCGConfig
from tropt.common import Targets
from tropt.loss import PrefillCELoss
from tropt.model.huggingface.lm import LMHFModel
from tropt.optimizer import OptimizerResult
from tropt.optimizer.gcg_optimizer import GCGOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker import WandbTracker
from tropt.utils.refusal_dir import compute_refusal_directions

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
        tracker.finish()

    
# All LLM attack zoo methods available for tropt_zoo
_LLM_ZOO_METHODS = ["gcg", "beast", "iris", "gcg_hij", "gcg_perplexity", "rasliteplus_llm"]


@app.command()
def tropt_zoo(
    model_name: str = typer.Option("google/gemma-2-2b-it", help="HuggingFace model identifier"),
    sample_indices: List[int] = typer.Option(
        DEFAULT_INDICES,
        help="Specific indices to run. Usage: --sample-indices 225 --sample-indices 89",
    ),
    methods: List[str] = typer.Option(
        _LLM_ZOO_METHODS,
        help=f"Attack methods to run. Available: {_LLM_ZOO_METHODS}",
    ),
    seed: int = typer.Option(RANDOM_SEED, help="Random seed for reproducibility"),
):
    """
    Run multiple LLM attack-zoo recipes on advbench samples.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(DATASET_PATH)
    df["message_id"] = range(len(df))

    # Load model once; use_prefix_cache=False for broad attack compatibility
    model = LMHFModel(
        model_name=model_name,
        device=device,
        forward_pass_batch_size=1024,
        use_prefix_cache=False,
        dtype="bfloat16",
    )

    for message_id in sample_indices:
        row = df.loc[df.message_id == message_id].iloc[0]
        instruction = row["message_template"]
        target = row["target_response_prefix"]

        for method in methods:
            run_name = f"tropt[{method},{model_name.split('/')[-1]},m={message_id}]"
            if seed != RANDOM_SEED:
                run_name += f"_s={seed}"

            tracker = WandbTracker(
                run_name,
                tags=["tropt", method],
                project_name=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                config_dump=dict(
                    optimized_message_id=message_id,
                    name=run_name,
                    model_name=model_name,
                    implementation="tropt",
                    method=method,
                    random_seed=seed,
                ),
            )

            print(f"Running: {run_name}")
            if method == "gcg":
                run_gcg(instruction=instruction, target_response=target, model_obj=model, tracker=tracker)
            elif method == "beast":
                run_beast(instruction=instruction, target_output=target, model_obj=model, tracker=tracker)
            elif method == "iris":
                run_iris(
                    instruction=instruction,
                    model_obj=model,
                    tracker=tracker,
                    initial_trigger=INITIAL_TRIGGER,
                )
            elif method == "gcg_hij":
                run_gcghij(instruction=instruction, target_output=target, model_obj=model, tracker=tracker)
            elif method == "gcg_perplexity":
                run_gcg_perplexity(instruction=instruction, target_response=target, model_obj=model, tracker=tracker)
            elif method == "rasliteplus_llm":
                run_rasliteplus_llm(instruction=instruction, model_obj=model, tracker=tracker)
            else:
                raise typer.BadParameter(f"Unknown method '{method}'. Available: {_LLM_ZOO_METHODS}")

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


if __name__ == "__main__":
    app()
