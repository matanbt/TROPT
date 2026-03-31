"""
Evaluating safety of responses to harmful instructions, using StrongReject's API and different methods.
https://strong-reject.readthedocs.io/en/latest/api/index.html

Requires the `evaluate` extra: ``uv sync --extra evaluate``
"""

from typing import Any, Dict, List, Literal

import pandas as pd
import torch
import wandb
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset
from transformers import pipeline

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER


def evaluate_jailbreakness_of_responses(
        instructions: List[str], responses: List[str],
        override_evaluators: List[str] = None,
        batch_size: int = 8
    ) -> Dict[str, List[float]]:
    """
    Returns: a DataFrame with the evaluation results; column for `score` (between 0.0 to 1.0) and `evaluator` name.
    """
    assert len(instructions) == len(responses), "instructions-responses mismatch"

    dataset = Dataset.from_dict({
        "forbidden_prompt": instructions,
        "response": responses,
        "id": list(range(len(instructions)))
    })

    evaluators = override_evaluators or [
        # "string_matching",
        "strongreject_finetuned",  # https://huggingface.co/qylu4156/strongreject-15k-v1
        # "harmbench"  # https://huggingface.co/cais/HarmBench-Llama-2-13b-cls; https://github.com/centerforaisafety/HarmBench/blob/main/docs/evaluation_pipeline.md#%EF%B8%8F-step-3---evaluate-completions
        # gpt4_judge
    ]
    eval_dataset = evaluate_dataset(
        dataset,
        evaluators=evaluators,
        empty_model_cache=False,
        batch_size=batch_size,
    )

    eval_dataset = eval_dataset.to_pandas()
    eval_dataset = eval_dataset.sort_values(by="id")

    ## Map each evaluator name to a list of scores, sorted by id
    eval_dataset = eval_dataset.pivot_table(index=["id", "forbidden_prompt", "response"],
                                            columns="evaluator", values="score").reset_index()
    eval_dataset = eval_dataset.sort_values(by="id")  # sort according to the input one

    # Extract the scores of each evaluatr to a list
    evaluator_to_score_list = {}
    for evaluator in evaluators:
        evaluator_to_score_list[evaluator] = eval_dataset[evaluator].tolist()
    return evaluator_to_score_list

    # return {
    #     "mean_metrics": eval_dataset.drop(columns='id').mean(numeric_only=True).to_dict(),
    #     "jailbreakness_per_message": eval_dataset.to_dict(orient='records'),
    #     "jailbreakness_per_evaluator": evaluator_to_score_list
    # }

ADVBENCH_PLUS_PATH = "scripts/attack_evaluate/advbench_plus.csv"
CLEARHARM_PATH = "scripts/attack_evaluate/clearharm.csv"

def _generate_responses_litellm(
    messages_list: List[List[dict]], model_name: str, max_new_tokens: int,
) -> List[str]:
    """Generate responses via LiteLLM for API-based models."""
    from tropt.model.litellm_proxy.lm import LiteLLMModel

    model = LiteLLMModel(model_name=model_name)
    texts = [msgs[0]["content"] for msgs in messages_list]
    output = model.invoke_from_texts(
        input_texts=texts,
        require_generation=True,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    return output.generated_response_strs


def evaluate_triggers(
    model_name: str,
    trigger_strs: List[str],
    trigger_ids: List[Any] = None,
    harmful_dataset: Literal["advbench_plus", "clearharm"] = "clearharm",
    batch_size: int = 128,
    greedy_decode: bool = True,
    max_new_tokens: int = 128,
    model_backend: Literal["hf_pipeline", "litellm"] = "hf_pipeline",
    evaluators: List[str] = None,
    eval_batch_size: int = 16,
) -> pd.DataFrame:
    """
    Evaluate a list of triggers on a behavior dataset, returning a DataFrame with jailbreakness scores.

    Args:
        model_name: Model name/path to evaluate (HF identifier or LiteLLM model string).
        trigger_strs: List of trigger strings to evaluate.
        model_backend: "hf_pipeline" for local HuggingFace, "litellm" for API-based generation.
        evaluators: StrongReject evaluator names (default: ["strongreject_finetuned"]).
        eval_batch_size: Batch size for the StrongReject evaluator.
    """
    evaluators = evaluators or ["strongreject_finetuned"]

    # Load HF model directly via pipeline (isolated, no TROPT wrapping)
    pipe = None
    if model_backend == "hf_pipeline":
        pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    # Load behavior dataset
    # with columns: 'message', 'target_response_prefix', 'source', 'template_message'
    if harmful_dataset == "advbench_plus":
        base_df = pd.read_csv(ADVBENCH_PLUS_PATH)
    elif harmful_dataset == "clearharm":
        base_df = pd.read_csv(CLEARHARM_PATH)
    else:
        raise ValueError(f"Unsupported harmful_dataset: {harmful_dataset}")

    base_df['message_id'] = range(len(base_df))

    if trigger_ids is None:
        trigger_ids = list(range(len(trigger_strs)))

    all_results = []

    for trigger_id, trigger_str in zip(trigger_ids, trigger_strs):
        # Create a copy for this specific trigger to avoid modifying base_df
        df = base_df.copy()

        # Add metadata columns
        df['trigger_id'] = trigger_id
        df['trigger_str'] = trigger_str

        # Create triggered messages
        df['triggered_message'] = df['message_template'].apply(
            lambda x: x.replace(OPTIMIZED_TRIGGER_PLACEHOLDER, trigger_str)
        )

        # Get model responses
        messages = [[{"role": "user", "content": t}] for t in df['triggered_message']]
        if model_backend == "litellm":
            df['response'] = _generate_responses_litellm(
                messages, model_name, max_new_tokens,
            )
        else:
            outputs = pipe(
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=not greedy_decode,
                batch_size=batch_size,
                return_full_text=False,
            )
            df['response'] = [out[0]["generated_text"] for out in outputs]

        # Evaluate jailbreakness
        metric_to_scores = evaluate_jailbreakness_of_responses(
            instructions=df['message'].tolist(),
            responses=df['response'].tolist(),
            override_evaluators=evaluators,
            batch_size=eval_batch_size,
        )

        # Add metrics to df (one column per metric)
        for metric in metric_to_scores:
            df[metric] = metric_to_scores[metric]

        all_results.append(df)

    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)

    return final_df

#### Helper scripts to run evaluation from Wandb runs ####
WANDB_ENTITY = "my_username_or_team"  # modify to your wandb entity (username or team name)
WANDB_PROJECT = "tropt-runs"
DATASET_PATH = "scripts/attack_evaluate/advbench_plus.csv"
def wandb_to_trigger_eval_pipeline(
    model_name: str = "google/gemma-2-2b-it",
    output_path: str = "evaluation_results.csv",

    asr_measure = 'strongreject_finetuned',
    output_csv_path: str = "universality_scores.csv"
):
    api = wandb.Api()
    runs = api.runs(
        path=f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        filters={"state": "finished"}
    )

    metadata_df = []

    for run in runs:
        # Extract the optimized trigger
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

    # Run Evaluation
    eval_df = evaluate_triggers(
        model_name=model_name,
        trigger_strs=metadata_df['trigger'].tolist(),
        trigger_ids=metadata_df['trigger_id'].tolist(),
        harmful_dataset="clearharm",
        batch_size=8,
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

    ## Create universality dataframe
    universality_df = final_df.groupby(['implementation', 'method', 'trigger_id'])[asr_measure].mean().reset_index()
    universality_df = universality_df[universality_df.implementation == 'tropt']
    universality_df.rename(columns={asr_measure: 'universality_score'}, inplace=True)
    universality_df['name'] = universality_df['implementation'] + '--' + universality_df['method']

    # 3. Save to CSV
    universality_df.to_csv(output_csv_path, index=False)
    print(f"Universality scores saved to: {output_csv_path}")
