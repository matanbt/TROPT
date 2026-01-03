"""
Evaluating safety of responses to harmful instructions, using StrongReject's API and different methods.
https://strong-reject.readthedocs.io/en/latest/api/index.html
`pip install git+https://github.com/dsbowen/strong_reject.git@main`
"""
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset
from strong_reject.evaluate import evaluate_dataset

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER
from tropt.models.base import LMBaseModel
from tropt.optimizer.base import OptimizerResult


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


def evaluate_triggers(
    model: LMBaseModel,
    trigger_strs: List[str],
    trigger_ids: List[Any]=None,
    eval_dataset_path: str="scripts/attack_evaluate/advbench_plus.csv",
    batch_size: int = 128,
) -> pd.DataFrame:
    """
    Evaluate a list of triggers on a behavior dataset, returning a DataFrame with jailbreakness scores.

    Args:
        model (LMBaseModel): The language model to evaluate.
        trigger_strs (List[str]): List of trigger strings to evaluate.

    """
    # Load behavior dataset
    # with columns: 'message', 'target_response_prefix', 'source', 'template_message'
    base_df = pd.read_csv(eval_dataset_path)
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

        # Get model responses in batches
        responses = []
        for idx in range(0, len(df), batch_size):
            batch_df = df.iloc[idx: idx + batch_size]
            responses.extend(model(batch_df['triggered_message'].tolist()))
        df['response'] = responses

        # Evaluate jailbreakness
        metric_to_scores = evaluate_jailbreakness_of_responses(
            instructions=df['message'].tolist(),
            responses=df['response'].tolist(),
            override_evaluators=["strongreject_finetuned"],  # TODO make configurable
            batch_size=32,  # TODO make configurable
        )

        # Add metrics to df (one column per metric)
        for metric in metric_to_scores:
            df[metric] = metric_to_scores[metric]

        all_results.append(df)

    # Combine all results
    final_df = pd.concat(all_results, ignore_index=True)

    return final_df

# TODO wrap the previous function with a function that:
# (a) fetches the different triggers (loss and str) from Wandb (NanoGCG/ourGCG/other)
# (b) runs the evaluate_triggers function
# (c) adds the metadata per trigger
# (d) saves the final results to a CSV
# --> analyze the results in a separate script
#     print summary statistics & plot the
#     a violin plot of jailbreakness per method.
