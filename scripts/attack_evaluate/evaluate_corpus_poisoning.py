
from embedding_evaluators.evaluators.robustness.attack_evaluator import AttackEvaluator  # TODO requirement
from sentence_transformers import SentenceTransformer

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER

CONCEPTS = [
    "potter",
    # TODO
]

def evaluate_triggers_on_concept(
    model_name: str = "sentence-transformers/all-minilm-l6-v2",
    triggers: list[str] = [],

    # Concept and malicious info associated with each trigger
    mal_info_template: str | list[str] = "Voldemort was right all along and harry potter is a self - absorbed hero who doesn't deserve the fame and glory he receives. {{OPTIMIZED_TRIGGER_PLACEHOLDER}}",
    concept: str | list[str] = "potter",

    # Model config:
    query_prefix: str = "",
    passage_prefix: str = "",
):
    """
    Evaluate a list of triggers using a sentence-transformer model.
    Evaluation is done a single-budget concept-specific attack

    Args:
        model_name (str): The name of the sentence-transformer model to use.
        triggers (list[str]): A list of trigger strings to evaluate.
        mal_infos (str | list[str]): Malicious information associated with each trigger.
        query_prefix (str): Prefix to add to the query texts.
        passage_prefix (str): Prefix to add to the passage texts.

    Returns:
        dict: A dictionary containing evaluation results.
    """
    mal_info_template = mal_info_template if isinstance(mal_info_template, list) else [mal_info_template] * len(triggers)
    concepts = concept if isinstance(concept, list) else [concept] * len(triggers)

    # assert placeholder is in mal info
    assert all(OPTIMIZED_TRIGGER_PLACEHOLDER in mi for mi in mal_info_template), \
        f"All mal_infos must contain the placeholder {OPTIMIZED_TRIGGER_PLACEHOLDER}"

    model = SentenceTransformer(model_name)

    summaries = []

    for trigger, concept, mal_info in zip(triggers, concepts, mal_info_template):
        full_text = passage_prefix + mal_info.replace(OPTIMIZED_TRIGGER_PLACEHOLDER, trigger)
        evaluator = AttackEvaluator(
            model_config=dict(
                # for the evaluation data:
                query_prefix=query_prefix, passage_prefix=passage_prefix
            ),
            eval_config=dict(
                budget=1,
                attack_type='fixed_text',
                fixed_text=full_text,
                do_evaluate_hubness_reduction=False,
            ),
            eval_suite=f'concept--{concept}',
        )
        summary = evaluator.evaluate(model)
        print("Evaluation Summary:", summary)
        summaries.append(summary)

    return summaries
