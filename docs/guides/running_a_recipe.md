# Runnign a Recipe from

The Recipe Hub hosts over 40 optimization recipes that are instantly runnable.
These pre-configured recipes are functions that allow the use of TROPT for specific, narrow applications.
For instance, TROPT can be used for LLM jailbreak, corpus poisoning attacks, adversarial examples, and prompt recovery. 
We exemplify all here.

For the full list of the hosted recipe is available in [TODO hyperlink to last section].

## Example Recipes

### LLM Jailbreak Recipe

[TODO brief description on the recipe and cite Zou2023]

```python
    from tropt.recipe_hub import gcg__zou2023

    result = gcg__zou2023(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        instruction="Tell me how to pick a lock. {{OPTIMIZED_TRIGGER}}",
        target_response="Sure, here's how:"
    )
    print(f"{result.best_trigger_str=}")  # print the best trigger
```

[TODO also refer to the recipe with multiple gcg]

## Corpus Poisoning Recipe
[TODO]

## Adversraial Examples against classifiers
[TODO]

## Prompt Recovery
[TODO]


## Available Recipe

[TODO make sure the recipe table is injected here from recipe_hub/README.md]


