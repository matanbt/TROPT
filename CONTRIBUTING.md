
[TODONOW] decouple all the contribution stuff to here, instead of in the docs. Docs are dedicated to 

## Contributing a recipe to the *Recipe Hub*

the common pattern for these function; naming convention etc.

If you want to contribute the recipe to the package's Recipe Hub (not just use it in your own script), add your function to `tropt/recipe_hub/__init__.py`. Use the [naming convention](#naming-convention) — the function name and dict key must be identical.

```python
# Bare name (not a paper reproduction):
from tropt.recipe_hub.myattack import run_myattack

RECIPES = {
    ...
    "myattack": run_myattack,
}

# Paper reproduction (algorithm + hparams faithful to first-author+year paper):
from tropt.recipe_hub.MyAttack import run_myattack__doe2024

RECIPES = {
    ...
    "myattack__doe2024": run_myattack__doe2024,
}
```





<!-- [[TODO make this more concise, and this should ONLY be part of the contributing secetion:]] -->
### Naming convention

Recipe entry-point function names and the `RECIPES` dict keys in `tropt/recipe_hub/__init__.py` use the same string. The format is:

```
{method}[_{variant}][_{task}][__{paperYYYY}]
```

- `method` — short canonical name (`gcg`, `arca`, `beast`, ...).
- `_{variant}` — algorithmic variant, when one method has several (`gcgp_whitebox`, `gcgp_blackbox`, `gcg_perplexity`).
- `_{task}` — distinct task application of the same method (`advdecoding_jailbreak` vs `advdecoding_retrieval`, `uat_classifier` vs `uat_prompt_injection`).
- `__{paperYYYY}` — the **reproduction tag** (double underscore; first author + year, lowercased, no separator). Reserved for recipes that precisely reproduce a published method.

### When to use the reproduction tag

Use `__{paperYYYY}` *only* if all three hold:

1. The recipe implements the paper's algorithm step-by-step (no different optimizer, no different per-step rule, no missing core ingredient like a buffer or a momentum term).
2. The recipe's defaults match the hyperparameters the paper reports (or directly cites from the paper's reference implementation).
3. If the recipe ports the method to a different setting than the paper's main experiment (e.g. CLIP→causal-LM), the algorithm transfers cleanly and the loss function is the canonical analogue. State the port in the docstring.

If any of these fails — different optimizer, missing scheduling, hparams not from the paper, recipe introduces non-paper knobs as defaults — **omit the tag**. Use the bare `{method}[_{variant}][_{task}]` form instead.

Examples (live in the repo):

| Recipe key | Why this form |
| --- | --- |
| `gcg__zou2023` | Algorithm 1 + paper hparams (B=512, k=256, T=500). |
| `gcg_perplexity` | GCG composed with `TriggerPerplexityLoss`; not in any paper. |
| `gcgp_whitebox__hayase2024` | GCG+ white-box variant from §4.1 of the QCG paper. |
| `prs` | Recipe documents explicit deviations from the paper (schedule, restarts, no judge). |
| `arca_toxic_reverse` | Reproduces the *task* of Jones et al. §4.2.1 but uses GCG instead of ARCA — algorithm differs. |

### Multiple recipes from the same paper

When a single paper introduces multiple variants and the recipe hub exposes them as separate functions, each one carries the same paper tag, with the variant slot disambiguating: `pal__sitawarin2024`, `ral__sitawarin2024`, `gcgp_pal__sitawarin2024`.



## Contributing a loss

If you want to contribute the loss to the package (not just use it in your own script):

1. **Register** — Export from [`tropt/loss/__init__.py`](../../tropt/loss/__init__.py).
2. **Test** — Add tests under `tests/loss/`.


## Contributing an optimzier



