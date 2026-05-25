# Contributing to TROPT

This document covers contributions back to the **TROPT package itself**: registering new components, naming conventions, tests, and Recipe Hub entries.

If you only want to *use* TROPT (with custom losses, optimizers, models, or recipes living in your own scripts), the user-facing guides under [`docs/guides/`](docs/guides/) are what you want; they are intentionally scoped to library use, with no contribution boilerplate. The sections below assume you have already followed the relevant guide and have a working component.

<!-- TODONOW REREAD IT  -->



For contributors and developers: See [DESIGN.md](DESIGN.md) for comprehensive design philosophy and architectural details.


```bash
# Install in development mode
uv sync --extra dev

# Run tests
uv run pytest

# Run linting
uv run ruff check .
```


---

## Contributing a Loss

After implementing your loss per [`docs/guides/adding_a_loss.md`](docs/guides/adding_a_loss.md):

1. **File placement**: Add the loss to the appropriate file under [`tropt/loss/`](tropt/loss/) (or create a new module if it's a new category).
2. **Register / export**: Export the loss class from [`tropt/loss/__init__.py`](tropt/loss/__init__.py).
3. **Tests**: Add unit tests under `tests/loss/`. Cover output shape, sign convention, and at least one known input/output pair (numerical correctness matters).
4. **Compatibility matrix**: Re-run `python docs/scripts/generate_compat_matrix.py` and commit the regenerated `docs/guides/compatibility_matrix.md`.

---

## Contributing an Optimizer

After implementing your optimizer per [`docs/guides/adding_an_optimizer.md`](docs/guides/adding_an_optimizer.md):

1. **File placement**: Add the optimizer module under [`tropt/optimizer/`](tropt/optimizer/) (one optimizer per file is the convention).
2. **Register / export**: Export the optimizer class from [`tropt/optimizer/__init__.py`](tropt/optimizer/__init__.py).
3. **Tests**: Add tests under `tests/optimizer/`. Cover `model_requirements` validation, basic optimization, and any optimizer-specific features (schedulers, restarts, buffers, etc.). Use `tests/optimizer/` as a template.
4. **Recipe Hub entry** *(optional but recommended for published methods)*: Add a recipe in `tropt/recipe_hub/` that exposes the optimizer with paper-faithful defaults (see "Contributing a Recipe" below).
5. **Compatibility matrix**: Re-run `python docs/scripts/generate_compat_matrix.py` and commit the regenerated `docs/guides/compatibility_matrix.md`.

---

## Contributing a Model Backend

After implementing your model per [`docs/guides/adding_a_model.md`](docs/guides/adding_a_model.md):

1. **File placement**: Create a directory under [`tropt/model/`](tropt/model/) for your backend (e.g. `tropt/model/my_backend/`), with one file per concrete model class.
2. **Register / export**: Export the model class from [`tropt/model/__init__.py`](tropt/model/__init__.py).
3. **Tests**: Add tests under `tests/models/`. Cover initialization, the inference method, and each mixin method, in both single- and multi-template cases.
4. **Compatibility matrix**: Re-run `python docs/scripts/generate_compat_matrix.py` and commit the regenerated `docs/guides/compatibility_matrix.md`.

---

## Contributing a Recipe to the *Recipe Hub*

After implementing your recipe per [`docs/guides/adding_a_recipe.md`](docs/guides/adding_a_recipe.md):

1. **File placement**: Add a Python file under [`tropt/recipe_hub/`](tropt/recipe_hub/) following the [naming convention](#naming-convention) below. One paper / family of variants per file.
2. **Register**: Add the entry-point function to [`tropt/recipe_hub/__init__.py`](tropt/recipe_hub/__init__.py). The function name and the `RECIPES` dict key must be identical:

    ```python
    # Bare name (not a paper reproduction):
    from tropt.recipe_hub.MyAttack import myattack

    RECIPES = {
        ...
        "myattack": myattack,
    }

    # Paper reproduction (algorithm + hyperparameters faithful to first-author + year):
    from tropt.recipe_hub.MyAttack__doe2024 import myattack__doe2024

    RECIPES = {
        ...
        "myattack__doe2024": myattack__doe2024,
    }
    ```

3. **README**: Add a row to the appropriate section of [`tropt/recipe_hub/README.md`](tropt/recipe_hub/README.md) (key, description, target model, required access, paper, file).
4. **Smoke test**: Make sure the recipe runs end-to-end on a small model (e.g. `google/gemma-3-270m-it`) before submitting.

### Naming convention

Recipe entry-point function names and the `RECIPES` dict keys use the same string. The format is:

```
{method}[_{variant}][_{task}][__{paperYYYY}]
```

- `method`: short canonical name (`gcg`, `arca`, `beast`, ...).
- `_{variant}`: algorithmic variant, when one method has several (`gcgp_whitebox`, `gcgp_blackbox`, `gcg_perplexity`).
- `_{task}`: distinct task application of the same method (`advdecoding_jailbreak` vs `advdecoding_retrieval`, `uat_classifier` vs `uat_prompt_injection`).
- `__{paperYYYY}`: the **reproduction tag** (double underscore; first author + year, lowercased, no separator). Reserved for recipes that precisely reproduce a published method.

#### When to use the reproduction tag

Use `__{paperYYYY}` *only* if all three hold:

1. The recipe implements the paper's algorithm step-by-step (no different optimizer, no different per-step rule, no missing core ingredient like a buffer or a momentum term).
2. The recipe's defaults match the hyperparameters the paper reports (or directly cites from the paper's reference implementation).
3. If the recipe ports the method to a different setting than the paper's main experiment (e.g. CLIP→causal-LM), the algorithm transfers cleanly and the loss function is the canonical analogue. State the port in the docstring.

If any of these fails (different optimizer, missing scheduling, hparams not from the paper, recipe introduces non-paper knobs as defaults), **omit the tag**. Use the bare `{method}[_{variant}][_{task}]` form instead.

Examples (live in the repo):

| Recipe key | Why this form |
| --- | --- |
| `gcg__zou2023` | Algorithm 1 + paper hparams (B=512, k=256, T=500). |
| `gcg_perplexity` | GCG composed with `TriggerPerplexityLoss`; not in any paper. |
| `gcgp_whitebox__hayase2024` | GCG+ white-box variant from §4.1 of the QCG paper. |
| `prs` | Recipe documents explicit deviations from the paper (schedule, restarts, no judge). |
| `arca_toxic_reverse` | Reproduces the *task* of Jones et al. §4.2.1 but uses GCG instead of ARCA; algorithm differs. |

#### Multiple recipes from the same paper

When a single paper introduces multiple variants and the recipe hub exposes them as separate functions, each one carries the same paper tag, with the variant slot disambiguating: `pal__sitawarin2024`, `ral__sitawarin2024`, `gcgp_pal__sitawarin2024`.

---

## Development Workflow

```bash
# Install in development mode with all dependencies
uv sync --all-extras

# Install pre-commit hooks
pre-commit install

# Lint / type-check / test
uv run ruff check
uv run ty check
uv run pytest
```

