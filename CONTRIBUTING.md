# Contributing to TROPT

This document covers contributions back to the **TROPT package itself**: registering new components, naming conventions, tests, and Recipe Hub entries.

If you only want to *use* TROPT — with custom losses, optimizers, models, or recipes living in your own scripts — the user-facing guides under [`docs/guides/`](docs/guides/) are what you want; they are intentionally scoped to library use, with no contribution boilerplate. The sections below assume you have already followed the relevant guide and have a working component.

## Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) (v2.1). Be respectful, assume good faith, and keep discussion technical. Report concerns to the maintainer listed in [`pyproject.toml`](pyproject.toml).

## Setup

```bash
git clone https://github.com/matanbt/tropt.git
cd tropt
uv sync --all-extras    # dev install with all optional extras
pre-commit install      # optional today (hook config WIP)
```

We use [`uv`](https://docs.astral.sh/uv/) for dependency management — always invoke tools via `uv run` (e.g. `uv run pytest`).

## Required checks

Before opening a PR, run these locally — they cover what the maintainers will check first:

```bash
uv run ruff check                       # lint
uv run ruff format --check              # formatting
uv run ty check                         # type-check
uv run pytest                           # tests (CI runs this on every PR)
python docs/scripts/generate_compat_matrix.py   # if you touched model/, optimizer/, or any mixin
```

CI today enforces only `pytest`; the others are local-only but expected to pass.

## Reporting bugs

Open an issue with `[bug]` in the title. Include:
- Python + TROPT versions and platform (`uv pip show tropt`, OS).
- Minimal repro (the smallest model and shortest script that triggers it — `google/gemma-3-270m-it` is fine for white-box paths).
- Full traceback, not a summary.
- What you expected vs. what happened.

## Requesting features / proposing recipes

Open an issue with `[feature]` or `[recipe]` before writing code, especially for non-trivial changes. Include motivation, the paper or use case driving it, and (for new recipes) which existing component is missing or needs extending. This avoids wasted effort when a request doesn't fit the four-component design.

## PR workflow

1. Fork → create a topic branch off `main` (e.g. `feature/my-loss`, `fix/gcg-retokenize`).
2. Make focused commits — one logical change per commit. Use clear messages; reference the issue number if there is one.
3. Run the [required checks](#required-checks) locally.
4. Open a PR against `main`. The description should state *what* and *why*, link the issue, and call out any breaking changes or new dependencies.
5. Keep the PR small. Split refactors from feature work when feasible.

## Read the architecture first

A contribution that touches a component must conform to its interface contract. Read in this order:

- [`DESIGN.md`](DESIGN.md) — full design philosophy (modularity, backend vs. frontend, why the contracts are the way they are).
- [`docs/guides/`](docs/guides/) — step-by-step per-component walkthroughs (source of truth for each contribution type).
- **Agent files.**
    - [`CLAUDE.md`](CLAUDE.md) — fast-path orientation: the four components, mixin contract, `ModelInput`/`ModelOutput`/`Targets`, setup-then-compute.
    - [`skills/tropt/SKILL.md`](skills/tropt/SKILL.md) — task-routed pointers and cross-cutting pitfalls (mixin mismatches, thinking-model alignment, multi-model OOM, etc.).


## Adding a component

| Adding | Read first | Module goes in | Export from |
| --- | --- | --- | --- |
| a **loss** | [`adding_a_loss.md`](docs/guides/adding_a_loss.md) | [`tropt/loss/`](tropt/loss/) | [`tropt/loss/__init__.py`](tropt/loss/__init__.py) |
| an **optimizer** | [`adding_an_optimizer.md`](docs/guides/adding_an_optimizer.md) | [`tropt/optimizer/`](tropt/optimizer/) (one optimizer per file) | [`tropt/optimizer/__init__.py`](tropt/optimizer/__init__.py) |
| a **model backend** | [`adding_a_model.md`](docs/guides/adding_a_model.md) | [`tropt/model/<backend>/`](tropt/model/) (one file per concrete class) | [`tropt/model/__init__.py`](tropt/model/__init__.py) |
| a **recipe** | [`adding_a_recipe.md`](docs/guides/adding_a_recipe.md) | [`tropt/recipe_hub/`](tropt/recipe_hub/) (one paper / family per file) | [`tropt/recipe_hub/__init__.py`](tropt/recipe_hub/__init__.py) |

For every contribution:

1. **Add tests** following the conventions and per-component checklists in [`TESTING.md`](TESTING.md); match the layout of the existing tests under [`tests/`](tests/). Cover output shape, sign conventions, mixin requirements, and known input/output pairs. Note that the end-to-end suite runs tiny CPU models with few steps, so tests assert shapes and finiteness — not loss decrease or trigger content, which are flaky at that scale.
2. **Regenerate the compatibility matrix** if you touched a model, optimizer, mixin, or loss `require_*` flag: `python docs/scripts/generate_compat_matrix.py`. Commit the regenerated [`docs/guides/compatibility_matrix.md`](docs/guides/compatibility_matrix.md).
3. **Update the relevant README row** — recipes need an entry in [`tropt/recipe_hub/README.md`](tropt/recipe_hub/README.md) (key, description, target model, required access, paper, file).
4. **Smoke-test end-to-end** on a small model (e.g. `google/gemma-3-270m-it`) before submitting.
5. **Don't bypass mixin validation.** `BaseOptimizer.__init__` rejects models missing required mixins — this is the single guarantee that an optimizer only calls methods the model implements. Add the mixin to the model rather than weakening the check.

## Recipe naming convention

Recipe entry-point function names and the `RECIPES` dict keys use the same string:

```
{method}[_{variant}][_{task}][__{paperYYYY}]
```

| Slot | Meaning | Example |
| --- | --- | --- |
| `method` | short canonical name | `gcg`, `arca`, `beast` |
| `_{variant}` | algorithmic variant of the method | `gcgp_whitebox`, `gcg_perplexity` |
| `_{task}` | distinct task application | `advdecoding_jailbreak`, `uat_prompt_injection` |
| `__{paperYYYY}` | **reproduction tag** (double underscore; first author + year, lowercased, no separator) | `gcg__zou2023` |

### Bar for inclusion

A new recipe must clear one of these bars:

- **Paper-backed reproduction.** Implements a published method faithfully (see reproduction tag rules below).
- **Genuine novelty.** A non-paper composition that demonstrably extends the hub's coverage — e.g., composing an existing optimizer with a new loss to target a setting the hub doesn't yet support.

We will close low-effort recipes that are neither (e.g. a hub entry that is essentially `gcg__zou2023` with a tweaked default and no paper or motivating result). If unsure, open a `[recipe]` issue first.

### When to use the reproduction tag

Use `__{paperYYYY}` *only* if all three hold:

1. The recipe implements the paper's algorithm step-by-step (no different optimizer, no different per-step rule, no missing core ingredient like a buffer or a momentum term).
2. The recipe's defaults match the hyperparameters the paper reports (or directly cites from the paper's reference implementation).
3. If the recipe ports the method to a different setting than the paper's main experiment (e.g. CLIP → causal-LM), the algorithm transfers cleanly and the loss function is the canonical analogue. State the port in the docstring.

If any of these fails, **omit the tag** — use the bare `{method}[_{variant}][_{task}]` form and document deviations in the docstring.

Examples (live in the repo):

| Recipe key | Why this form |
| --- | --- |
| `gcg__zou2023` | Algorithm 1 + paper hparams (B=512, k=256, T=500). |
| `gcg_perplexity` | GCG composed with `TriggerPerplexityLoss`; not in any paper. |
| `gcgp_whitebox__hayase2024` | GCG+ white-box variant from §4.1 of the QCG paper. |
| `prs` | Recipe documents explicit deviations from the paper (schedule, restarts, no judge). |
| `arca_toxic_reverse` | Reproduces the *task* of Jones et al. §4.2.1 but uses GCG instead of ARCA; algorithm differs. |

**Multiple recipes from the same paper.** When a single paper introduces multiple variants and the hub exposes them as separate functions, each carries the same paper tag, with the variant slot disambiguating: `pal__sitawarin2024`, `ral__sitawarin2024`, `gcgp_pal__sitawarin2024`.





---
[TODO is the following even relevant in lishgt of the above?]

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
