# Contributing to TROPT

TROPT enables a wide range of contributions: from adding new recipes of published work (to the Recipe Hub), to adding a useful, new loss or optimizer, or additional model integrations, to resolving bugs or adding new features to the package.

While the [guides](https://www.tropt.dev/guides/index.html) in our docs should be helpful in *implementing* new TROPT components (e.g., new recipe, loss, optimizer, etc.), they do not refer to the *integration* of new components within the package's code, which we also touch upon below.

TROPT aims to be a growing library and a research hub. You are encouraged to contribute your own research, or implementations of existing works, under the TROPT framework. 


---

## Opening Issues

To report a **bug**, open an issue with `[bug]` in the title. Include an informative description of the bug, the full logs of the run, the environment you run on, and provide means of reproducing it (e.g., a minimal reproduction script with a lightweight model). Clearly articulate the expected behavior vs. what happens.

You may also open issues to request/suggest adding a **new optimizer or loss**, a **new model** backend, a **new recipe** to the Recipe Hub, or a general **feature** proposal; please use an appropriate title prefix (`[model]`, `[loss]`, `[recipe]`, `[feature]`, ...). Please ensure the motivation for the inclusion of this addition in TROPT is clear; e.g., in the case of adding a new component/recipe, clearly cite the relevant paper and/or use case driving your suggestion.


## Opening PR

### 1. Developer Setup & Checks

Make sure that you are developing in TROPT's common dev env and run the required checks before opening the PR.

**Setup.**
```bash
git clone https://github.com/matanbt/tropt.git
cd tropt
uv sync --all-extras    # dev install with all optional extras
```


**Required checks**


Before opening a PR, make sure these pass locally:

```bash
uv run ruff check                       # lint        (enforced in CI)
uv run ruff format --check              # formatting   (local-only)
uv run ty check                         # type-check   (local-only)
uv run pytest                           # tests       (enforced in CI)
```

CI enforces `ruff check` and `pytest` on every PR; `ruff format --check` and `ty check` are local-only but expected to pass.


### 2. Adhere to TROPT Convention

In your implementation it is important that you'd adhere to existing convention followed by TROPT. Specifically, if you are adding a new component (model, loss, recipe, etc.), you must refer to the section on component integration into TROPT below.

**Tests.** If your implementation includes general additions to TROPT's core logic (e.g., modifying HuggingFace's backend to support a new feature), consider backing the change with a **test**.

**Docs.** If your addition makes existing documentation (whether it's a guide or a docstring within the repo) stale, please update it to reflect your change.

For further readings, we provide additional material (for you, or your favorite coding agent), under:
- [`DESIGN.md`](DESIGN.md) — full design philosophy (modularity, backend vs. frontend, why the contracts are the way they are).
- [`docs/guides/`](docs/guides/) — step-by-step per-component walkthroughs.
- **Agent files.**
    - [`CLAUDE.md`](CLAUDE.md) — fast-path orientation: the four components, mixin contract, `ModelInput`/`ModelOutput`/`Targets`, setup-then-compute.
    - [`skills/tropt/SKILL.md`](skills/tropt/SKILL.md) — task-routed pointers and cross-cutting pitfalls (mixin mismatches, thinking-model alignment, multi-model OOM, etc.).

### 3. PR workflow

After implementing your addition to TROPT, and running the essential checks, open a PR as follows:
1. Fork → create a topic branch off `main`, preferably with an informative name (e.g. `feature/my-loss`, `fix/gcg-retokenize`).
2. Open a PR against `main`. The description should state *what* and *why*, link the issue, and call out any breaking changes (preferably none) or new dependencies.


---

The following details how to integrate your custom components (e.g., loss, optimizer, recipes) into the package.

## Adding a component

| Adding | Read first | Module goes in | Export from |
| --- | --- | --- | --- |
| a **loss** | [`adding_a_loss.md`](docs/guides/adding_a_loss.md) | [`tropt/loss/`](tropt/loss/) | [`tropt/loss/__init__.py`](tropt/loss/__init__.py) |
| an **optimizer** | [`adding_an_optimizer.md`](docs/guides/adding_an_optimizer.md) | [`tropt/optimizer/`](tropt/optimizer/) (one optimizer per file) | [`tropt/optimizer/__init__.py`](tropt/optimizer/__init__.py) |
| a **model backend** | [`adding_a_model.md`](docs/guides/adding_a_model.md) | [`tropt/model/<backend>/`](tropt/model/) (one file per concrete class) | [`tropt/model/__init__.py`](tropt/model/__init__.py) |
| a **recipe** | [`adding_a_recipe.md`](docs/guides/adding_a_recipe.md) | [`tropt/recipe_hub/`](tropt/recipe_hub/) (one paper / family per file) | [`tropt/recipe_hub/__init__.py`](tropt/recipe_hub/__init__.py) |


Integrating a new component/recipe into TROPT should be fairly easy. While the implementation of the component/recipe as a standalone requires following a guide (under the _Read First_ column), to add this implementation to the package you need to add it as a module to the relevant location (_Module goes in_) and register it in the `__init__.py` file. 

Some additions (such as recipes) require reflecting them in the docs (i.e., updating `recipe_hub/README.md`). Some additions also have implicit / explicit repo conventions, for instance: in losses we aim for inheritance by category, and in the Recipe Hub we include a naming convention (read below).

While this should cover the general flow of component additions, it is always a best practice to follow an existing component and how it is integrated in TROPT (e.g., if you add an optimizer, you can follow `GCGOptimizer` and its integration in the repo).

## Adding a new recipe

**Consider Recipe Importance.** To avoid cluttering TROPT's Recipe Hub with countless recipes (as the space of possible recipes is effectively endless), we currently set the bar for new recipes to be ones that either reproduce the methodology of a published paper, or propose a novel, well-motivated use case that currently does not exist in TROPT. 

**Reproducibility.** If you are implementing a recipe that refers to existing research, it must be implemented as close to the original implementation as possible, including adhering to its parameters. It is preferable that such implementations will be backed with an empirical evaluation comparing your implementation with the original implementation (if such exists). This would allow researchers using the library to reliably use TROPT's implementation in research and future evaluations.

**Recipe Naming Convention.** TROPT sets a naming convention for the Recipe Hub, to help organize the recipes and direct (some of them) to existing papers.
The naming refers to the function that implements the recipe and its registration under `recipe_hub/__init__.py`. 

The convention is structured as:
```
{method}[_{variant}][_{task}][__{paperYYYY}]
```
with each slot interpreted as:
| Slot | Meaning | Example |
| --- | --- | --- |
| `method` | short canonical name | `gcg`, `arca`, `beast` |
| `_{variant}` | algorithmic variant of the method | `gcgp_whitebox`, `gcg_perplexity` |
| `_{task}` | distinct task application | `advdecoding_jailbreak`, `uat_prompt_injection` |
| `__{paperYYYY}` | **reproduction tag** (double underscore; first author + year, lowercased, no separator) | `gcg__zou2023` |


For example `gcg__zou2023` reproduces the GCG LLM jailbreak (Zou et al. 2023) with its _exact_ parameters from the original paper. `gcg_perplexity`, on the other hand, implements a variant of GCG that does not appear in the original paper, and combines a perplexity term with the original recipe.


