# Optimizer Benchmarks (`scripts/opt-bench/`)

Reproducible benchmark pipeline for evaluating TROPT's discrete text optimizers and jailbreak enhancement tricks.

## Experiments

### Exp1 — Optimizer Sweep (`exp1.py`)
Fixes the loss (PrefillCE) and sweeps all available optimizers.
- **White-box**: gradient-based, beam-search, and continuous optimizers on HF models.
- **Black-box**: zeroth-order optimizers on API models via LiteLLM (FirstTokenNLL loss).

### Exp2 — Jailbreak Tweaks (`exp2.py`)
Fixes the optimizer (GCG) and sweeps attack tricks: loss variants, template tricks, and activation steering.
- **Single-instruction**: one trigger per (message, seed).
- **Multi-instruction**: one universal trigger optimized across all messages per seed.

Variants: `gcg_vanilla`, `gcg_attn_hijack`, `gcg_cw`, `gcg_prs_template`, `gcg_iris_target`, `gcg_steering`.

### Evaluation (`eval.py`)
Builds three CSVs from wandb runs, shared by both experiments:
- **CSV I**: run metadata and best results from wandb.
- **CSV II**: triggered generation + BLEU score.
- **CSV III**: universality evaluation via StrongReject.

### Analysis (`analyze.ipynb`)
Paper-ready plots: bar plots, box plots, optimization dynamics, and LaTeX tables.

## Quick Start

```bash
# Run everything (3 white-box models + 1 black-box)
bash scripts/opt-bench/run_all.sh

# Or run individually:
python -m scripts.opt-bench.exp1 whitebox --model-name google/gemma-2-9b-it
python -m scripts.opt-bench.exp2 single --model-name google/gemma-2-9b-it
python -m scripts.opt-bench.exp2 multi --model-name google/gemma-2-9b-it

# Evaluate
python -m scripts.opt-bench.eval build-csv-i --run-type tweakbench_single
python -m scripts.opt-bench.eval build-csv-ii --csv-i-path results/csv_i.csv --model-name google/gemma-2-9b-it
python -m scripts.opt-bench.eval build-csv-iii --csv-i-path results/csv_i.csv --model-name google/gemma-2-9b-it
```

## Configuration

Override models via environment variables:
```bash
WHITEBOX_MODELS="model/a model/b" BLACKBOX_MODEL="openai/gpt-5-nano" bash scripts/opt-bench/run_all.sh
```

Constants (seeds, message IDs, trigger length, wandb entity) are defined at the top of `exp1.py` and `exp2.py`.
