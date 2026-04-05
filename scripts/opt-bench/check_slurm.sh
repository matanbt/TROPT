#!/usr/bin/env bash
# Monitor (and optionally re-launch) the exp1/exp2 slurm jobs.
#
# Job naming scheme (one slurm job = one (exp, model, seed) triple):
#   exp{N}{letter}{seed_idx}
#     N        ∈ {1, 2}            — exp1.py / exp2.py
#     letter   ∈ {l, g, q}         — Llama / Gemma / Qwen
#     seed_idx ∈ {1, 2, 3}         — position in SEEDS (42, 123, 777)
# Example: exp1g1 = exp1 on Gemma with seed 42.
#
# Expects scripts/opt-bench/eval.slurm to exist; the slurm wrapper is expected
# to end with `bash scripts/opt-bench/run_all.sh`, which reads the env vars
# EXP_FILTER / MODEL_FILTER / SEED_FILTER we pass via --export.
#
# Usage:
#   bash scripts/opt-bench/check_slurm.sh                # list missing jobs
#   bash scripts/opt-bench/check_slurm.sh --run-missing  # also sbatch them
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_FILE="$SCRIPT_DIR/eval.slurm"

# Keep these aligned with run_all.sh's WHITEBOX_MODELS and SEEDS.
declare -A LETTER_TO_MODEL=(
    [l]="meta-llama/Llama-3.1-8B-Instruct"
    [g]="google/gemma-3-12b-it"
    [q]="Qwen/Qwen3-8B"
)
LETTERS=(l g q)
SEEDS=(42 123 777)
EXPS=(1 2)

# ─── Build expected job list ────────────────────────────────────────────────
EXPECTED=()
for exp in "${EXPS[@]}"; do
    for letter in "${LETTERS[@]}"; do
        for i in 1 2 3; do
            EXPECTED+=("exp${exp}${letter}${i}")
        done
    done
done
# exp1 OpenAI blackbox — single standalone job (covers ALL seeds × ALL msgs).
EXPECTED+=("exp1chat")

# ─── Fetch current squeue job names ─────────────────────────────────────────
if ! command -v squeue >/dev/null 2>&1; then
    echo "ERROR: squeue not found on PATH; are you on a slurm head node?" >&2
    exit 1
fi
CURRENT=$(squeue --me --noheader --format="%j" 2>/dev/null || true)

MISSING=()
for job in "${EXPECTED[@]}"; do
    if ! grep -qx "$job" <<<"$CURRENT"; then
        MISSING+=("$job")
    fi
done

n_current=$(printf '%s\n' "$CURRENT" | grep -c . || true)
echo "=== Expected: ${#EXPECTED[@]}   queued/running (mine): ${n_current}   missing: ${#MISSING[@]} ==="
echo "--- currently running/queued (mine) ---"
printf '%s\n' "$CURRENT" | sed 's/^/  /'
echo "--- missing ---"
for j in "${MISSING[@]}"; do
    echo "  $j"
done

# ─── Optionally relaunch ────────────────────────────────────────────────────
if [[ "${1:-}" == "--run-missing" ]]; then
    if [[ ${#MISSING[@]} -eq 0 ]]; then
        echo "Nothing to relaunch."
        exit 0
    fi
    if [[ ! -f "$SLURM_FILE" ]]; then
        echo "ERROR: $SLURM_FILE not found" >&2
        exit 1
    fi
    echo
    echo "=== Relaunching ${#MISSING[@]} missing jobs via sbatch ==="
    for job in "${MISSING[@]}"; do
        if [[ "$job" == "exp1chat" ]]; then
            echo "  sbatch $job  exp1 OpenAI blackbox (EXP_FILTER=1bb)"
            sbatch -J "$job" --export=ALL,EXP_FILTER=1bb "$SLURM_FILE"
            continue
        fi
        if [[ ! "$job" =~ ^exp([12])([lgq])([123])$ ]]; then
            echo "  [skip] $job — unrecognized format"
            continue
        fi
        exp_num="${BASH_REMATCH[1]}"
        letter="${BASH_REMATCH[2]}"
        idx="${BASH_REMATCH[3]}"
        model="${LETTER_TO_MODEL[$letter]}"
        seed="${SEEDS[$((idx - 1))]}"
        echo "  sbatch $job  exp=$exp_num model=$model seed=$seed"
        sbatch \
            -J "$job" \
            --export=ALL,EXP_FILTER="$exp_num",MODEL_FILTER="$model",SEED_FILTER="$seed" \
            "$SLURM_FILE"
    done
fi
