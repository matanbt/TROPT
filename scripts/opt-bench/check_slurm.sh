#!/usr/bin/env bash
# Monitor (and optionally re-launch) the exp1/exp2/exp2u slurm jobs.
#
# Job naming scheme (one slurm job = one (exp, model, seed) triple):
#   exp{N}{letter}{seed_idx}
#     N        ∈ {1, 2, 2u}        — exp1.py / exp2.py single / exp2.py multi
#     letter   ∈ {l, g, q, m}      — Llama / Gemma-3-12B / Qwen / Gemma-4-MoE
#                                    (exp2 and exp2u are pinned to {g}; see EXP2_LETTERS)
#     seed_idx ∈ {1, 2, 3}         — position in SEEDS (42, 123, 777)
# Example: exp1g1 = exp1 on Gemma-3 with seed 42; exp2ug1 = exp2 multi on Gemma-3.
#
# Expects scripts/opt-bench/eval.slurm to exist; the slurm wrapper is expected
# to end with `bash scripts/opt-bench/run_all.sh`, which reads the env vars
# EXP_FILTER / MODEL_FILTER / SEED_FILTER we pass via --export.
#
# Usage:
#   bash scripts/opt-bench/check_slurm.sh                # list missing jobs (exp1 + exp2 + exp2u)
#   bash scripts/opt-bench/check_slurm.sh --run-missing  # also sbatch them
#   bash scripts/opt-bench/check_slurm.sh --exp1         # restrict to exp1 (analogously --exp2 single, --exp2u multi)
#   bash scripts/opt-bench/check_slurm.sh --eval         # skip exp sweep; submit one eval job
#                                                        # (run_all.sh's evaluation section)
#   bash scripts/opt-bench/check_slurm.sh --eval-fast    # like --eval but skip heavy build-csv-iii
# Flags combine, e.g. `--eval --exp1` evaluates only exp1; `--exp2u --run-missing` relaunches
# only exp2 multi jobs.
#
# Useful recipes:
#   # See what's missing right now (exp1 + exp2 + exp2u):
#   bash scripts/opt-bench/check_slurm.sh
#
#   # Relaunch anything not currently queued/running:
#   bash scripts/opt-bench/check_slurm.sh --run-missing
#
#   # Only worry about exp1 (listing, and relaunching missing ones):
#   bash scripts/opt-bench/check_slurm.sh --exp1 --run-missing
#
#   # Run evaluations after the sweep finishes (full pipeline, csv-i + ii + iii):
#   bash scripts/opt-bench/check_slurm.sh --eval
#
#   # Fast eval pass (skip build-csv-iii) for exp1 only:
#   bash scripts/opt-bench/check_slurm.sh --eval-fast --exp1
set -euo pipefail

# ─── Parse args ─────────────────────────────────────────────────────────────
RUN_MISSING=0
MODE="check"   # "check" | "eval" | "eval-fast"
EXP_LIMIT=""   # "" | "1" | "2" | "2u"
for arg in "$@"; do
    case "$arg" in
        --run-missing) RUN_MISSING=1 ;;
        --eval)        MODE="eval" ;;
        --eval-fast)   MODE="eval-fast" ;;
        --exp1)        EXP_LIMIT="1" ;;
        --exp2)        EXP_LIMIT="2" ;;
        --exp2u)       EXP_LIMIT="2u" ;;
        -h|--help)     sed -n '2,39p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "ERROR: unknown arg: $arg" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_FILE="$SCRIPT_DIR/eval.slurm"

# Keep these aligned with run_all.sh's WHITEBOX_MODELS and SEEDS.
declare -A LETTER_TO_MODEL=(
    [l]="meta-llama/Llama-3.1-8B-Instruct"
    [g]="google/gemma-3-12b-it"
    [q]="Qwen/Qwen3-8B"
    [m]="google/gemma-4-26B-A4B-it"
)
EXP1_LETTERS=(l g q m)
EXP2_LETTERS=(g)    # exp2 is pinned to gemma-3-12b (see run_all.sh EXP2_MODELS)
SEEDS=(42 123 777)
EXPS=(1 2 2u)
if [[ -n "$EXP_LIMIT" ]]; then
    EXPS=("$EXP_LIMIT")
fi

# ─── Eval-only mode ─────────────────────────────────────────────────────────
# Submit a single job that skips the exp sweep (EXP_FILTER unset) and goes
# straight to run_all.sh's evaluation section. --exp1/--exp2 narrows *which*
# exp's evals run via EVAL_EXP_FILTER; --eval-fast additionally skips the
# heavy build-csv-iii step via SKIP_CSV_III=1.
if [[ "$MODE" == "eval" || "$MODE" == "eval-fast" ]]; then
    if [[ ! -f "$SLURM_FILE" ]]; then
        echo "ERROR: $SLURM_FILE not found" >&2
        exit 1
    fi
    export_args="ALL"
    [[ -n "$EXP_LIMIT" ]] && export_args="$export_args,EVAL_EXP_FILTER=$EXP_LIMIT"
    [[ "$MODE" == "eval-fast" ]] && export_args="$export_args,SKIP_CSV_III=1"
    job_name="eval${EXP_LIMIT}"
    [[ -z "$EXP_LIMIT" ]] && job_name="eval"
    [[ "$MODE" == "eval-fast" ]] && job_name="${job_name}f"
    echo "=== Submitting $MODE job (name=$job_name, exp=${EXP_LIMIT:-all}) ==="
    sbatch -J "$job_name" --export="$export_args" "$SLURM_FILE"
    exit 0
fi

# ─── Build expected job list ────────────────────────────────────────────────
EXPECTED=()
for exp in "${EXPS[@]}"; do
    case "$exp" in
        1) letters=("${EXP1_LETTERS[@]}") ;;
        2|2u) letters=("${EXP2_LETTERS[@]}") ;;
    esac
    for letter in "${letters[@]}"; do
        for i in 1 2 3; do
            EXPECTED+=("exp${exp}${letter}${i}")
        done
    done
done
# exp1 OpenAI blackbox — single standalone job (covers ALL seeds × ALL msgs).
for exp in "${EXPS[@]}"; do
    if [[ "$exp" == "1" ]]; then
        EXPECTED+=("exp1chat")
        break
    fi
done

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
if [[ $RUN_MISSING -eq 1 ]]; then
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
        # if [[ "$job" == "exp1chat" ]]; then
        #     echo "  sbatch $job  exp1 OpenAI blackbox (EXP_FILTER=1bb)"
        #     sbatch -J "$job" --export=ALL,EXP_FILTER=1bb "$SLURM_FILE"
        #     continue
        # fi
        if [[ ! "$job" =~ ^exp(1|2u|2)([lgqm])([123])$ ]]; then
            echo "  [skip] $job — unrecognized format"
            continue
        fi
        exp_num="${BASH_REMATCH[1]}"
        letter="${BASH_REMATCH[2]}"
        idx="${BASH_REMATCH[3]}"
        model="${LETTER_TO_MODEL[$letter]}"
        seed="${SEEDS[$((idx - 1))]}"

        # skip seed idx != 1
        # if [[ "$idx" != "1" ]]; then
        #     echo "  [skip] $job — idx=$idx (only idx=1 is run to save resources)"
        #     continue
        # fi

        # skip exp != 1 unless --exp{1,2} was explicitly passed (then EXP_LIMIT
        # already constrains which exps are queued — honor the user's intent).
        if [[ -z "$EXP_LIMIT" && "$exp_num" != "1" ]]; then
            echo "  [skip] $job — exp_num=$exp_num (default skips exp2/exp2u; pass --exp2 or --exp2u to override)"
            continue
        fi

        # skip non gemma model:
        # if [[ "$model" != "google/gemma-4-26B-A4B-it" ]]; then
        #     echo "  [skip] $job — model=$model (only Gemma is run to save resources)"
        #     continue
        # fi

        # skip non llama model:
        # if [[ "$model" != "meta-llama/Llama-3.1-8B-Instruct" ]]; then
        #     echo "  [skip] $job — model=$model (only Llama is run to save resources)"
        #     continue
        # fi

        # Gemma-4 MoE (m = gemma-4-26B-A4B-it) is much larger than the rest —
        # route it to the H100 partition/node (the default eval.slurm pins
        # nodelist to gpu-sharifm nodes, which would reject the h100 partition).
        # exp2u (multi-message tweak sweep) is also routed to H100 for headroom.
        extra_sbatch_args=()
        if [[ "$letter" == "m" || "$exp_num" == "2u" ]]; then
            extra_sbatch_args+=(
                --partition=gpu-h100-killable
                --nodelist=n-102,t-100
                --time=1200
            )
        fi

        echo "  sbatch $job  exp=$exp_num model=$model seed=$seed ${extra_sbatch_args[*]}"
        sbatch \
            -J "$job" \
            "${extra_sbatch_args[@]}" \
            --export=ALL,EXP_FILTER="$exp_num",MODEL_FILTER="$model",SEED_FILTER="$seed" \
            "$SLURM_FILE"
    done
fi
