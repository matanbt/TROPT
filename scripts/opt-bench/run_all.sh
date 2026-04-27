#!/usr/bin/env bash
# Run all optimizer benchmark experiments and evaluations.
#
# Usage:
#   bash scripts/opt-bench/run_all.sh
#
# Override models via environment variables:
#   WHITEBOX_MODELS="model1 model2" bash scripts/opt-bench/run_all.sh
set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
# Scale control: bump this list when ready for full runs.
#   Smoke test:  MSG_IDS="0"
#   Medium:      MSG_IDS="0 1 2 3 4"
#   Full:        MSG_IDS="0 1 2 3 4 5 6 7 8 9"
MSG_IDS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14"
SEEDS="42 123 777"


WHITEBOX_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-12b-it"
    "Qwen/Qwen3-8B"
    "google/gemma-4-26B-A4B-it"
    # ----- other models -----
    # "HuggingFaceTB/SmolLM2-135M-Instruct"  # <-- sanity check
    # "mistralai/Mistral-7B-Instruct-v0.3"  # <-- optinal
)
# exp2 is pinned to a single model (cheaper sweep for tweak ablations).
EXP2_MODELS=(
    "google/gemma-3-12b-it"
)
BLACKBOX_MODEL="openai/gpt-4o-mini"

# When set (e.g. "openai/gpt-4o-mini"), build-csv-iii uses strongreject_rubric with
# this OpenAI judge instead of strongreject_finetuned. Empty = use the local
# fine-tuned Gemma judge. Requires OPENAI_API_KEY in the environment.
JUDGE_OPENAI_MODEL=""

WANDB_PROJECT="tropt-optbench"
WANDB_PROJECT_EXP2="tropt-enhancebench"  # exp2.py uses a separate project
RESULTS_DIR="scripts/opt-bench/results"

mkdir -p "$RESULTS_DIR"

# ─── Slurm-filter support ───────────────────────────────────────────────────
# When launched per-job by check_slurm.sh, these env vars narrow the sweep to a
# single (exp, model, seed) tuple. Empty means "run everything" (legacy behavior).
EXP_FILTER="${EXP_FILTER:-}"      # "" | "1" | "2" | "2u" | "1bb" (2 = exp2 single, 2u = exp2 multi, 1bb = exp1 OpenAI blackbox only)
MODEL_FILTER="${MODEL_FILTER:-}"  # "" | substring of a WHITEBOX_MODELS entry
SEED_FILTER="${SEED_FILTER:-}"    # "" | single seed, e.g. "42"
# Eval-only flow (only consulted when EXP_FILTER is empty, since non-empty
# EXP_FILTER short-circuits out before the evaluation section).
EVAL_EXP_FILTER="${EVAL_EXP_FILTER:-}"  # "" | "1" | "2" | "2u" — restrict eval section to one experiment (2 = exp2 single, 2u = exp2 multi)
SKIP_CSV_III="${SKIP_CSV_III:-0}"       # "1" to skip the heavy build-csv-iii step in evaluate()

if [[ -n "$MODEL_FILTER" ]]; then
    _filtered=()
    for _m in "${WHITEBOX_MODELS[@]}"; do
        if [[ "$_m" == *"$MODEL_FILTER"* ]]; then _filtered+=("$_m"); fi
    done
    WHITEBOX_MODELS=("${_filtered[@]}")
    echo "[filter] WHITEBOX_MODELS -> ${WHITEBOX_MODELS[*]}"
    _filtered=()
    for _m in "${EXP2_MODELS[@]}"; do
        if [[ "$_m" == *"$MODEL_FILTER"* ]]; then _filtered+=("$_m"); fi
    done
    EXP2_MODELS=("${_filtered[@]}")
    echo "[filter] EXP2_MODELS -> ${EXP2_MODELS[*]}"
fi
if [[ -n "$SEED_FILTER" ]]; then
    SEEDS="$SEED_FILTER"
    echo "[filter] SEEDS -> $SEEDS"
fi
if [[ -n "$EXP_FILTER" ]]; then
    echo "[filter] EXP_FILTER -> $EXP_FILTER (exp3 + evals disabled)"
fi

# Build repeated --msg-ids flags from the space-separated list
MSG_ID_FLAGS=""
for mid in $MSG_IDS; do
    MSG_ID_FLAGS="$MSG_ID_FLAGS --msg-ids $mid"
done

SEED_FLAGS=""
for s in $SEEDS; do
    SEED_FLAGS="$SEED_FLAGS --seeds $s"
done
echo "Message IDs: $MSG_IDS  |  Seeds: $SEEDS"

# ─── Exp1: Optimizer Benchmarks ─────────────────────────────────────────────
if [[ "$EXP_FILTER" == "1" ]]; then
    echo "=== Exp1: White-box optimizer sweep ==="
    for model in "${WHITEBOX_MODELS[@]}"; do
        echo "--- Model: $model ---"
        python scripts/opt-bench/exp1.py whitebox --model-name "$model" $MSG_ID_FLAGS $SEED_FLAGS
    done

    ## [Disabled -- only for reprod exp]
    # echo "=== Exp1: External NanoGCG sweep ==="
    # for model in "${WHITEBOX_MODELS[@]}"; do
    #     echo "--- Model: $model ---"
    #     python scripts/opt-bench/exp1.py external-nanogcg --model-name "$model" $MSG_ID_FLAGS $SEED_FLAGS
    # done

fi

# Exp1 black-box (OpenAI) has no whitebox-model/seed axis — it's a single job
# of its own (slurm name: exp1chat), dispatched via EXP_FILTER=1bb.
if [[ "$EXP_FILTER" == "1bb" ]]; then
    echo "=== Exp1: Black-box optimizer sweep ==="
    python scripts/opt-bench/exp1.py blackbox --model-name "$BLACKBOX_MODEL" $MSG_ID_FLAGS $SEED_FLAGS
fi

# ─── Exp2: Jailbreak Tweaks Benchmarks ──────────────────────────────────────
# Split into two filters: "2" = single-instruction sweep; "2u" = multi-instruction
# sweep. Kept separate so each can be scheduled / re-launched / evaluated on its
# own (multi is heavier and routed to H100 by check_slurm.sh).
if [[ "$EXP_FILTER" == "2" ]]; then
    echo "=== Exp2: Single-instruction tweak sweep ==="
    for model in "${EXP2_MODELS[@]}"; do
        echo "--- Model: $model ---"
        python scripts/opt-bench/exp2.py single --model-name "$model" $MSG_ID_FLAGS $SEED_FLAGS
    done
fi

if [[ "$EXP_FILTER" == "2u" ]]; then
    echo "=== Exp2: Multi-instruction tweak sweep ==="
    for model in "${EXP2_MODELS[@]}"; do
        echo "--- Model: $model ---"
        python scripts/opt-bench/exp2.py multi --model-name "$model" $MSG_ID_FLAGS $SEED_FLAGS
    done
fi


# ─── Evaluations ────────────────────────────────────────────────────────────
# Each (experiment, run_type) gets its own CSV set to avoid collisions.
# Evals are manual / out of scope for per-job slurm flow.
if [[ -n "$EXP_FILTER" ]]; then
    echo "=== Skipping evaluations (EXP_FILTER=$EXP_FILTER) ==="
    echo "=== Done. ==="
    exit 0
fi

evaluate() {
    local run_type="$1"
    local prefix="$2"
    local eval_model="$3"
    local use_litellm="${4:-false}"
    local wandb_project="${5:-$WANDB_PROJECT}"

    local csv_i="$RESULTS_DIR/${prefix}_csv_i.csv"
    local csv_ii="$RESULTS_DIR/${prefix}_csv_ii.csv"
    local csv_iii="$RESULTS_DIR/${prefix}_csv_iii.csv"

    echo "--- Eval: $prefix (run_type=$run_type, model=$eval_model, project=$wandb_project) ---"

    python -m scripts.opt-bench.eval build-csv-i \
        --wandb-project "$wandb_project" \
        --run-type "$run_type" \
        --model-name "$eval_model" \
        --output-path "$csv_i"

    local litellm_flag=""
    if [ "$use_litellm" = "true" ]; then
        litellm_flag="--use-litellm"
    fi

    python -m scripts.opt-bench.eval build-csv-ii \
        --csv-i-path "$csv_i" \
        --model-name "$eval_model" \
        --output-path "$csv_ii" \
        $litellm_flag

    if [[ "$SKIP_CSV_III" == "1" ]]; then
        echo "  [skip] build-csv-iii for $prefix (SKIP_CSV_III=1)"
    else
        local judge_flag=""
        if [[ -n "$JUDGE_OPENAI_MODEL" ]]; then
            judge_flag="--judge-openai-model $JUDGE_OPENAI_MODEL"
        fi
        python -m scripts.opt-bench.eval build-csv-iii \
            --csv-i-path "$csv_i" \
            --model-name "$eval_model" \
            --output-path "$csv_iii" \
            $litellm_flag \
            $judge_flag
    fi
}

WHITEBOX_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-12b-it"
    "Qwen/Qwen3-8B"
    "google/gemma-4-26B-A4B-it"
    # ----- other models -----
    # "HuggingFaceTB/SmolLM2-135M-Instruct"  # <-- sanity check
    # "mistralai/Mistral-7B-Instruct-v0.3"  # <-- optinal
)
EXP2_MODELS=(
    "google/gemma-3-12b-it"
)

# Exp1 white-box: evaluate with each white-box model
if [[ -z "$EVAL_EXP_FILTER" || "$EVAL_EXP_FILTER" == "1" ]]; then
    for model in "${WHITEBOX_MODELS[@]}"; do
        short=$(echo "$model" | sed 's|.*/||')
        evaluate "optbench_whitebox" "exp1_wb_${short}" "$model"
    done
fi

# Exp1 black-box: evaluate with the black-box model via LiteLLM
# short_bb=$(echo "$BLACKBOX_MODEL" | sed 's|.*/||')
# evaluate "optbench_blackbox" "exp1_bb_${short_bb}" "$BLACKBOX_MODEL" "true"

# Exp2 single + multi: evaluate with each white-box model (separate wandb project)
if [[ -z "$EVAL_EXP_FILTER" || "$EVAL_EXP_FILTER" == "2" ]]; then
    for model in "${EXP2_MODELS[@]}"; do
        short=$(echo "$model" | sed 's|.*/||')
        evaluate "enhancebench_single" "exp2_single_${short}" "$model" "false" "$WANDB_PROJECT_EXP2"
    done
fi

if [[ -z "$EVAL_EXP_FILTER" || "$EVAL_EXP_FILTER" == "2u" ]]; then
    for model in "${EXP2_MODELS[@]}"; do
        short=$(echo "$model" | sed 's|.*/||')
        evaluate "enhancebench_multi" "exp2_multi_${short}" "$model" "false" "$WANDB_PROJECT_EXP2"
    done
fi

echo "=== All done. Results in $RESULTS_DIR/ ==="
