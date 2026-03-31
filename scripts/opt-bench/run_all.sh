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
MSG_IDS="${MSG_IDS:-0}"

WHITEBOX_MODELS=(
    ${WHITEBOX_MODELS:-
        "meta-llama/Llama-3.1-8B-Instruct"
        "google/gemma-2-9b-it"
        "mistralai/Mistral-7B-Instruct-v0.3"
    }
)
BLACKBOX_MODEL="${BLACKBOX_MODEL:-openai/gpt-4o-mini}"

WANDB_PROJECT="tropt-optbench"
RESULTS_DIR="scripts/opt-bench/results"

mkdir -p "$RESULTS_DIR"

# Build repeated --msg-ids flags from the space-separated list
MSG_ID_FLAGS=""
for mid in $MSG_IDS; do
    MSG_ID_FLAGS="$MSG_ID_FLAGS --msg-ids $mid"
done
echo "Message IDs: $MSG_IDS"

# ─── Exp1: Optimizer Benchmarks ─────────────────────────────────────────────
echo "=== Exp1: White-box optimizer sweep ==="
for model in "${WHITEBOX_MODELS[@]}"; do
    echo "--- Model: $model ---"
    python -m scripts.opt-bench.exp1 whitebox --model-name "$model" $MSG_ID_FLAGS
done

echo "=== Exp1: Black-box optimizer sweep ==="
python -m scripts.opt-bench.exp1 blackbox --model-name "$BLACKBOX_MODEL" $MSG_ID_FLAGS

# ─── Exp2: Jailbreak Tweaks Benchmarks ──────────────────────────────────────
echo "=== Exp2: Single-instruction tweak sweep ==="
for model in "${WHITEBOX_MODELS[@]}"; do
    echo "--- Model: $model ---"
    python -m scripts.opt-bench.exp2 single --model-name "$model" $MSG_ID_FLAGS
done

echo "=== Exp2: Multi-instruction tweak sweep ==="
for model in "${WHITEBOX_MODELS[@]}"; do
    echo "--- Model: $model ---"
    python -m scripts.opt-bench.exp2 multi --model-name "$model" $MSG_ID_FLAGS
done

# ─── Evaluations ────────────────────────────────────────────────────────────
# Each (experiment, run_type) gets its own CSV set to avoid collisions.

evaluate() {
    local run_type="$1"
    local prefix="$2"
    local eval_model="$3"
    local use_litellm="${4:-false}"

    local csv_i="$RESULTS_DIR/${prefix}_csv_i.csv"
    local csv_ii="$RESULTS_DIR/${prefix}_csv_ii.csv"
    local csv_iii="$RESULTS_DIR/${prefix}_csv_iii.csv"

    echo "--- Eval: $prefix (run_type=$run_type, model=$eval_model) ---"

    python -m scripts.opt-bench.eval build-csv-i \
        --wandb-project "$WANDB_PROJECT" \
        --run-type "$run_type" \
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

    python -m scripts.opt-bench.eval build-csv-iii \
        --csv-i-path "$csv_i" \
        --model-name "$eval_model" \
        --output-path "$csv_iii" \
        $litellm_flag
}

# Exp1 white-box: evaluate with each white-box model
for model in "${WHITEBOX_MODELS[@]}"; do
    short=$(echo "$model" | sed 's|.*/||')
    evaluate "optbench_whitebox" "exp1_wb_${short}" "$model"
done

# Exp1 black-box: evaluate with the black-box model via LiteLLM
short_bb=$(echo "$BLACKBOX_MODEL" | sed 's|.*/||')
evaluate "optbench_blackbox" "exp1_bb_${short_bb}" "$BLACKBOX_MODEL" "true"

# Exp2 single: evaluate with each white-box model
for model in "${WHITEBOX_MODELS[@]}"; do
    short=$(echo "$model" | sed 's|.*/||')
    evaluate "tweakbench_single" "exp2_single_${short}" "$model"
done

# Exp2 multi: evaluate with each white-box model
for model in "${WHITEBOX_MODELS[@]}"; do
    short=$(echo "$model" | sed 's|.*/||')
    evaluate "tweakbench_multi" "exp2_multi_${short}" "$model"
done

echo "=== All done. Results in $RESULTS_DIR/ ==="
