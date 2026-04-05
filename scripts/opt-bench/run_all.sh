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
MSG_IDS="0"
SEEDS="42 123 777"


WHITEBOX_MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "google/gemma-3-12b-it"
    "Qwen/Qwen3-8B"
    # ----- other models -----
    # "HuggingFaceTB/SmolLM2-135M-Instruct"  # <-- sanity check
    # "mistralai/Mistral-7B-Instruct-v0.3"  # <-- optinal
)
BLACKBOX_MODEL="openai/gpt-4o-mini"

WANDB_PROJECT="tropt-optbench"
RESULTS_DIR="scripts/opt-bench/results"

mkdir -p "$RESULTS_DIR"

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
echo "=== Exp1: White-box optimizer sweep ==="
for model in "${WHITEBOX_MODELS[@]}"; do
    echo "--- Model: $model ---"
    python scripts/opt-bench/exp1.py whitebox --model-name "$model" $MSG_ID_FLAGS $SEED_FLAGS
done

echo "=== Exp1: External NanoGCG sweep ==="
for model in "${WHITEBOX_MODELS[@]}"; do
    echo "--- Model: $model ---"
    python scripts/opt-bench/exp1.py external-nanogcg --model-name "$model" $MSG_ID_FLAGS $SEED_FLAGS
done

echo "=== Exp1: Black-box optimizer sweep ==="
python scripts/opt-bench/exp1.py blackbox --model-name "$BLACKBOX_MODEL" $MSG_ID_FLAGS $SEED_FLAGS

# ─── Exp2: Jailbreak Tweaks Benchmarks ──────────────────────────────────────
echo "=== Exp2: Single-instruction tweak sweep ==="
for model in "${WHITEBOX_MODELS[@]}"; do
    echo "--- Model: $model ---"
    python scripts/opt-bench/exp2.py single --model-name "$model" $MSG_ID_FLAGS $SEED_FLAGS
done

echo "=== Exp2: Multi-instruction tweak sweep ==="
for model in "${WHITEBOX_MODELS[@]}"; do
    echo "--- Model: $model ---"
    python scripts/opt-bench/exp2.py multi --model-name "$model" $MSG_ID_FLAGS $SEED_FLAGS
done


# ─── Exp3: Corpus Poisoning ────────────────────────────────────────────────
echo "=== Exp3: Corpus poisoning (GASLITE on E5) ==="
python scripts/opt-bench/exp3-corpois.py gaslite-e5


echo "=== Exp3: Corpus poisoning (RandomSearch on OpenAI) ==="
python scripts/opt-bench/exp3-corpois.py rs-openai

echo ">> After running Exp3, we need to now send Abed the 10 adv passages, inject MSMARCO, and test the results on all the held-{in,out} queries for measures. <<"





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
# TODO the wandb project here is different! we should fix it!!
for model in "${WHITEBOX_MODELS[@]}"; do
    short=$(echo "$model" | sed 's|.*/||')
    evaluate "enhancebench_single" "exp2_single_${short}" "$model"
done

# Exp2 multi: evaluate with each white-box model
for model in "${WHITEBOX_MODELS[@]}"; do
    short=$(echo "$model" | sed 's|.*/||')
    evaluate "enhancebench_multi" "exp2_multi_${short}" "$model"
done

echo "=== All done. Results in $RESULTS_DIR/ ==="
