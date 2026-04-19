# ─── Exp3: Corpus Poisoning ────────────────────────────────────────────────
# Exp3 + evals are out of scope for the per-job slurm flow; only run in full mode.
# echo "=== Exp3: Corpus poisoning (GASLITE on E5) ==="
# python scripts/opt-bench/exp3-corpois.py gaslite-e5

# echo "=== Exp3: Corpus poisoning (RandomSearch on OpenAI) ==="
# python scripts/opt-bench/exp3-corpois.py rs-openai

echo "=== Exp3: Prompt Injection ==="
python scripts/opt-bench/exp3-pinj.py

echo "=== Exp3: Toxicity Auditing ==="
python scripts/opt-bench/exp3-toxic.py

echo ">> After running all exp3. <<"
