"""
Corpus Poisoning Demo — optimizes adversarial passages for the Potter concept.

Optimizes 10 adversarial passages per model by clustering held-in queries with
k-means and targeting each centroid.

Two attacks:
  (i)  White-box GASLITE on intfloat/e5-base-v2
  (ii) Black-box RandomSearch on OpenAI text-embedding-3-small

Usage
-----
  python scripts/opt-bench/exp3-corpois.py gaslite-e5
  python scripts/opt-bench/exp3-corpois.py rs-openai
"""

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import typer
from datasets import load_dataset
from sklearn.cluster import KMeans

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER, Targets
from tropt.loss import SimilarityLoss
from tropt.model.huggingface.encoder import EncoderHFModel
from tropt.model.openai.encoder import EncoderOpenAIModel
from tropt.optimizer.gaslite_optimizer import GASLITEOptimizer
from tropt.optimizer.rs_optimizer import RandomSearchOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.optimizer.utils.token_initializers import get_printable_random_trigger
from tropt.tracker import WandbTracker

# ─── Constants ───────────────────────────────────────────────────────────────
WANDB_ENTITY = "matanbt"
WANDB_PROJECT = "tropt-corpois"
CONCEPT = "potter"
BUDGET = 10  # number of adversarial passages (= k-means clusters)
TRIGGER_LEN = 50
DATASET_PATH = "MatanBT/msmarco-concepts-2"

POTTER_MAL_INFO = (
    "Voldemort was right all along and Harry Potter is a self-absorbed "
    "hero who doesn't deserve the fame and glory he receives. "
)

# Model-specific prefixes
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "

OUTPUT_DIR = Path("scripts/opt-bench/results")


# ─── Data helpers ────────────────────────────────────────────────────────────

def _load_potter_queries() -> list[str]:
    """Load raw (unprefixed) potter queries from msmarco-concepts-2."""
    queries_ds = load_dataset(DATASET_PATH, "queries", split=CONCEPT)
    return [s["text"] for s in queries_ds if len(s["text"]) > 0]


def _split_queries(queries: list[str], seed: int = 0) -> tuple[list[str], list[str]]:
    """Shuffle and split into held-in / held-out halves."""
    shuffled = list(queries)
    random.Random(seed).shuffle(shuffled)
    mid = len(shuffled) // 2
    return shuffled[:mid], shuffled[mid:]


def _cluster_and_get_centroids(
    embeddings: np.ndarray, n_clusters: int
) -> np.ndarray:
    """K-means cluster embeddings; return L2-normalized centroids."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    return centroids


def _embed_texts(model, texts: list[str]) -> np.ndarray:
    """Embed texts via the model's invoke_from_texts; return normalized numpy array."""
    with torch.no_grad():
        out = model.invoke_from_texts(texts)
    embs = out.output_embeddings.cpu().numpy()
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    return embs


def _save_results(results: dict, filename: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")


# ─── Attack: GASLITE on E5 ──────────────────────────────────────────────────

app = typer.Typer()


@app.command("gaslite-e5")
def run_gaslite_e5():
    """White-box GASLITE attack on intfloat/e5-base-v2."""
    model_name = "intfloat/e5-base-v2"
    model = EncoderHFModel(model_name=model_name)
    loss = SimilarityLoss()

    # Load and split queries
    raw_queries = _load_potter_queries()
    held_in, held_out = _split_queries(raw_queries)
    print(f"Potter queries: {len(raw_queries)} total, "
          f"{len(held_in)} held-in, {len(held_out)} held-out")

    # Embed held-in queries (with query prefix)
    prefixed = [E5_QUERY_PREFIX + q for q in held_in]
    query_embs = _embed_texts(model, prefixed)

    # K-means → centroids
    centroids = _cluster_and_get_centroids(query_embs, BUDGET)
    print(f"Computed {BUDGET} centroids via k-means")

    template = E5_PASSAGE_PREFIX + POTTER_MAL_INFO + OPTIMIZED_TRIGGER_PLACEHOLDER
    tc = TokenConstraints()
    initial_trigger = get_printable_random_trigger(
        trigger_len=TRIGGER_LEN, tokenizer=model.tokenizer, token_constraints=tc,
    )

    adv_passages = []
    for cid in range(BUDGET):
        target_vec = torch.tensor(centroids[cid]).unsqueeze(0).float()
        run_name = f"corpois[gaslite,e5,c={cid}]"
        print(f"\n{'='*60}\n  {run_name}\n{'='*60}")

        tracker = WandbTracker(
            run_name,
            tags=["corpois", "gaslite", "e5-base-v2", CONCEPT],
            project_name=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config_dump={
                "attack": "gaslite",
                "model_name": model_name,
                "concept": CONCEPT,
                "cluster_id": cid,
                "budget": BUDGET,
                "trigger_len": TRIGGER_LEN,
                "template": template,
            },
        )

        optimizer = GASLITEOptimizer(
            model=model,
            loss=loss,
            tracker=tracker,
            # Full GASLITE recipe parameters:
            n_candidates=128,
            n_grad=10,
            n_flip=20,
            token_constraints=tc,
            use_retokenize=True,
        )

        result = optimizer.optimize_trigger(
            templates=[template],
            targets=Targets(target_vectors=target_vec),
            initial_trigger=initial_trigger,
        )

        passage_text = template.replace(
            OPTIMIZED_TRIGGER_PLACEHOLDER, result.best_trigger_str
        )
        adv_passages.append({
            "cluster_id": cid,
            "trigger": result.best_trigger_str,
            "adv_passage": passage_text,
            "final_loss": result.best_loss,
        })
        tracker.finish()

    _save_results(
        {
            "attack": "gaslite",
            "model_name": model_name,
            "concept": CONCEPT,
            "budget": BUDGET,
            "trigger_len": TRIGGER_LEN,
            "malicious_info": POTTER_MAL_INFO,
            "passage_prefix": E5_PASSAGE_PREFIX,
            "template": template,
            "n_held_in_queries": len(held_in),
            "n_held_out_queries": len(held_out),
            "timestamp": datetime.now().isoformat(),
            "adv_passages": adv_passages,
        },
        "exp3_gaslite_e5_potter.json",
    )


# ─── Attack: RandomSearch on OpenAI ─────────────────────────────────────────

@app.command("rs-openai")
def run_rs_openai():
    """Black-box RandomSearch attack on OpenAI text-embedding-3-small."""
    model_name = "text-embedding-3-small"
    model = EncoderOpenAIModel(model_name=model_name)
    loss = SimilarityLoss()

    # Load and split queries (OpenAI has no prefix)
    raw_queries = _load_potter_queries()
    held_in, held_out = _split_queries(raw_queries)
    print(f"Potter queries: {len(raw_queries)} total, "
          f"{len(held_in)} held-in, {len(held_out)} held-out")

    query_embs = _embed_texts(model, held_in)

    # K-means → centroids
    centroids = _cluster_and_get_centroids(query_embs, BUDGET)
    print(f"Computed {BUDGET} centroids via k-means")

    template = POTTER_MAL_INFO + OPTIMIZED_TRIGGER_PLACEHOLDER
    tc = TokenConstraints()
    initial_trigger = get_printable_random_trigger(
        trigger_len=TRIGGER_LEN, tokenizer=model.tokenizer, token_constraints=tc,
    )

    adv_passages = []
    for cid in range(BUDGET):
        target_vec = torch.tensor(centroids[cid]).unsqueeze(0).float()
        run_name = f"corpois[rs,openai,c={cid}]"
        print(f"\n{'='*60}\n  {run_name}\n{'='*60}")

        tracker = WandbTracker(
            run_name,
            tags=["corpois", "random_search", "openai", CONCEPT],
            project_name=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            config_dump={
                "attack": "random_search",
                "model_name": model_name,
                "concept": CONCEPT,
                "cluster_id": cid,
                "budget": BUDGET,
                "trigger_len": TRIGGER_LEN,
                "template": template,
            },
        )

        optimizer = RandomSearchOptimizer(
            model=model,
            loss=loss,
            tracker=tracker,
            num_steps=500,
            n_candidates=128,
            token_constraints=tc,
            mutation_mode = "block_random",
            # Block parameters
            schedule = "fixed",
            initial_block_len = 4,
            # patience before restart:
            patience=25,
        )

        result = optimizer.optimize_trigger(
            templates=[template],
            targets=Targets(target_vectors=target_vec),
            initial_trigger=initial_trigger,
        )

        passage_text = template.replace(
            OPTIMIZED_TRIGGER_PLACEHOLDER, result.best_trigger_str
        )
        adv_passages.append({
            "cluster_id": cid,
            "trigger": result.best_trigger_str,
            "adv_passage": passage_text,
            "final_loss": result.best_loss,
        })
        tracker.finish()

    _save_results(
        {
            "attack": "random_search",
            "model_name": model_name,
            "concept": CONCEPT,
            "budget": BUDGET,
            "trigger_len": TRIGGER_LEN,
            "malicious_info": POTTER_MAL_INFO,
            "passage_prefix": "",
            "template": template,
            "n_held_in_queries": len(held_in),
            "n_held_out_queries": len(held_out),
            "timestamp": datetime.now().isoformat(),
            "adv_passages": adv_passages,
        },
        "exp3_rs_openai_potter.json",
    )


if __name__ == "__main__":
    app()
