import argparse
import json
import pprint
import time
from pathlib import Path
from typing import Any, Callable, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from datasets.search import FaissIndex
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm.auto import tqdm

import sys

sys.path.append("/home/sharifm/students/ishayyemini/TROPT")

from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER
from tropt.loss import EmbeddingBasedLoss, SimilarityLoss
from tropt.models import EncoderBaseModel, EncoderHFModel, EncoderOpenAIModel
from tropt.optimizer import RASLITEPlusOptimizer
from tropt.optimizer.combi_optimizer import CombiOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints

# LOAD MSMARCO

# LOAD RESULTS (OR ESTIMATIONS)

DATASET_NAME = "sentence-transformers/msmarco-corpus"
SPLIT = "train"
INDEX_PATH = "../../indices/msmarco_arctic_v2.index"
INDEX_MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"

models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "Snowflake/snowflake-arctic-embed-m",
    "intfloat/e5-base-v2",
    "sentence-transformers/gtr-t5-base",
    "facebook/contriever",
    "facebook/contriever-msmarco",
    "sentence-transformers/msmarco-roberta-base-ance-firstp",
]
similarities = {
    "sentence-transformers/all-MiniLM-L6-v2": "cos_sim",
    "sentence-transformers/all-mpnet-base-v2": "cos_sim",
    "Snowflake/snowflake-arctic-embed-m": "cos_sim",
    "intfloat/e5-base-v2": "cos_sim",
    "sentence-transformers/gtr-t5-base": "cos_sim",
    "facebook/contriever": "dot",
    "facebook/contriever-msmarco": "dot",
    "sentence-transformers/msmarco-roberta-base-ance-firstp": "dot",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": "dot",
}

toxic_prefixes = []
global_corpus = None
global_queries = None

global_results: Optional[dict] = None
global_index = None
global_index_model = None

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


def get_toxic_passage(embedder_model_name: str):
    if len(toxic_prefixes) == 0:
        toxic_prefixes.extend(
            [
                row["text"]
                for row in load_dataset("toxigen/toxigen-data")["train"]
                if row["toxicity_ai"] == 5 and 50 < len(row["text"]) < 100
            ]
        )
    passage = np.random.choice(toxic_prefixes).strip()
    if embedder_model_name == "intfloat/e5-base-v2":
        passage = "passage: " + passage

    return passage


def load_data(embedder_model_name: str):
    global global_corpus, global_queries
    if global_corpus is None:
        global_corpus = load_dataset(DATASET_NAME, name="passage", split=SPLIT)
    if global_queries is None:
        global_queries = load_dataset(DATASET_NAME, name="query", split=SPLIT)

    corpus, queries = global_corpus, global_queries

    if embedder_model_name == "intfloat/e5-base-v2":
        corpus = {
            pid: {"text": ("passage: " + content["text"])}
            for pid, content in global_corpus.items()
        }
        queries = {qid: ("query: " + text) for qid, text in global_queries.items()}
    elif embedder_model_name == "Snowflake/snowflake-arctic-embed-m":
        queries = {
            qid: ("Represent this sentence for searching relevant passages:" + text)
            for qid, text in global_queries.items()
        }

    return corpus, queries.to_pandas()


def load_results(embedder_model_name: str) -> dict:
    global global_results
    if global_results:
        return global_results

    results_filename = f"msmarco-test_1.0_{embedder_model_name.split('/')[1]}_{similarities[embedder_model_name]}.json"
    try:
        local_results_path = hf_hub_download(
            repo_id="MatanBT/retrieval-datasets-similarities",
            filename=results_filename,
            repo_type="dataset",
        )
    except Exception as e:
        raise ValueError(f"Could not download results file: {e}")
    with open(local_results_path) as f:
        global_results = json.load(f)

    return global_results


def load_index() -> tuple[FaissIndex, SentenceTransformer]:
    global global_index, global_index_model
    if global_index is not None and global_index_model is not None:
        return global_index, global_index_model

    global_index = faiss.read_index(INDEX_PATH)

    global_index_model = SentenceTransformer(
        INDEX_MODEL_NAME, device=device, trust_remote_code=True
    )
    global_index_model.half()

    return global_index, global_index_model


def estimate_best_passage(
    target_text: str, calc_sim: Callable[[list[str]], Tensor], top_k: int = 50
) -> tuple[int, float]:
    index, index_model = load_index()
    corpus, queries = load_data(INDEX_MODEL_NAME)

    with torch.no_grad():
        query_index_emb = index_model.encode(
            [target_text],
            prompt_name="query",
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        ).astype("float32")

    D, I = index.search(
        query_index_emb,
    )

    best_pids = [idx + 1 for idx in I[0]]
    best_texts = [corpus[pid]["text"] for pid in best_pids]

    scores = calc_sim(best_texts)
    max_loss, max_idx = torch.max(scores, dim=0)

    best_pid = best_pids[int(max_idx.item())]
    best_sim = float(max_loss.item())

    return best_pid, best_sim


# Check similarity against:
#   info only
#   stuffing
#   TROPT.RASLITE
#   TROPT.Combi

# FOR NOW - only open source
# WRITE HISTORY and everything to compare and make graphs later!


def parse_args():
    parser = argparse.ArgumentParser(description="Test combi attack")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--out", default="combi_attack")
    return parser.parse_args()


def run_attacks(
    embedder_model_name: str, trials: int, seed: int = 42
) -> dict[str, Any]:
    best_similarities = []
    info_similarities = []
    stuffing_similarities = []

    raslite_similarities = []
    raslite_times = []
    raslite_adv = []
    raslite_losses = []

    combi_similarities = []
    combi_times = []
    combi_adv = []
    combi_losses = []

    # square_start = []
    # num_tokens = []
    # num_calls = []
    # histories = []

    corpus, queries = load_data(embedder_model_name)

    chosen_qids = np.random.choice(
        queries["qid"].to_numpy(), size=(trials,), replace=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if embedder_model_name.startswith("openai/"):
        embedder_model_name = embedder_model_name.replace("openai/", "")
        model = EncoderOpenAIModel(model_name=embedder_model_name)
    else:
        model = EncoderHFModel(model_name=embedder_model_name)
        device = model.device

    loss = SimilarityLoss()  # TODO handle dot product loss!

    for i, qid in enumerate(tqdm(chosen_qids)):
        print(f"Trial number {i + 1}")

        q = queries[queries["qid"] == qid]["text"].item()
        print(f"{qid}: {q}")

        info = get_toxic_passage(embedder_model_name)
        print(f"info: {info}")

        prefix_info = info + " " + OPTIMIZED_TRIGGER_PLACEHOLDER
        target_vector = model([q])

        inputs, _ = model.prepare_text_inputs(
            texts=[prefix_info],
            targets={loss.TARGET_KEY: target_vector},
        )

        def calc_sim(strs: str | list[str]) -> float | Tensor:
            if isinstance(strs, str):
                strs = [strs]
            losses = model.compute_loss_from_texts(
                candidate_trigger_strs=strs,
                inputs=inputs,
                loss_func=loss,
            )
            if losses.shape[0] == 1:
                return float(-losses[0])
            else:
                return -losses

        # GET BEST PASSAGE
        if embedder_model_name.startswith("openai/"):
            best_pid, best_sim = estimate_best_passage(q, calc_sim)
        else:
            results = load_results(embedder_model_name)
            best_pid = list(results[qid].keys())[0]
            best_sim = calc_sim(corpus[best_pid]["text"])

        p = corpus[best_pid]["text"]
        print(f"best passage: {best_pid}: {p}")

        print(f"Similarity between query and original best passage: {best_sim}")
        info_sim = calc_sim(info)
        print(f"Similarity between query and info passage: {info_sim}")
        stuffing_sim = calc_sim(info + " " + q.replace("query: ", ""))
        print(f"Similarity between query and stuffing passage: {stuffing_sim}")

        best_similarities.append(best_sim)
        info_similarities.append(info_sim)
        stuffing_similarities.append(stuffing_sim)

        ### RASLITE ###

        optimizer = RASLITEPlusOptimizer(
            model=model,
            util_lm=None,
            loss=loss,
            # Set parameters (combining RASLITE defaults with GASLITEPlus enhancements):
            num_steps=1500,
            token_constraints=TokenConstraints(
                disallow_non_ascii=True, disallow_special_tokens=True
            ),
            use_retokenize=False,
            # Original features:
            n_candidates=128,
            n_flip=1,
            # Plus features:
            use_random_logits=True,
            flip_pos_method="ordered",
            buffer_size=10,
            n_bulk_flips=1,
        )

        start = time.time()
        result = optimizer.optimize_trigger(
            texts=[prefix_info],
            targets={loss.TARGET_KEY: target_vector},
            initial_trigger="! " * 100,
        )
        end = time.time()

        raslite_similarities.append(-result.best_loss)
        raslite_times.append(end - start)
        raslite_adv.append(result.full_prompt[0])
        raslite_losses.append(result.losses)

        ### COMBI ###

        optimizer = CombiOptimizer(
            model=model,
            loss=loss,
            best_sim=best_sim,
        )

        start = time.time()
        result = optimizer.optimize_trigger(
            texts=[prefix_info],
            targets={loss.TARGET_KEY: target_vector},
            initial_trigger="! " * 100,
            target_text=q,
        )
        end = time.time()

        combi_similarities.append(-result.best_loss)
        combi_times.append(end - start)
        combi_adv.append(result.full_prompt[0])
        combi_losses.append(result.losses)

        print(f"raslite: {raslite_adv[-1]}")
        print(f"combi: {combi_adv[-1]}")
        print(
            f"Similarity between query and RASLITE passage: {raslite_similarities[-1]}"
        )
        print(f"Similarity between query and combi passage: {combi_similarities[-1]}")

        print("\n", flush=True)

    record = {
        "model": embedder_model_name,
        "dataset": DATASET_NAME,
        "similarity": (
            similarities[embedder_model_name]
            if embedder_model_name in similarities
            else "cosine"
        ),
        "trials": trials,
        "best_similarities": best_similarities,
        "info_similarities": info_similarities,
        "stuffing_similarities": stuffing_similarities,
        "raslite_similarities": raslite_similarities,
        "raslite_times": raslite_times,
        "raslite_adv": raslite_adv,
        "raslite_losses": raslite_losses,
        "combi_similarities": combi_similarities,
        "combi_times": combi_times,
        "combi_adv": combi_adv,
        "combi_losses": combi_losses,
        "success_rate": float(
            sum(np.array(combi_similarities) > np.array(best_similarities)) / trials
        ),
    }

    pprint.pp(record)
    return record


def main():
    args = parse_args()
    out_path = Path(args.out).with_suffix(".csv")

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    rec = run_attacks(embedder_model_name=args.model, trials=args.trials, seed=seed)
    # Write (append or create)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        combined = pd.concat([existing, pd.DataFrame([rec])], ignore_index=True)
        combined.drop_duplicates(
            subset=["model", "dataset", "trials"], keep="last"
        ).to_csv(out_path, index=False)
    else:
        pd.DataFrame([rec]).to_csv(out_path, index=False)


if __name__ == "__main__":
    main()
