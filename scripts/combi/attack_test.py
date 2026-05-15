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
from datasets import load_dataset
from datasets.search import FaissIndex
from huggingface_hub import hf_hub_download
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
from torch import Tensor
from tqdm.auto import tqdm

import sys
import os

# Add the root of the project to the python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from tropt.tracker import BaseTracker
from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER, Targets
from tropt.loss import SimilarityLoss
from tropt.model.huggingface.encoder import EncoderHFModel
from tropt.model.openai.encoder import EncoderOpenAIModel
from tropt.optimizer import RASLITEPlusOptimizer
from tropt.optimizer.combi_optimizer import CombiOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints

# LOAD MSMARCO

# LOAD RESULTS (OR ESTIMATIONS)

DATASET_NAME = "sentence-transformers/msmarco-corpus"
DEV_SPLIT_DATASET = "mteb/msmarco"
SPLIT = "train"
INDEX_PATH = "indices/msmarco_arctic_v2.index"
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
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
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

toxic_prefixes: list[str] = []
global_corpus: Optional[DataFrame] = None
global_queries: Optional[DataFrame] = None

global_results: Optional[dict] = None
global_index: Optional[FaissIndex] = None
global_index_model: Optional[SentenceTransformer] = None

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


class ModelDataset:
    def __init__(self, df: DataFrame, kind: str, embedder_model_name: str):
        self.df = df
        self.kind = "queries" if kind == "queries" else "corpus"
        self.embedder_model_name = embedder_model_name

    def __getitem__(self, key: int) -> str:
        prefix = ""
        if self.embedder_model_name == "intfloat/e5-base-v2":
            if self.kind == "corpus":
                prefix = "passage: "
            else:
                prefix = "query: "
        elif self.embedder_model_name == "Snowflake/snowflake-arctic-embed-m":
            if self.kind == "queries":
                prefix = "Represent this sentence for searching relevant passages: "
        item_key = "qid" if self.kind == "queries" else "pid"
        return prefix + self.df[self.df[item_key] == key]["text"].item()

    def keys(self) -> list[int]:
        item_key = "qid" if self.kind == "queries" else "pid"
        return self.df[item_key].to_list()


class TokenTracker(BaseTracker):
    def _init(self, config: Optional[dict] = None):
        pass

    def __init__(self, initial_tokens: int):
        super().__init__("nothing")
        self.initial_tokens = initial_tokens
        self.tokens = []

    def _log(self, data: dict):
        if "loss" in data and "total_tokens" in data:
            self.tokens.append(data["total_tokens"] - self.initial_tokens)

    def log(self, data: dict):
        self._log(data)

    def get_tokens(self):
        return self.tokens

    def _finish(self, summary: Optional[dict] = None):
        pass


def get_toxic_passages(embedder_model_name: str, k: int = 1) -> list[str]:
    if len(toxic_prefixes) == 0:
        toxic_prefixes.extend(
            [
                row["text"]
                for row in load_dataset("toxigen/toxigen-data")["train"]
                if row["toxicity_ai"] == 5 and 50 < len(row["text"]) < 100
            ]
        )
    choices = np.random.choice(toxic_prefixes, size=(k,))
    prefix = "passage: " if embedder_model_name == "intfloat/e5-base-v2" else ""
    passages = [prefix + p.strip() for p in choices]

    return passages


def load_data(embedder_model_name: str) -> tuple[ModelDataset, ModelDataset]:
    global global_corpus, global_queries

    if global_corpus is None:
        global_corpus = load_dataset(
            DATASET_NAME, name="passage", split=SPLIT
        ).to_pandas()

    if global_queries is None:
        # We have cached results only for dev, not the whole "train" split
        global_queries = load_dataset(
            DATASET_NAME, name="query", split=SPLIT
        ).to_pandas()
        dev_dataset = load_dataset(DEV_SPLIT_DATASET, name="default", split="dev")
        dev_qids = [int(x["query-id"]) for x in dev_dataset]
        global_queries = global_queries[global_queries["qid"].isin(dev_qids)]

    return (
        ModelDataset(global_corpus, "corpus", embedder_model_name),
        ModelDataset(global_queries, "queries", embedder_model_name),
    )


def load_results(embedder_model_name: str) -> dict[str, dict[str, float]]:
    global global_results
    if global_results:
        return global_results

    try:
        results_filename = f"msmarco-test_1.0_{embedder_model_name.split('/')[1]}_{similarities[embedder_model_name]}.json"
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
    corpus, _ = load_data(INDEX_MODEL_NAME)

    with torch.no_grad():
        query_index_emb = index_model.encode(
            [target_text],
            prompt_name="query",
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=device,
        ).astype("float32")

    _, indices = index.search(query_index_emb, k=top_k)

    best_pids = [idx + 1 for idx in indices[0]]
    best_texts = [corpus[pid] for pid in best_pids]

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
    parser.add_argument("--raslite", action="store_true")
    return parser.parse_args()


def run_attacks(embedder_model_name: str, trials: int, raslite: bool) -> dict[str, Any]:
    best_pids = []
    info_strs = []

    best_similarities = []
    info_similarities = []
    stuffing_similarities = []

    raslite_similarities = []
    raslite_times = []
    raslite_adv = []
    raslite_losses = []
    raslite_tokens = []

    combi_similarities = []
    combi_times = []
    combi_adv = []
    combi_losses = []
    combi_tokens = []

    # square_start = []
    # num_tokens = []
    # num_calls = []
    # histories = []

    corpus, queries = load_data(embedder_model_name)

    chosen_qids = np.random.choice(
        queries.keys(), size=(trials,), replace=False
    ).tolist()

    info_strs = get_toxic_passages(embedder_model_name, trials)

    if embedder_model_name.startswith("openai/"):
        model = EncoderOpenAIModel(
            model_name=embedder_model_name.removeprefix("openai/")
        )
    else:
        model = EncoderHFModel(model_name=embedder_model_name)

    if (
        embedder_model_name in similarities
        and similarities[embedder_model_name] == "dot"
    ):
        # loss = DotProductLoss() # TODO readd this?
        loss = SimilarityLoss()
    else:
        loss = SimilarityLoss()

    for i, qid in enumerate(tqdm(chosen_qids)):
        print(f"Trial number {i + 1}")

        q = queries[qid]
        print(f"{qid}: {q}")

        info = info_strs[i]
        print(f"info: {info}")

        prefix_info = info + " " + OPTIMIZED_TRIGGER_PLACEHOLDER
        target_vector = model([q])

        model.set_inputs_from_texts(
            templates=[prefix_info], targets=Targets(target_vectors=target_vector)
        )

        def calc_sim(strs: str | list[str]) -> float | Tensor:
            if isinstance(strs, str):
                strs = [strs]
            losses = model.compute_loss_from_texts(
                candidate_trigger_strs=strs,
                loss_func=loss,
            )
            if losses.shape[0] == 1:
                return float(-losses[0])
            else:
                return -losses

        # GET BEST PASSAGE
        try:
            results = load_results(embedder_model_name)
            best_pid = int(list(results[str(qid)].keys())[0])
            best_sim = calc_sim(corpus[best_pid])
        except ValueError:
            best_pid, best_sim = estimate_best_passage(q, calc_sim)

        best_pids.append(best_pid)

        p = corpus[best_pid]
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
        if raslite:
            raslite_tracker = TokenTracker(model.get_usage_stats()["total_tokens"])

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
                # Early stopping:
                early_stopping_loss=-best_sim,
                tracker=raslite_tracker,
            )

            start = time.time()
            result = optimizer.optimize_trigger(
                templates=[prefix_info],
                targets=Targets(target_vectors=target_vector),
                initial_trigger="! " * 100,
            )
            end = time.time()

            raslite_similarities.append(-result.best_loss)
            raslite_times.append(end - start)
            raslite_adv.append(result.best_trigger_str)
            raslite_losses.append(result.losses)
            raslite_tokens.append(raslite_tracker.get_tokens())

        ### COMBI ###
        combi_tracker = TokenTracker(model.get_usage_stats()["total_tokens"])

        optimizer = CombiOptimizer(
            model=model,
            loss=loss,
            best_sim=best_sim,
            tracker=combi_tracker,
        )

        start = time.time()
        result = optimizer.optimize_trigger(
            templates=[prefix_info],
            targets=Targets(target_vectors=target_vector),
            initial_trigger="! " * 100,
            target_text=q,
        )
        end = time.time()

        combi_similarities.append(-result.best_loss)
        combi_times.append(end - start)
        combi_adv.append(result.best_trigger_str)
        combi_losses.append(result.losses)
        combi_tokens.append(combi_tracker.get_tokens())

        if raslite:
            print(f"raslite: {raslite_adv[-1]}")
        print(f"combi: {combi_adv[-1]}")
        if raslite:
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
        "qids": chosen_qids,
        "best_pids": best_pids,
        "info_strs": info_strs,
        "best_similarities": best_similarities,
        "info_similarities": info_similarities,
        "stuffing_similarities": stuffing_similarities,
        "raslite_similarities": raslite_similarities,
        "raslite_times": raslite_times,
        "raslite_adv": raslite_adv,
        "raslite_losses": raslite_losses,
        "raslite_tokens": raslite_tokens,
        "combi_similarities": combi_similarities,
        "combi_times": combi_times,
        "combi_adv": combi_adv,
        "combi_losses": combi_losses,
        "combi_tokens": combi_tokens,
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

    rec = run_attacks(
        embedder_model_name=args.model, trials=args.trials, raslite=args.raslite
    )
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
