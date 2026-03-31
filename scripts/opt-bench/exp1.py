"""
Optimizer benchmark pipeline for TROPT.

All optimizers are instantiated directly under the same PrefillCELoss and
the same model, isolating optimizer algorithm quality from attack-recipe choices.

Usage
-----
  python scripts/opt-bench/eval.py whitebox --model-name google/gemma-2-2b-it
  python scripts/opt-bench/eval.py build-csv-i
  python scripts/opt-bench/eval.py build-csv-ii --model-name google/gemma-2-2b-it
  python scripts/opt-bench/eval.py build-csv-iii --model-name google/gemma-2-2b-it
"""
import tropt.optimizer.utils.token_initializers
import os
from dataclasses import dataclass
from typing import Callable, List

import pandas as pd
import torch
import typer
import wandb
from transformers import pipeline

from scripts.attack_evaluate.evaluate_jailbreakness import evaluate_triggers
from tropt.common import OPTIMIZED_TRIGGER_PLACEHOLDER, Targets
from tropt.loss import PrefillCELoss
from tropt.model.huggingface.lm import LMHFModel
from tropt.optimizer import (
    ARCAOptimizer,
    AutoPromptOptimizer,
    BaseOptimizer,
    BeamSearchOptimizer,
    GASLITEOptimizer,
    GASLITEPlusOptimizer,
    GBDAOptimizer,
    GCGOptimizer,
    GCGPlusOptimizer,
    HotFlipOptimizer,
    RandomSearchOptimizer,
)
from tropt.optimizer.pal_optimizer import PALOptimizer
from tropt.optimizer.pez_optimizer import PEZOptimizer
from tropt.optimizer.qcg_optimizer import QCGOptimizer
from tropt.optimizer.rasliteplus_optimizer import RASLITEPlusOptimizer
from tropt.optimizer.soft_optimizer import SoftPromptOptimizer
from tropt.optimizer.utils.token_constraints import TokenConstraints
from tropt.tracker import WandbTracker

# ─── Constants ───────────────────────────────────────────────────────────────
WANDB_ENTITY = "matanbt"           # set to your wandb entity
WANDB_PROJECT = "tropt-optbench"
SEEDS = [42, 123, 777]
MSG_IDS = list(range(5))          # first 5 ClearHarm messages
TRIGGER_LEN = 20
CLEARHARM_PATH = "scripts/attack_evaluate/clearharm.csv"
UTIL_MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"   # utility LM for AdvDecoding
_RUN_TYPE = "optbench_whitebox"

_LOSS = PrefillCELoss()
_TC = TokenConstraints()


# ─── Optimizer Registry ──────────────────────────────────────────────────────
@dataclass
class OptimizerConfig:
    name: str
    # Signature: (model, tracker, seed, util_lm=None) -> BaseOptimizer
    # util_lm is only populated when needs_util_lm=True; other factories may ignore it.
    optimizer_factory: Callable[..., BaseOptimizer]
    is_soft: bool = False         # True for continuous optimizers (lower-bound reference)
    needs_util_lm: bool = False   # Requires a separate utility LM loaded by the runner


WHITEBOX_OPTIMIZER_CONFIGS: list[OptimizerConfig] = [
    # ── Gradient-based discrete optimizers ──────────────────────────────────
    OptimizerConfig("gcg",
        lambda model, tracker, seed, **_: GCGOptimizer(
            model=model, loss=_LOSS, tracker=tracker, seed=seed,
            num_steps=500, n_candidates=512, sample_topk=256, sample_n_replace=1,
            token_constraints=_TC, use_retokenize=True,
        )),
    # OptimizerConfig("gcgplus_grad",   # gradient-based candidate selection
    #     lambda model, tracker, seed, **_: GCGPlusOptimizer(
    #         model=model, loss=_LOSS, proxy_model=model, tracker=tracker, seed=seed,
    #         num_steps=500, candidate_selection="gradient",
    #         n_candidates=512, sample_topk=256, sample_n_replace=(1, 1),
    #         candidate_oversample_factor=1.1,
    #         token_constraints=_TC, use_retokenize=True,
    #     )),
    # OptimizerConfig("gcgplus_grad_momentum",   # gradient-based candidate selection
    #     lambda model, tracker, seed, **_: GCGPlusOptimizer(
    #         model=model, loss=_LOSS, proxy_model=model, tracker=tracker, seed=seed,
    #         num_steps=500, candidate_selection="gradient",
    #         n_candidates=512, sample_topk=256, sample_n_replace=(1, 1),
    #         momentum=0.9,
    #         candidate_oversample_factor=1.1,
    #         token_constraints=_TC, use_retokenize=True,
    #     )),
    OptimizerConfig("gcgplus_rand",   # random candidate selection (ablation vs gcgplus_grad)
        lambda model, tracker, seed, **_: GCGPlusOptimizer(
            model=model, loss=_LOSS, proxy_model=model, tracker=tracker, seed=seed,
            num_steps=500, candidate_selection="random",
            n_candidates=512, sample_topk=256, sample_n_replace=(1, 1),
            candidate_oversample_factor=1.1,
            token_constraints=_TC, use_retokenize=True,
        )),
    OptimizerConfig("gaslite",
        lambda model, tracker, seed, **_: GASLITEOptimizer(
            model=model, loss=_LOSS, tracker=tracker, seed=seed,
            num_steps=100, n_grad=10, n_flip=7, n_candidates=256,
            token_constraints=_TC, use_retokenize=True,
        )),
    # OptimizerConfig("gasliteplus",
    #     lambda model, tracker, seed, **_: GASLITEPlusOptimizer(
    #         model=model, loss=_LOSS, tracker=tracker, seed=seed,
    #         num_steps=100, n_grad=10, n_flip=7, n_bulk_flips=3,
    #         buffer_size=10, flip_pos_method="ordered", n_candidates=256,
    #         token_constraints=_TC, use_retokenize=True,
    #     )),
    # OptimizerConfig("raslite",       # self-proxy: target model supplies both logits & text loss
    #     lambda model, tracker, seed, **_: RASLITEPlusOptimizer(
    #         model=model, loss=_LOSS, tracker=tracker, seed=seed, use_random_logits=True,
    #         util_model=model, num_steps=100, n_flip=7, n_bulk_flips=3,
    #         buffer_size=10, flip_pos_method="ordered", n_candidates=256,
    #         token_constraints=_TC,
    #     )),
    OptimizerConfig("hotflip",
        lambda model, tracker, seed, **_: HotFlipOptimizer(
            model=model, loss=_LOSS, tracker=tracker, seed=seed,
            num_steps=500, token_constraints=_TC, use_retokenize=True,
        )),
    OptimizerConfig("autoprompt",
        lambda model, tracker, seed, **_: AutoPromptOptimizer(
            model=model, loss=_LOSS, tracker=tracker, seed=seed,
            num_steps=500, n_candidates=512, sample_topk=256,
            token_constraints=_TC, 
            use_retokenize=False,  # no retok in the paper
        )),
    OptimizerConfig("arca",
        lambda model, tracker, seed, **_: ARCAOptimizer(
            model=model, loss=_LOSS, tracker=tracker, seed=seed,
            num_steps=500, n_candidates=512, sample_topk=256, n_grad_avg=32,
            token_constraints=_TC, 
            use_retokenize=False,  # no retok in the paper
        )),
    OptimizerConfig("gbda",
        lambda model, tracker, seed, **_: GBDAOptimizer(
            model=model, loss=_LOSS, tracker=tracker, seed=seed,
            num_steps=500,
            n_grad_samples=10,
            learning_rate=0.3,
            initial_coeff=15.0,
            temp_start=1.0,
            temp_end=1.0,  # No temp annealing in original paper
            n_final_gumbel_samples=100,
            gd_optimizer=torch.optim.Adam,
            use_lr_schedule=False,  # No LR decay in original paper
        )),
    OptimizerConfig("gbda+",   # GBDA with SoftGCG hyperparameters (gradual schedule, random init, grad clip)
        lambda model, tracker, seed, **_: GBDAOptimizer(
            model=model, loss=_LOSS, tracker=tracker, seed=seed,
            num_steps=2000,
            n_grad_samples=1,
            learning_rate=0.1,
            temp_schedule="gradual",
            n_final_gumbel_samples=0,
            gd_optimizer=torch.optim.Adam,
            use_lr_schedule=False,
            grad_clip_norm=1.0,
            init_mode="random",
            init_noise_scale=2.0,
        )),
    OptimizerConfig("pal",   # PAL: proxy-guided with gradient candidate selection
        lambda model, tracker, seed, **_: PALOptimizer(
            model=model, loss=_LOSS, proxy_model=model, tracker=tracker, seed=seed,
            candidate_selection="gradient", num_steps=500,
            n_candidates=128, sample_topk=256, n_candidates_after_proxy_filter=32,
            sample_n_replace=1,
            candidate_oversample_factor=1.1, 
            token_constraints=_TC,
        )),
    OptimizerConfig("ral",   # RAL variant: random candidates, no proxy filtering
        lambda model, tracker, seed, **_: PALOptimizer(
            model=model, loss=_LOSS, proxy_model=model, tracker=tracker, seed=seed,
            candidate_selection="random", num_steps=500,
            n_candidates=32, sample_topk=256, n_candidates_after_proxy_filter=None,
            sample_n_replace=1,
            candidate_oversample_factor=1.1, 
            token_constraints=_TC,
        )),
    OptimizerConfig("qcg",        # buffer-based query attack; proxy=self (whitebox)
        lambda model, tracker, seed, **_: QCGOptimizer(
            model=model, loss=_LOSS, proxy_model=model, tracker=tracker, seed=seed,
            num_steps=500, n_proxy_candidates=8192, n_target_candidates=32,
            buffer_size=128, candidate_oversample_factor=1.1,
            token_constraints=_TC,
        )),
    OptimizerConfig("random_search",
        lambda model, tracker, seed, **_: RandomSearchOptimizer(
            model=model, loss=_LOSS, tracker=tracker, seed=seed,
            num_steps=500, n_candidates=128, token_constraints=_TC,
        )),
    # ── Beam-search / decoding-based optimizers ──────────────────────────────
    OptimizerConfig("beast",    # BeamSearchOptimizer; util_lm=None → uses target model
        lambda model, tracker, seed, **_: BeamSearchOptimizer(
            model=model, loss=_LOSS, tracker=tracker, seed=seed,
            util_lm=None, 
            num_steps=TRIGGER_LEN,  # each step appends one token
            beam_size=15, branching_factor=15,
            top_k=None,  # Paper uses full distribution multinomial sampling
            token_constraints=_TC,
        )),
    OptimizerConfig("adv_decoding",  # BeamSearchOptimizer + SmolLM2 utility LM
        lambda model, tracker, seed, util_lm=None, **_: BeamSearchOptimizer(
            model=model, loss=_LOSS, tracker=tracker, seed=seed,
            util_lm=util_lm,
            util_lm_prefix="Write a sentence with a lot of triggers. {{OPTIMIZED_TRIGGER}}",
            num_steps=TRIGGER_LEN,  # each step appends one token
            beam_size=96, # we increase this bc it's cheap
            branching_factor=10, top_k=10,
            temperature=1.0, token_constraints=_TC,
        ),
        needs_util_lm=True),
    OptimizerConfig("pez",       # optimizes continuous embeddings, projects to discrete each step
        lambda model, tracker, seed, **_: PEZOptimizer(
            model=model, loss=_LOSS, tracker=tracker, seed=seed,
            num_steps=3000, learning_rate=0.1, weight_decay=0.1,
        )),
    # ── Soft/continuous optimizers — lower-bound references ─────────────────
    OptimizerConfig("soft_prompt",
        lambda model, tracker, seed, **_: SoftPromptOptimizer(
            model=model, loss=_LOSS, tracker=tracker, seed=seed,
            num_steps=100, learning_rate=0.001,
        ),
        is_soft=True),
]

_OPT_BY_NAME = {cfg.name: cfg for cfg in WHITEBOX_OPTIMIZER_CONFIGS}
app = typer.Typer()

# TOOD random initial trigger -- from VALID ids!

# ─── Helpers ─────────────────────────────────────────────────────────────────
def _model_short(model_name: str) -> str:
    return model_name.split("/")[-1]


def _run_name(opt_name: str, model_name: str, msg_id: int, seed: int) -> str:
    return f"optbench[{opt_name},{_model_short(model_name)},m={msg_id},s={seed}]"


def _finished_run_names() -> set[str]:
    api = wandb.Api()
    return {r.name for r in api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}", filters={"state": "finished"})}


# ─── Commands ────────────────────────────────────────────────────────────────
@app.command()
def whitebox(
    model_name: str = typer.Option("google/gemma-2-2b-it", help="HuggingFace model identifier"),
    msg_ids: List[int] = typer.Option(MSG_IDS, help="ClearHarm message IDs"),
    seeds: List[int] = typer.Option(SEEDS, help="Random seeds"),
    optimizers: List[str] = typer.Option(
        [c.name for c in WHITEBOX_OPTIMIZER_CONFIGS],
        help="Optimizer names to run",
    ),
    skip_existing: bool = typer.Option(True, help="Skip already-finished wandb runs"),
):
    """Run whitebox optimizers on a HuggingFace model, all under PrefillCELoss."""
    selected = [_OPT_BY_NAME[n] for n in optimizers]

    finished: set[str] = _finished_run_names() if skip_existing else set()
    print(f"Skipping {len(finished)} already-finished runs.")

    df = pd.read_csv(CLEARHARM_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LMHFModel(
        model_name=model_name,
        device=device,
        use_prefix_cache=False,
        dtype="bfloat16",
    )
    model.set_flop_counting("manual")

    # Load utility LM once if any selected optimizer needs it
    util_lm = None
    if any(cfg.needs_util_lm for cfg in selected):
        print(f"Loading utility LM: {UTIL_MODEL}")
        util_lm = LMHFModel(model_name=UTIL_MODEL, device=device,
                             use_prefix_cache=False, dtype="bfloat16")

    for msg_id in msg_ids:
        row = df.iloc[msg_id]
        instruction: str = row["message_template"]
        target: str = row["target_response_prefix"]

        for cfg in selected:
            for seed in seeds:
                run_name = _run_name(cfg.name, model_name, msg_id, seed)

                if skip_existing and run_name in finished:
                    print(f"  skip  {run_name}")
                    continue
                print(f"  run   {run_name}")

                tracker = WandbTracker(
                    run_name,
                    tags=[_RUN_TYPE, cfg.name],
                    project_name=WANDB_PROJECT,
                    entity=WANDB_ENTITY,
                    config_dump={
                        "run_type": _RUN_TYPE,
                        "model_name": model_name,
                        "optimizer_name": cfg.name,
                        "msg_id": msg_id,
                        "seed": seed,
                        "optimized_instruction": instruction,
                        "optimized_target": target,
                        "loss_name": "PrefillCE",
                        "is_soft": cfg.is_soft,
                    },
                )

                torch.manual_seed(seed)
                model.reset_usage_stats()
                initial_trigger = tropt.optimizer.utils.token_initializers.get_printable_random_trigger(
                    trigger_len=TRIGGER_LEN, tokenizer=model.tokenizer,
                    blacklist_ids=_TC.get_blacklist_ids(model.tokenizer)
                )
                optimizer = cfg.optimizer_factory(model, tracker, seed, util_lm=util_lm)
                optimizer.optimize_trigger(
                    templates=[instruction],
                    targets=Targets(target_response_strs=[target]),
                    initial_trigger=initial_trigger,
                )

                usage = model.get_usage_stats()
                wandb.run.summary.update({
                    "final/total_flops": usage.get("total_flops"),
                    "final/total_tokens": usage.get("total_input_tokens"),
                })
                tracker.finish()



if __name__ == "__main__":
    app()
