import os
import sys

import hydra
from omegaconf import OmegaConf

# Add the root of the project to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from runner import main
from tropt.models.huggingface.encoder import EncoderHFModel
from tropt.models.huggingface.lm import LMHFModel
from tropt.optimizer.beast_optimizer import BEASTOptimizer
from tropt.optimizer.gaslite_optimizer import GASLITEOptimizer
from tropt.optimizer.gasliteplus_optimizer import GASLITEPlusOptimizer

## uncomment to enable debug logging
# logging.basicConfig(level=logging.DEBUG)

def run_sanity_checks():
    config_dir = os.path.join("runner", "configs", "experiments")
    exp_names = [f.split('.yaml')[0] for f in os.listdir(config_dir) if f.endswith('.yaml')]
    exp_names = [f"test_runs/{name}" for name in exp_names]  # enable for quick test runs

    for exp_name in exp_names:
        # print(f"Running sanity check for experiment: {exp_name}")
        # if exp_name != 'gcg': continue
        if exp_name != 'test_runs/gcg': continue
        # if exp_name != 'test_runs/gasliteplus': continue

        ## disable Wandb:
        os.environ["WANDB_MODE"] = "disabled"

        print(f"===== Running sanity check for: {exp_name} =====")

        with hydra.initialize(version_base=None, config_path="../runner/configs"):
            cfg = hydra.compose(config_name="default", overrides=[f"+experiments={exp_name}"])
            main(cfg)
            print(f"===== Sanity check for {exp_name}: PASSED =====")
        print("\n" * 4)

if __name__ == "__main__":
    run_sanity_checks()
