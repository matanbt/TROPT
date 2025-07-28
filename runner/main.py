import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tropt.optimizer.utils.token_initializers import (
    get_printable_random_trigger,
)

# Register custom resolvers
OmegaConf.register_new_resolver("rand_tensor", lambda *size: torch.rand(size))
OmegaConf.register_new_resolver("rand_trigger", lambda t_len: get_printable_random_trigger(trigger_len=t_len))


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:

    # Instantiate the model, loss, and optimizer
    model = hydra.utils.instantiate(cfg.model)
    loss = hydra.utils.instantiate(cfg.loss)
    tracker_factory = hydra.utils.instantiate(cfg.tracker)
    tracker = tracker_factory(experiment_name=cfg.tracker.experiment_name)
    # collect additional kwargs for optimizer
    optimizer_add_kwargs = dict()
    if cfg.get("util_lm") is not None:
        util_lm = hydra.utils.instantiate(cfg.util_lm)
        optimizer_add_kwargs["util_lm"] = util_lm
    # Instantiate the optimizer
    optimizer_factory = hydra.utils.instantiate(cfg.optimizer)
    optimizer = optimizer_factory(
        model=model, loss=loss, tracker=tracker, **optimizer_add_kwargs
    )

    with tracker:
        ## Run the optimization
        # TODO support basic logic single instruction/target by repeating it
        texts = OmegaConf.to_container(cfg.texts, resolve=True)
        targets = OmegaConf.to_container(cfg.targets, resolve=True)
        result = optimizer.optimize_trigger(
            texts=texts,
            targets=targets,
            initial_trigger=cfg.initial_trigger,
        )

        # Print the results
        print("=" * 20)
        print(f"Best trigger: {result.best_trigger_str}")
        print(f"Best loss: {result.best_loss}")
        print("=" * 20)


if __name__ == "__main__":
    main()
