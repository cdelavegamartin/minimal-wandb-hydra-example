import pprint

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
import os



@hydra.main(version_base=None, config_path="configs/", config_name="defaults")
def run_experiment(cfg: DictConfig) -> None:

    
    pprint.pprint(cfg, indent=2)

    wandb.init(**cfg.wandb.setup, config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),)
    for epoch in range(10):
        wandb.log({"a": cfg.dummy_hyperparams.a, "b":cfg.dummy_hyperparams.b}, step=epoch)

    np.save("test.npy", np.random.randn(10, 10))
    with open(f"Output-a{cfg.dummy_hyperparams.a}-b{cfg.dummy_hyperparams.b}.txt", "w") as text_file:
        text_file.write(f"a: {cfg.dummy_hyperparams.a}, b: {cfg.dummy_hyperparams.b}")
        #  write all the hydra and wandb dirs
        text_file.write(f"\n os.getcwd(): {os.getcwd()}")
        text_file.write(f"\n hydra.core.hydra_config.HydraConfig.get().runtime.output_dir: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
        text_file.write(f"\n hydra.core.hydra_config.HydraConfig.get().runtime.cwd: {hydra.core.hydra_config.HydraConfig.get().runtime.cwd}")
        text_file.write(f"\n hydra.utils.get_original_cwd(): {hydra.utils.get_original_cwd()}")
        text_file.write(f"\n wandb.run.dir: {wandb.run.dir}")
    wandb.finish()
        

if __name__ == "__main__":
    os.environ['WANDB_START_METHOD'] = 'thread'
    run_experiment()
