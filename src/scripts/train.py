from omegaconf import DictConfig
import hydra
from loguru import logger
from torch.optim import Optimizer

from src.core.logger import WandbLogger
from src.core.model import Model
from src.core.trainer import Trainer


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def train(cfg: DictConfig):
    model: Model = hydra.utils.instantiate(cfg.model)
    optimizer: Optimizer = hydra.utils.instantiate(cfg.optimizer)
    trainer = hydra.utils.instantiate(cfg.trainer)


if __name__ == "__main__":
    train()
