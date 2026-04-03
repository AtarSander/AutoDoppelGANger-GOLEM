from abc import ABC, abstractmethod
from typing import Dict

import wandb
from einops import rearrange
from torch import Tensor
from torchvision.utils import make_grid


class BaseLogger(ABC):
    @abstractmethod
    def log_metrics(self, metrics, step: int) -> None: ...

    def log_config(self, config: Dict) -> None: ...

    def log_images(self, samples: Tensor, step: int) -> None: ...

    @abstractmethod
    def finish(self): ...


class WandbLogger(BaseLogger):
    def __init__(self, project: str, config: Dict):
        self.run = wandb.init(project=project, config=config)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        self.run.log(metrics, step=step)

    def log_images(self, samples: Tensor, step: int, nrow: int = 4) -> None:
        grid = make_grid(samples, nrow=nrow)
        grid = rearrange(grid, "c h w -> h w c").cpu()

        self.run.log({"samples_grid": wandb.Image(grid)}, step=step)

    def finish(self):
        self.run.finish()
