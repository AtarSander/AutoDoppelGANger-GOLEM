from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import torch
from logger import BaseLogger
from model import Model
from torch import Device, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer(ABC):
    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        device: Device,
        logger: BaseLogger,
        checkpoint_path: Path,
        log_frequency: int = 1,
        log_samples_num: int = 8,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.checkpoint_path = checkpoint_path
        self.log_frequency = log_frequency
        self.log_samples_num = log_samples_num

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
        self.global_step = 0

        for epoch in tqdm(range(epochs)):
            train_metrics = self.train_epoch(train_loader)
            validation_metrics = self.validation(val_loader)

            self.logger.log_metrics(train_metrics, step=self.global_step)
            self.logger.log_metrics(validation_metrics, step=self.global_step)

            if epoch % self.log_frequency == 0:
                with torch.no_grad:
                    samples = self.model.sample(self.log_samples_num)
                self.logger.log_images(samples, step=self.global_step)

                self.save_checkpoint(epoch)

        self.logger.finish()

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        running_metrics: Dict[str, float] = {}

        for batch in tqdm(train_loader, desc="Train", leave=False):
            batch_metrics = self.train_step(batch)
            self.aggregate_metrics(running_metrics, batch_metrics)

            self.global_step += 1

        return self.finalize_metrics(running_metrics, len(train_loader), prefix="train")

    def validation(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        running_metrics: Dict[str, float] = {}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Valid", leave=False):
                batch_metrics = self.validation_step(batch)
                self.aggregate_metrics(running_metrics, batch_metrics)

        return self.finalize_metrics(running_metrics, len(val_loader), prefix="val")

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, Tensor]: ...

    @abstractmethod
    def validation_step(self, batch: Any) -> Dict[str, Tensor]: ...

    @abstractmethod
    def save_checkpoint(self, epoch: int) -> None: ...

    def aggregate_metrics(
        self, running_metrics: Dict[str, float], batch_metrics: Dict[str, Tensor]
    ) -> None:
        for key, value in batch_metrics.items():
            running_metrics[key] = running_metrics.get(key, 0.0) + float(value.detach().item())

    def finalize_metrics(
        self, running_metrics: Dict[str, float], num_steps: int, prefix: str
    ) -> Dict[str, float]:
        if num_steps == 0:
            return {}
        return {f"{prefix}/{k}": v / num_steps for k, v in running_metrics.items()}
