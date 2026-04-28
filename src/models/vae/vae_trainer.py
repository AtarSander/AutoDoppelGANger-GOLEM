from pathlib import Path
from typing import Dict

import torch
from jaxtyping import Float
from torch import Tensor

from src.trainer import Trainer
from src.models.vae.vae import VAE


class VAETrainer(Trainer):
    def __init__(self, model: VAE, *args, **kwargs):
        super().__init__(model, *args, **kwargs)

    def train_step(self, batch: Float[Tensor, "b c h w"]) -> Dict[str, Float[Tensor, "b"]]:  # noqa: F821
        outputs = self.model(batch)
        loss_dict = self.model.loss_dict(batch, outputs)
        loss_dict["loss"].backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss_dict

    def validation_step(self, batch: Float[Tensor, "b c h w"]) -> Dict[str, Float[Tensor, "b"]]:  # noqa: F821
        outputs = self.model(batch)
        loss_dict = self.model.loss_dict(batch, outputs)

        return loss_dict

    def save_checkpoint(self, epoch: int) -> None:
        filepath = self.checkpoint_path / f"vae_checkpoint_epoch{epoch}.pth"
        torch.save(self.model, filepath)

    def load_checkpoint(self, filename: Path | str) -> None:
        filepath = self.checkpoint_path / filename
        self.model.load_state_dict(torch.load(filepath))
