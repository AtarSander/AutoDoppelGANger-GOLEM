from abc import ABC, abstractmethod
from typing import Dict

import torch.nn as nn
from torch import Device, Tensor


class Model(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def sample(self, num_samples: int, **kwargs) -> Tensor: ...

    @abstractmethod
    def loss_dict(self, batch: Tensor) -> Dict[str, Tensor]: ...

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def device(self) -> Device:
        return (self.parameters()).device
