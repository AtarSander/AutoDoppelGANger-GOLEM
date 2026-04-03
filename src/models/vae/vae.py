from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from beartype import beartype
from einops import rearrange, reduce
from jaxtyping import Float
from torch import Tensor

from src.model import Model


@dataclass
class EncoderConfig:
    in_channels: int
    hidden_dims: List[int]
    kernel_size: int
    stride: int
    padding: int
    encoder_out_shape: Tuple[int, int, int]


@dataclass
class DecoderConfig:
    out_channels: int
    hidden_dims: List[int]
    kernel_size: int
    stride: int
    padding: int
    output_padding: int


class VAE(Model):
    def __init__(
        self,
        latent_dim: int,
        encoder_cfg: EncoderConfig,
        decoder_cfg: DecoderConfig,
    ):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        _, H, W = self.encoder_cfg.encoder_out_shape

        self.encoder = self.build_encoder()
        self.mu, self.log_var = (
            nn.Linear(self.decoder_cfg.in_channels * H * W, self.latent_dim),
            nn.Linear(self.decoder_cfg.in_channels * H * W, self.latent_dim),
        )
        self.decoder = self.build_decoder()
        self.reconstruction_criterion = nn.BCELoss(reduction="sum")

    @beartype
    def forward(
        self, x: Float[Tensor, "b c h w"]
    ) -> Tuple[Float[Tensor, "b c h w"], Float[Tensor, "b d"], Float[Tensor, "b d"]]:
        mu, log_var = self.encode(x)

        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std

        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var

    @beartype
    def loss_dict(
        self,
        batch: Float[Tensor, "b c h w"],
        outputs: Tuple[Float[Tensor, "b c h w"], Float[Tensor, "b d"], Float[Tensor, "b d"]],
    ) -> Dict[str, Float[Tensor, "b"]]:  # noqa: F821
        x_reconstructed, mu, log_var = outputs
        reconstruction_loss = self.reconstruction_criterion(x_reconstructed, batch)
        kl_divergence = -0.5 * reduce(1 + log_var - mu**2 - log_var.exp(), "b d -> b", "sum")
        return {
            "reconstruction_loss": reconstruction_loss,
            "kl_divergence": kl_divergence,
            "loss": reconstruction_loss + kl_divergence,
        }

    @beartype
    def sample(
        self,
        num_samples: int | None = None,
        z: Float[Tensor, "n d"] | None = None,
        **kwargs,
    ) -> Float[Tensor, "n c h w"]:
        if z is None:
            z = torch.randn(num_samples, self.latent_dim)

        z = z.to(self.device)
        generated_samples = self.decode(z)
        return generated_samples.cpu()

    @beartype
    def encode(
        self, x: Float[Tensor, "b c h w"]
    ) -> Tuple[Float[Tensor, "b d"], Float[Tensor, "b d"]]:
        output = self.encoder(x)
        output = rearrange(output, "b c h w -> b (c h w)")
        mu = self.mu(output)
        log_var = self.log_var(output)
        return mu, log_var

    @beartype
    def decode(self, z: Float[Tensor, "b d"]) -> Float[Tensor, "b c h w"]:
        output = self.decoder_input(z)
        _, H, W = self.encoder_cfg.encoder_out_shape
        output = rearrange(output, "b (c h w) -> b c h w", h=H, w=W)
        output = self.decoder(output)
        return output

    def build_encoder(self) -> nn.Sequential:
        blocks = nn.Sequential()
        in_channels = self.encoder_cfg.in_channels
        for h_dim in self.encoder_cfg.hidden_dims:
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=h_dim,
                    kernel_size=self.encoder_cfg.kernel_size,
                    stride=self.encoder_cfg.stride,
                    padding=self.encoder_cfg.padding,
                ),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(),
            )
            blocks.append(block)
            in_channels = h_dim
        return blocks

    def build_decoder(self) -> nn.Sequential:
        in_channels = self.decoder_cfg.in_channels
        _, H, W = self.encoder_cfg.encoder_out_shape
        self.decoder_input = nn.Linear(self.latent_dim, in_channels * H * W)
        blocks = nn.Sequential()

        for h_dim in self.decoder_cfg.hidden_dims:
            block = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=h_dim,
                    kernel_size=self.decoder_cfg.kernel_size,
                    stride=self.decoder_cfg.stride,
                    padding=self.decoder_cfg.padding,
                    output_padding=self.decoder_cfg.output_padding,
                ),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(),
            )
            blocks.append(block)
            in_channels = h_dim

        final_block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                in_channels,
                kernel_size=self.decoder_cfg.kernel_size,
                stride=self.decoder_cfg.stride,
                padding=self.decoder_cfg.padding,
                output_padding=self.decoder_cfg.output_padding,
            ),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels,
                out_channels=self.encoder_cfg.in_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Sigmoid(),
        )
        blocks.append(final_block)
        return blocks

    def initialize_weights(self):
        self.initialize_decoder_weights()
        self.initialize_encoder_weights()

    def initialize_encoder_weights(self):
        nn.init.xavier_normal_(self.mu.weight)
        nn.init.constant_(self.mu.bias, 0)
        nn.init.xavier_normal_(self.log_var.weight)
        nn.init.constant_(self.log_var.bias, 0)
        for m in self.encoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def initialize_decoder_weights(self):
        nn.init.xavier_normal_(self.decoder_input.weight)
        nn.init.constant_(self.decoder_input.bias, 0)
        for m in self.decoder.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
