import torch.nn as nn
import torch

class BetaVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_dims, device):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.encoder = nn.Sequential(
            *self.build_encoder(in_channels)
        )
        self.mu, self.log_var = self.build_distr_params()
        self.decoder_input = self.build_decoder_input()
        self.decoder = nn.Sequential(
            *self.build_decoder(),
        )
        if device.type == "cuda":
            self.cuda()

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparametrize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var

    def encode(self, x):
        output = self.encoder(x)
        output = torch.flatten(output, start_dim=1)
        mu = self.mu(output)
        log_var = self.log_var(output)
        return mu, log_var

    def reparametrize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z):
        output = self.decoder_input(z)
        output = output.view(-1, 512, 2, 2)
        output = self.decoder(output)
        return output

    def build_encoder(self, in_channels):
        blocks = []
        for h_dim in self.hidden_dims:
            blocks.append(self.encoder_block(in_channels, h_dim))
            in_channels = h_dim
        return blocks

    def build_distr_params(self):
        return (nn.Linear(self.hidden_dims[-1]*4, self.latent_dim),
               nn.Linear(self.hidden_dims[-1]*4, self.latent_dim))

    def encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def build_decoder_input(self):
        return nn.Linear(self.latent_dim, self.hidden_dims[-1] * 4)

    def build_decoder(self):
        blocks = []
        hidden_dims = self.hidden_dims[::-1]
        for i in range(1, len(hidden_dims)):
            in_channels = hidden_dims[i - 1]
            out_channels = hidden_dims[i]
            blocks.append(self.decoder_block(in_channels, out_channels))
        blocks.append(self.decoder_final(out_channels))
        return blocks

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def decoder_final(self, in_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

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

