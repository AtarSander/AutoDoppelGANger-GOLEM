import os
import glob
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
from src.beta_vae import BetaVAE
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class BetaVAETrainer:
    def __init__(self, in_channels, latent_dim, hidden_dims, device, log_dir, weights_dir, checkpoint_interval=20):
        self.model = BetaVAE(in_channels, latent_dim, hidden_dims, device)
        self.device = device
        self.log_dir = log_dir
        self.weights_dir = weights_dir
        self.checkpoint_interval = checkpoint_interval

    def train(self, dataset, num_epochs, batch_size, beta, learning_rate, weights_filename=None):
        self.load_data(dataset, batch_size)
        self.model.initialize_weights()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.beta = beta
        self.reconstruction_loss = nn.BCELoss(reduction="sum")
        for epoch in range(num_epochs):
            for batch_idx, (x, _) in enumerate(self.loaded_data):
                x = (x + 1) / 2
                x = x.to(self.device)
                x_reconstructed, mu, log_var = self.model.forward(x)

                total_loss, reconstruction_loss, kl_divergence = self.calculate_losses(x_reconstructed, x, mu, log_var)
                self.backpropagation(total_loss)
                if batch_idx % 100 == 0:
                    self.print_training_stats(num_epochs, epoch, batch_idx,
                                              len(self.loaded_data), total_loss, reconstruction_loss, kl_divergence.mean())
            if weights_filename and epoch > 0 and epoch % self.checkpoint_interval == 0:
                self.checkpoint_weights(weights_filename, epoch)
        if weights_filename:
            self.save_model_weights(weights_filename, num_epochs-1)

    def load_data(self, dataset, batch_size):
        self.loaded_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def calculate_losses(self, x_reconstructed, x, mu, log_var):
        reconstruction_loss = self.reconstruction_loss(x_reconstructed, x)
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim = 1)
        total_loss = reconstruction_loss + self.beta * kl_divergence.mean()
        return total_loss, reconstruction_loss, kl_divergence

    def backpropagation(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def generate_samples(self, num_samples, z=None):
        self.model.eval()
        if z is None:
            z = torch.randn(num_samples, self.model.latent_dim).to(self.device)
        with torch.no_grad():
            generated_samples = self.model.decode(z)
        self.model.train()
        return generated_samples.cpu()

    def generate_disentangled_samples(self, base_vector, num_variations=10, dim_range=(-3, 3)):
        self.model.eval()
        values = np.linspace(dim_range[0], dim_range[1], num_variations)
        dims_to_vary = np.random.choice(self.model.latent_dim, 10, replace=False)
        generated_images = []
        for dim in dims_to_vary:
            for val in values:
                varied_vector = base_vector.clone()
                varied_vector[:, dim] = val
                with torch.no_grad():
                    imgs = self.model.decode(varied_vector)
                for img in imgs:
                    img = img.cpu().numpy().transpose(1, 2, 0)  # Correctly transpose each image
                    generated_images.append(img)
        self.model.train()
        return generated_images

    def print_training_stats(self, num_epochs, epoch, batch_idx, dataset_size, loss, reconstruction_loss, kl_divergence):
        print(
              f"EPOCH: [{(epoch+1):2d}/{num_epochs:2d}], Batch [{batch_idx:3d} / {dataset_size:3d}] \
                   Loss: {loss:.6f}, Reconstruction Loss: {reconstruction_loss:.6f}, KL Divergence: {kl_divergence:.6f}"
             )

    def setup_tensorboard(self):
        self.writer_real = SummaryWriter(self.log_dir+"/real")
        self.writer_fake = SummaryWriter(self.log_dir+"/fake")
        fixed_noise = torch.randn(32, self.noise_dim, 1, 1).to(self.device)
        return fixed_noise

    def checkpoint_weights(self, name, epoch):
        previous_checkpoint = self.weights_dir+name+'_'+str(max(0, epoch-20))+'.pth'
        if os.path.isfile(previous_checkpoint):
            os.remove(previous_checkpoint)
        self.save_model_weights(name+'_'+str(epoch))

    def save_model_weights(self, name, epoch=0):
        subract_val = epoch % self.checkpoint_interval
        last_checkpoint = self.weights_dir+name+'_'+str(epoch-subract_val)+'.pth'
        if os.path.isfile(last_checkpoint):
            os.remove(last_checkpoint)
        torch.save(self.model.state_dict(), self.weights_dir+name+'.pth')

    def load_model_weights(self, name):
        self.model.load_state_dict(torch.load(self.weights_dir+name, map_location=torch.device("cpu")))

    def tensor_board_grid(self, writer_real, writer_fake, real, step):
        self.model.eval()
        z = torch.randn(32, self.model.latent_dim).to(self.device)
        with torch.no_grad():
            generated_samples = self.model.decode(z)
        self.model.train()
        img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
        img_grid_fake = torchvision.utils.make_grid(generated_samples[:32], normalize=True)

        writer_real.add_image("Real", img_grid_real, global_step=step)
        writer_fake.add_image("Fake", img_grid_fake, global_step=step)

