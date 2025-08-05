import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, solution_dim):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.solution_dim = solution_dim
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, solution_dim * 2) # mean and log_std
        )

    def forward(self, z):
        params = self.fc(z)
        mean, log_std = params.chunk(2, dim=1)
        log_std = torch.clamp(log_std, -5, 2) # Clamp log_std for stability
        std = torch.exp(log_std) + 1e-6 # Add epsilon for numerical stability
        return mean, std

    def sample(self, num_samples, device):
        """Helper to generate samples"""
        self.eval() # Set to evaluation mode for sampling
        with torch.no_grad():
            z = torch.randn(num_samples, self.noise_dim, device=device)
            mean, std = self(z)
            # Reparameterization trick for sampling
            epsilon = torch.randn_like(std)
            samples = mean + std * epsilon
        return samples
