import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPM(nn.Module):
    def __init__(self, initial_prior=None, image_size=64, channels=3, timesteps=1000, device=None):
        super(DDPM, self).__init__()
        self.initial_prior = initial_prior
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.device = device

        self.beta = torch.linspace(0.0001, 0.02, timesteps).to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_cumprod = torch.cumprod(self.alpha, 0).to(self.device)
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0).to(self.device)
        
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, channels, kernel_size=3, padding=1)
        ).to(self.device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
        return (self.alpha_cumprod[t] ** 0.5) * x_start + ((1 - self.alpha_cumprod[t]) ** 0.5) * noise

    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
        
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.net(x_noisy)
        return F.mse_loss(predicted_noise, noise)

    def forward(self, x_start):
        t = torch.randint(0, self.timesteps, (x_start.size(0),), device=self.device).long()
        return self.p_losses(x_start, t)
    
    def generate(self, sample):
        for i in reversed(range(self.timesteps)):
            t = torch.full((sample.size(0),), i, device=self.device, dtype=torch.long)
            predicted_noise = self.net(sample)
            alpha = self.alpha[t][:, None, None, None]
            alpha_cumprod = self.alpha_cumprod[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]

            sample = (sample - (1 - alpha) / (1 - alpha_cumprod) ** 0.5 * predicted_noise) / alpha ** 0.5

            if i > 0:
                noise = torch.randn_like(sample)
                sample += beta ** 0.5 * noise
        
        return sample
