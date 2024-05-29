import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = self.relu(out)
        return out

class DDPM(nn.Module):
    def __init__(self, num_blocks=3):
        super(DDPM, self).__init__()
        self.diffusion_steps = 1000
        self.input_layer = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.residual_blocks = self._make_layer(ResidualBlock, 64, 128, num_blocks)
        self.output_layer = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, in_channels, out_channels, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def initialize_with_prior(self, prior):
        self.prior = prior

    def step(self):
        noise = torch.randn_like(self.prior)
        out = self.input_layer(self.prior + noise)
        out = self.residual_blocks(out)
        out = self.output_layer(out)
        return self.relu(out)

    def generate(self, shape):
        generated_image = torch.randn(shape).to(self.prior.device)
        for _ in range(self.diffusion_steps):
            generated_image = self.step()
        return generated_image
