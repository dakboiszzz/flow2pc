import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2:
            raise ValueError("time embedding dimension must be even")
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / half
        )
        angles = time[:, None] * frequencies[None] * 1000.0
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor, time_embedding: torch.Tensor):
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(inputs)))
        hidden = hidden + self.time_projection(time_embedding)[:, :, None, None]
        hidden = self.conv2(torch.nn.functional.silu(self.norm2(hidden)))
        return hidden + self.skip(inputs)


class UNet(nn.Module):
    def __init__(self, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.stem = nn.Conv2d(1, c1, 3, padding=1)
        self.encoder1 = ResidualBlock(c1, c1, time_dim)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.encoder2 = ResidualBlock(c2, c2, time_dim)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.middle1 = ResidualBlock(c3, c3, time_dim)
        self.middle2 = ResidualBlock(c3, c3, time_dim)
        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.decoder2 = ResidualBlock(c2 + c2, c2, time_dim)
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.decoder1 = ResidualBlock(c1 + c1, c1, time_dim)
        self.output_norm = nn.GroupNorm(8, c1)
        self.output = nn.Conv2d(c1, 1, 3, padding=1)

    def forward(self, inputs: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(time)
        level1 = self.encoder1(self.stem(inputs), time_embedding)
        level2 = self.encoder2(self.down1(level1), time_embedding)
        middle = self.middle2(
            self.middle1(self.down2(level2), time_embedding),
            time_embedding,
        )
        level2_up = self.decoder2(
            torch.cat((self.up2(middle), level2), dim=1),
            time_embedding,
        )
        level1_up = self.decoder1(
            torch.cat((self.up1(level2_up), level1), dim=1),
            time_embedding,
        )
        return self.output(torch.nn.functional.silu(self.output_norm(level1_up)))
