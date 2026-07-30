"""Small UNet velocity field u_t^theta for flow matching on MNIST (1x28x28)."""

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (B,) in [0,1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        ang = t[:, None] * freqs[None] * 1000.0
        return torch.cat([ang.sin(), ang.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.t_proj(temb)[:, :, None, None]
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """28 -> 14 -> 7 -> 14 -> 28, channels base*(1,2,4)."""

    def __init__(self, base=64, t_dim=256, in_ch=1, stages=2):
        super().__init__()
        self.stages = stages
        c1, c2, c3 = base, base * 2, base * 4
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.stem = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.d1 = ResBlock(c1, c1, t_dim)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)  # 28->14
        self.d2 = ResBlock(c2, c2, t_dim)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)  # 14->7
        if stages == 3:
            c4 = base * 8
            self.down3 = nn.Conv2d(c3, c4, 3, stride=2, padding=1)
            self.d3 = ResBlock(c3, c3, t_dim)
            self.mid1 = ResBlock(c4, c4, t_dim)
            self.mid2 = ResBlock(c4, c4, t_dim)
            self.up3 = nn.ConvTranspose2d(c4, c3, 4, stride=2, padding=1)
            self.u3 = ResBlock(c3 + c3, c3, t_dim)
        else:
            self.mid1 = ResBlock(c3, c3, t_dim)
            self.mid2 = ResBlock(c3, c3, t_dim)
        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)  # 7->14
        self.u2 = ResBlock(c2 + c2, c2, t_dim)
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)  # 14->28
        self.u1 = ResBlock(c1 + c1, c1, t_dim)
        self.out_norm = nn.GroupNorm(8, c1)
        self.out = nn.Conv2d(c1, in_ch, 3, padding=1)

    def forward(self, x, t):
        temb = self.t_embed(t)
        h0 = self.stem(x)
        h1 = self.d1(h0, temb)
        h2 = self.d2(self.down1(h1), temb)
        if self.stages == 3:
            h3 = self.d3(self.down2(h2), temb)
            m = self.mid2(self.mid1(self.down3(h3), temb), temb)
            m = self.u3(torch.cat([self.up3(m), h3], dim=1), temb)
            y2 = self.u2(torch.cat([self.up2(m), h2], dim=1), temb)
        else:
            m = self.mid2(self.mid1(self.down2(h2), temb), temb)
            y2 = self.u2(torch.cat([self.up2(m), h2], dim=1), temb)
        y1 = self.u1(torch.cat([self.up1(y2), h1], dim=1), temb)
        return self.out(torch.nn.functional.silu(self.out_norm(y1)))


@torch.no_grad()
def euler_sample(model, n, steps=100, device="cuda", shape=(1, 28, 28)):
    """Integrate dz/dt = u_theta(z,t) from t=0 (noise) to t=1."""
    z = torch.randn(n, *shape, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt, device=device)
        z = z + dt * model(z, t)
    return z


class MLPFlow(nn.Module):
    """MLP velocity field for tabular data (any D)."""

    def __init__(self, D, hidden=512, t_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.net = nn.Sequential(
            nn.Linear(D + t_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, D),
        )

    def forward(self, z, t):
        half = self.t_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        ang = t[:, None] * freqs[None] * 1000.0
        return self.net(torch.cat([z, ang.sin(), ang.cos()], dim=-1))
