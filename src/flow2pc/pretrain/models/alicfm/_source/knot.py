import torch
from torch import nn


class KnotVelocityMLP(nn.Module):
    """Outputs the continuous velocity field dx/dt.
    Two-layer MLP with 32 hidden units and SELU activations (paper spec).
    """
    def __init__(self, in_dim: int = 3, out_dim: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
        cat_input = torch.cat([x_t, t], dim=-1)
        return self.net(cat_input)

class KnotGeneratorMLP(nn.Module):
    """
    Parametrizes the path: G(x0, x1, t) = (1-t)x0 + t*x1 + t*(1-t)*f_phi(x0, x1, t)
    Two-layer MLP with 128 hidden units and ELU activation (paper spec).
    """
    def __init__(self, in_dim: int = 5, out_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
        cat_input = torch.cat([x_0, x_1, t], dim=-1)
        f_phi = self.net(cat_input)
        x_t = (1 - t) * x_0 + t * x_1 + (t * (1 - t)) * f_phi
        return x_t

class Discriminator(nn.Module):
    """Two-layer MLP with 128 hidden units and ELU activation (paper spec)."""
    def __init__(self, in_dim=3, w=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, w), nn.ELU(),
            nn.Linear(w, w), nn.ELU(),
            nn.Linear(w, 1)  # raw logits
        )
    def forward(self, x):
        return self.net(x)

def get_dx_dt(model, x0, x1, t):
    t_ = t.clone().requires_grad_(True)
    xt = model(x0, x1, t_)
    xt_x = xt[:, 0]
    xt_y = xt[:, 1]
    dx_dt_x = torch.autograd.grad(xt_x, t_, grad_outputs=torch.ones_like(xt_x), create_graph=True)[0]
    dx_dt_y = torch.autograd.grad(xt_y, t_, grad_outputs=torch.ones_like(xt_y), create_graph=True)[0]
    return torch.cat([dx_dt_x, dx_dt_y], dim=-1)
