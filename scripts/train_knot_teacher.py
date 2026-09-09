import argparse
import sys
from pathlib import Path

# Ensure local repository root takes precedence over any installed site-packages
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path or sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

import torch
from torch import nn
from torchcfm.optimal_transport import OTPlanSampler
from repc.data import generate_knot_distribution


class KnotVelocityMLP(nn.Module):
    def __init__(self, in_dim: int = 3, out_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Knot ALI-CFM teacher checkpoint")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/knot_alicfm_teacher.pt"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--K", type=int, default=60)
    parser.add_argument("--samples-per-marginal", type=int, default=500)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def precompute_ot_couplings(data: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute exact Optimal Transport plan pairings for all K-1 transitions."""
    K, N, D = data.shape
    ot_sampler = OTPlanSampler(method="exact")
    coupled_x0 = []
    coupled_x1 = []

    print(f"Precomputing exact OT plans across {K - 1} marginal transitions ({N} samples each)...")
    for k in range(K - 1):
        m0 = data[k]
        m1 = data[k + 1]
        p0, p1 = ot_sampler.sample_plan(m0, m1)
        coupled_x0.append(p0)
        coupled_x1.append(p1)

    # Shape: [K-1, N, D]
    x0_all = torch.stack(coupled_x0, dim=0).to(device)
    x1_all = torch.stack(coupled_x1, dim=0).to(device)
    return x0_all, x1_all


def main():
    args = parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = KnotVelocityMLP(in_dim=3, out_dim=2, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-6
    )

    print(f"Generating Knot distribution (K={args.K}, samples_per_marginal={args.samples_per_marginal})...")
    raw_data = generate_knot_distribution(K=args.K, samples_per_marginal=args.samples_per_marginal)
    # raw_data: [K, N, 2]
    K = args.K
    x0_coupled, x1_coupled = precompute_ot_couplings(raw_data, device=device)
    num_transitions, num_samples, _ = x0_coupled.shape

    print(f"Training Knot teacher for {args.steps} steps on {device}...")
    loss_ema = None
    log_interval = max(args.steps // 10, 100)

    for step in range(1, args.steps + 1):
        # Sample random transition intervals k in [0, K-2] and sample indices in [0, N-1]
        k_indices = torch.randint(0, num_transitions, (args.batch_size,), device=device)
        sample_indices = torch.randint(0, num_samples, (args.batch_size,), device=device)

        x0 = x0_coupled[k_indices, sample_indices]  # [B, 2]
        x1 = x1_coupled[k_indices, sample_indices]  # [B, 2]

        # Sample local progress tau in [0, 1] within the interval
        tau = torch.rand(args.batch_size, 1, device=device)
        xt = (1.0 - tau) * x0 + tau * x1

        # Global continuous time t in [0, 1] and continuous velocity u_t = dx/dt
        k_col = k_indices.unsqueeze(1).float()
        t_global = (k_col + tau) / (K - 1.0)
        ut = (K - 1.0) * (x1 - x0)

        # Predict velocity field conditioned on (x_t, t_global)
        vt = model(torch.cat([xt, t_global], dim=-1))
        loss = nn.functional.mse_loss(vt, ut)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        loss_ema = loss_val if loss_ema is None else 0.95 * loss_ema + 0.05 * loss_val

        if step % log_interval == 0 or step == args.steps:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Step {step:5d}/{args.steps} | "
                f"Loss: {loss_val:.6f} (EMA: {loss_ema:.6f}) | "
                f"LR: {current_lr:.2e}"
            )

    # Save checkpoint in ALI-CFM format with global time indicator
    checkpoint = {
        "model_type": "flow",
        "time_format": "global",
        "model_config": {
            "name": "alicfm",
            "checkpoint_path": str(args.output.resolve()),
            "num_marginals": args.K,
            "time_format": "global",
            "hidden_dim": args.hidden_dim,
        },
        "ot_cfm_model": model.state_dict(),
        "flow_matching_config": {"name": "exact_ot_cfm", "sigma": 0.0},
    }
    torch.save(checkpoint, args.output)
    print(f"Saved teacher checkpoint to {args.output}")


if __name__ == "__main__":
    main()
