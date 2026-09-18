import argparse
import sys
from pathlib import Path
import json

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path or sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd

from repc.data import generate_spiral_distribution
from train import build_student

def load_student(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_student(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

@torch.no_grad()
def integrate_rk4_student(
    model: torch.nn.Module,
    x0: torch.Tensor,
    num_steps: int = 59,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    dt = 1.0 / num_steps
    time_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    trajectories = [x0.clone()]
    current_x = x0.clone()
    for i in range(num_steps):
        t_curr = time_steps[i].expand(current_x.shape[0])
        t_mid = (time_steps[i] + 0.5 * dt).expand(current_x.shape[0])
        t_next = time_steps[i + 1].expand(current_x.shape[0])

        k1 = model.velocity(current_x, t_curr)
        k2 = model.velocity(current_x + 0.5 * dt * k1, t_mid)
        k3 = model.velocity(current_x + 0.5 * dt * k2, t_mid)
        k4 = model.velocity(current_x + dt * k3, t_next)

        current_x = current_x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectories.append(current_x.clone())
    return torch.stack(trajectories, dim=1)

def evaluate_run(run_dir: Path, output_dir: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = run_dir / "checkpoints" / "last.pt"
    if not ckpt_path.exists():
        print(f"Skipping {run_dir.name}, no checkpoint found.")
        return

    print(f"Loading student from {ckpt_path}...")
    model = load_student(ckpt_path, device)
    
    K = 60
    gt_data = generate_spiral_distribution(K=K, samples_per_marginal=500) # (60, 500, 2)
    _, N, _ = gt_data.shape
    
    x0 = gt_data[0].to(device) # start from actual exact empirical x0
    print(f"Simulating {N} trajectories for {K-1} steps to match GT marginals...")
    
    # Process in chunks of 50 to avoid OOM for giant models (h4096)
    batch_size = 50
    trajectories_list = []
    for i in range(0, N, batch_size):
        x0_batch = x0[i:i+batch_size]
        traj_batch = integrate_rk4_student(model, x0_batch, num_steps=K-1, device=device)
        trajectories_list.append(traj_batch)
    
    trajectories = torch.cat(trajectories_list, dim=0) # (N, K, 2)
    traj_np = trajectories.cpu().numpy()
    gt_np = gt_data.numpy()
    
    time_steps = torch.linspace(0.0, 1.0, K, device=device)
    
    results = []
    print(f"Evaluating timesteps...")
    for k in range(K):
        sim_pts = traj_np[:, k, :]
        gt_pts = gt_np[k, :, :]
        
        mean_mse = float(np.mean((sim_pts.mean(axis=0) - gt_pts.mean(axis=0)) ** 2))
        var_mse = float(np.mean((sim_pts.var(axis=0) - gt_pts.var(axis=0)) ** 2))
        
        # Calculate NLL
        with torch.no_grad():
            t = time_steps[k]
            gt_tensor = torch.from_numpy(gt_pts).to(device)
            log_probs_list = []
            for i in range(0, N, batch_size):
                lp_batch = model.log_prob(gt_tensor[i:i+batch_size], t)
                log_probs_list.append(lp_batch)
            log_probs = torch.cat(log_probs_list, dim=0)
            nll = -log_probs.mean().item()
            
        results.append({
            "timestep_idx": k,
            "t": float(time_steps[k].item()),
            "mean_mse": mean_mse,
            "var_mse": var_mse,
            "nll_nats": nll
        })
        
    df = pd.DataFrame(results)
    csv_out = output_dir / f"{run_dir.name}_timestep_metrics.csv"
    df.to_csv(csv_out, index=False)
    print(f"Saved metrics to {csv_out}")
    
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    axs[0].plot(df["t"], df["mean_mse"], marker='o', color='b')
    axs[0].set_ylabel("Mean MSE")
    axs[0].set_title(f"Mean MSE vs Timestep ({run_dir.name})")
    axs[0].grid(True)
    
    axs[1].plot(df["t"], df["var_mse"], marker='s', color='r')
    axs[1].set_ylabel("Variance MSE")
    axs[1].set_title("Variance MSE vs Timestep")
    axs[1].grid(True)
    
    axs[2].plot(df["t"], df["nll_nats"], marker='^', color='g')
    axs[2].set_ylabel("NLL (nats)")
    axs[2].set_xlabel("Time (t)")
    axs[2].set_title("NLL vs Timestep")
    axs[2].grid(True)
    
    plot_out = output_dir / f"{run_dir.name}_timestep_metrics.png"
    plt.tight_layout()
    plt.savefig(plot_out, dpi=150)
    plt.close()
    print(f"Saved plot to {plot_out}")
    
    # Also plot the ODE Trajectories overlaying the GT data just to visualize the "curve breaking" 
    # To keep it lightweight, we'll plot only 100 trajectories
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    cmap = plt.cm.viridis
    for k in range(K):
        color = cmap(k / (K - 1))
        ax.scatter(gt_np[k, :, 0], gt_np[k, :, 1], color=color, alpha=0.15, s=6)
    
    for b in range(min(100, N)):
        ax.plot(traj_np[b, :, 0], traj_np[b, :, 1], color="crimson", alpha=0.5, linewidth=1.0)
        
    ax.scatter(traj_np[:, 0, 0], traj_np[:, 0, 1], color="forestgreen", s=30, zorder=5, label="Start")
    ax.scatter(traj_np[:, -1, 0], traj_np[:, -1, 1], color="purple", s=30, zorder=5, label="End")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_title(f"ODE Path ({run_dir.name})", fontsize=14, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right", frameon=True)
    
    path_plot_out = output_dir / f"{run_dir.name}_ode_paths.png"
    plt.tight_layout()
    plt.savefig(path_plot_out, dpi=150)
    plt.close()
    print(f"Saved path plot to {path_plot_out}")

def main():
    runs = [
        "2026-09-18_01-07-19-464523",
        "2026-09-18_01-18-39-814576",
        "2026-09-18_02-04-02-889732"
    ]
    
    base_runs_dir = Path("outputs/runs/distill")
    output_dir = Path("outputs/benchmark/spiral-mega-scaling-sweep/timestep_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for run in runs:
        run_dir = base_runs_dir / run
        if run_dir.exists():
            print(f"--- Analyzing {run} ---")
            evaluate_run(run_dir, output_dir)
        else:
            print(f"Run directory not found: {run_dir}")

if __name__ == "__main__":
    main()
