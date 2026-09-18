import sys
from pathlib import Path
import json
import csv

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path or sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from repc.data import generate_spiral_distribution
from train import build_student

RUN_IDS = [
    "2026-09-18_01-07-19-464523",
    "2026-09-18_01-18-39-814576",
    "2026-09-18_02-04-02-889732",
]

@torch.no_grad()
def integrate_rk4_student_batched(model, x0, num_steps, device, chunk_size=128):
    dt = 1.0 / num_steps
    time_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    trajectories = [x0.clone()]
    current_x = x0.clone()
    
    def get_vel(x, t_val):
        t_tensor = t_val.expand(x.shape[0])
        out = torch.zeros_like(x)
        for i in range(0, x.shape[0], chunk_size):
            out[i:i+chunk_size] = model.velocity(x[i:i+chunk_size], t_tensor[i:i+chunk_size])
        return out

    for i in range(num_steps):
        k1 = get_vel(current_x, time_steps[i])
        k2 = get_vel(current_x + 0.5 * dt * k1, time_steps[i] + 0.5 * dt)
        k3 = get_vel(current_x + 0.5 * dt * k2, time_steps[i] + 0.5 * dt)
        k4 = get_vel(current_x + dt * k3, time_steps[i + 1])

        current_x = current_x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectories.append(current_x.clone())
    return torch.stack(trajectories, dim=1)

def load_student_model(run_id, device):
    ckpt_path = Path(f"outputs/runs/distill/{run_id}/checkpoints/last.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_student(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def load_run_config(run_id):
    config_path = Path(f"outputs/runs/distill/{run_id}/config.json")
    with config_path.open() as f:
        return json.load(f)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = 60
    print("Generating ground truth Spiral distribution...")
    # Generate 500 samples per marginal for stable metric evaluation
    N_samples = 500
    gt_data = generate_spiral_distribution(K=K, samples_per_marginal=N_samples)
    gt_data_tensor = gt_data.to(device)
    gt_np = gt_data.numpy()
    
    x0 = gt_data_tensor[0] # The exact initial marginal samples

    all_metrics = []
    
    # Dictionaries to hold data for plotting
    plot_data = {
        run_id: {"times": [], "mean_mse": [], "var_mse": [], "nll": []}
        for run_id in RUN_IDS
    }

    for run_id in RUN_IDS:
        print(f"\n--- Evaluating Run: {run_id} ---")
        model = load_student_model(run_id, device)
        config = load_run_config(run_id)
        
        run_name = config.get("wandb_run_name", run_id)
        print(f"Run Name: {run_name}")
        
        print("Simulating trajectories via batched RK4 to prevent OOM...")
        # num_steps = K - 1 so that each step aligns exactly with the K marginals
        trajectories = integrate_rk4_student_batched(model, x0, num_steps=K-1, device=device, chunk_size=64)
        traj_np = trajectories.cpu().numpy()
        
        print("Computing step-by-step metrics...")
        for k in range(K):
            t_val = k / (K - 1)
            
            sim_k = traj_np[:, k, :]
            gt_k = gt_np[k]
            gt_k_tensor = gt_data_tensor[k]
            
            # MSE
            mean_mse = float(np.mean((sim_k.mean(axis=0) - gt_k.mean(axis=0)) ** 2))
            var_mse = float(np.mean((sim_k.var(axis=0) - gt_k.var(axis=0)) ** 2))
            
            # NLL loss
            # Only compute NLL if t > 0, because at t=0 the ground truth is basically a point mass
            if k == 0:
                nll = 0.0
            else:
                nlls = []
                for i in range(0, gt_k_tensor.shape[0], 64):
                    chunk = gt_k_tensor[i:i+64]
                    t_chunk = torch.tensor(t_val, device=device).expand(chunk.shape[0])
                    log_probs = model.log_prob(chunk, t_chunk)
                    nlls.append(-log_probs.mean().item())
                nll = float(np.mean(nlls))
            
            all_metrics.append({
                "run_id": run_id,
                "run_name": run_name,
                "time_step_k": k,
                "t": t_val,
                "num_latents": config.get("num_latents"),
                "density_weight": config.get("density_weight", 0.0),
                "mean_mse": mean_mse,
                "var_mse": var_mse,
                "nll_loss": nll
            })
            
            plot_data[run_id]["times"].append(t_val)
            plot_data[run_id]["mean_mse"].append(mean_mse)
            plot_data[run_id]["var_mse"].append(var_mse)
            plot_data[run_id]["nll"].append(nll)
            
    # Save to CSV
    output_dir = Path("outputs/benchmark/best_models_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "time_step_metrics.csv"
    
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)
    print(f"\nMetrics saved to {csv_path}")
    
    # Plotting
    print("Generating plots...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    
    for run_id in RUN_IDS:
        times = plot_data[run_id]["times"]
        run_name_short = load_run_config(run_id).get("wandb_run_name", run_id).split("-scale-")[-1]
        
        # Mean MSE Plot
        axs[0].plot(times, plot_data[run_id]["mean_mse"], label=run_name_short, lw=2)
        
        # Var MSE Plot
        axs[1].plot(times, plot_data[run_id]["var_mse"], label=run_name_short, lw=2)
        
        # NLL Plot (skip t=0)
        axs[2].plot(times[1:], plot_data[run_id]["nll"][1:], label=run_name_short, lw=2)

    axs[0].set_title("Trajectory Mean MSE over Time")
    axs[0].set_xlabel("Time (t)")
    axs[0].set_ylabel("Mean MSE")
    axs[0].set_yscale("log")
    
    axs[1].set_title("Trajectory Variance MSE over Time")
    axs[1].set_xlabel("Time (t)")
    axs[1].set_ylabel("Variance MSE")
    axs[1].set_yscale("log")
    
    axs[2].set_title("NLL Loss over Time")
    axs[2].set_xlabel("Time (t)")
    axs[2].set_ylabel("Negative Log-Likelihood (Nats)")
    
    for ax in axs:
        ax.legend()
        
    plt.tight_layout()
    plot_path = output_dir / "metrics_over_time.png"
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Plots saved to {plot_path}")

if __name__ == "__main__":
    main()
