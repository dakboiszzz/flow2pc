import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path or sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repc.data import generate_spiral_distribution

def main():
    print("Generating spiral dataset...")
    # Generate 60 marginals, 500 samples each to get a nice dense plot
    K = 60
    gt_data = generate_spiral_distribution(K=K, samples_per_marginal=500)
    gt_np = gt_data.numpy()
    
    print("Plotting...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    cmap = plt.cm.viridis
    for k in range(K):
        color = cmap(k / (K - 1))
        ax.scatter(gt_np[k, :, 0], gt_np[k, :, 1], color=color, alpha=0.3, s=6)

    ax.set_aspect("equal")
    ax.set_title("Ground Truth Spiral Dataset", fontsize=14, fontweight="bold")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Save the plot
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "spiral_dataset_plot.png"
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Plot saved successfully to {output_path}")

if __name__ == "__main__":
    main()
