import sys
import subprocess
import wandb
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/knot_alicfm_teacher.pt"))
    parser.add_argument("--wandb-project", default="flow2pc")
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name="teacher_baseline",
        tags=["teacher_baseline", "knot"],
        job_type="baseline-eval"
    )
    
    output_img = Path("outputs") / "teacher_baseline_trajectories.png"
    output_img.parent.mkdir(exist_ok=True, parents=True)
    
    print("Running ODE path plotting for teacher...")
    cmd = [
        sys.executable, "scripts/plot_ode_path.py",
        "--model-type", "teacher",
        "--checkpoint", str(args.checkpoint),
        "--output", str(output_img)
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    # Parse metrics
    mean_mse, var_mse = None, None
    for line in result.stdout.splitlines():
        print(line)
        if line.startswith("METRIC_MEAN_MSE="):
            mean_mse = float(line.split("=")[1])
        elif line.startswith("METRIC_VAR_MSE="):
            var_mse = float(line.split("=")[1])
    
    print("Logging plot and metrics to W&B...")
    log_dict = {
        "teacher/ode_path": wandb.Image(str(output_img), caption="Teacher ODE Path")
    }
    if mean_mse is not None:
        log_dict["teacher/mean_mse"] = mean_mse
        log_dict["teacher/var_mse"] = var_mse
        
    wandb.log(log_dict)
    
    print("Baseline evaluation complete.")
    run.finish()

if __name__ == "__main__":
    main()
