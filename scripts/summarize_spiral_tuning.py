import argparse
import csv
import json
from pathlib import Path

COLUMNS = (
    "run_timestamp",
    "dataset",
    "num_latents",
    "sum_layer",
    "time_conditioning",
    "time_features",
    "learning_rate",
    "density_weight",
    "parameters",
    "parameter_ratio",
    "train_seconds",
    "final_train_loss",
    "best_validation_step",
    "best_validation_mse",
    "trajectory_mean_mse",
    "trajectory_var_mse",
    "trajectory_nll_nats"
)

def get_trajectory_metrics(run_dir: Path, is_teacher: bool = False) -> tuple[float | None, float | None, float | None]:
    cache_file = run_dir / "trajectory_metrics.json" if not is_teacher else Path("outputs") / "teacher_trajectory_metrics.json"
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            return data.get("mean_mse"), data.get("var_mse"), data.get("nll")
        except:
            pass
            
    import sys, subprocess
    if is_teacher:
        ckpt = Path("checkpoints/spiral_alicfm_teacher.pt")
        mtype = "teacher"
    else:
        ckpt = run_dir / "checkpoints" / "last.pt"
        mtype = "student"
        
    if not ckpt.exists():
        return None, None, None
        
    img_out = run_dir / "student_trajectories.png" if not is_teacher else Path("outputs") / "teacher_trajectories.png"
    cmd = [sys.executable, "scripts/plot_ode_path.py", "--model-type", mtype, "--dataset", "spiral", "--checkpoint", str(ckpt), "--output", str(img_out)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception:
        return None, None, None
        
    mean_mse, var_mse, nll = None, None, None
    for line in res.stdout.splitlines():
        if line.startswith("METRIC_MEAN_MSE="):
            mean_mse = float(line.split("=")[1])
        elif line.startswith("METRIC_VAR_MSE="):
            var_mse = float(line.split("=")[1])
        elif line.startswith("METRIC_NLL_NATS="):
            nll = float(line.split("=")[1])
            
    if mean_mse is not None:
        cache_file.write_text(json.dumps({"mean_mse": mean_mse, "var_mse": var_mse, "nll": nll}))
    return mean_mse, var_mse, nll

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("outputs/runs/distill"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmark/spiral-vanilla-tuning"),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip ODE trajectory evaluation to summarize instantly",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="param_scaling_sweep",
        help="Only include runs whose wandb_tags contain this string.",
    )
    return parser.parse_args()

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as input_file:
        return list(csv.DictReader(input_file))

def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())

def float_or_blank(value):
    return "" if value in (None, "") else float(value)

def collect_rows(runs_root: Path, fast: bool = False, tag: str | None = None) -> tuple[list[dict], dict | None]:
    rows = []
    
    # Add teacher baseline row
    t_mean, t_var, t_nll = (None, None, None) if fast else get_trajectory_metrics(Path("."), is_teacher=True)
    rows.append({
        "run_timestamp": "Teacher Baseline",
        "dataset": "spiral",
        "num_latents": "N/A",
        "sum_layer": "N/A",
        "time_conditioning": "alicfm_mlp",
        "time_features": "N/A",
        "learning_rate": "N/A",
        "density_weight": "N/A",
        "parameters": 133378,
        "parameter_ratio": 1.0,
        "train_seconds": 0.0,
        "final_train_loss": 0.0,
        "best_validation_step": 0,
        "best_validation_mse": 0.0,
        "trajectory_mean_mse": t_mean if t_mean is not None else "",
        "trajectory_var_mse": t_var if t_var is not None else "",
        "trajectory_nll_nats": t_nll if t_nll is not None else "",
    })
    
    selections = []
    # Collect all run directories
    run_dirs = sorted(runs_root.iterdir())
    for i, run_dir in enumerate(run_dirs):
        if not run_dir.is_dir():
            continue
            
        config = read_json(run_dir / "config.json")
        if not config:
            continue
            
        # Filter for spiral dataset
        if config.get("dataset") != "spiral":
            continue
        
        # Verify it's part of the requested sweep (check tags)
        tags = config.get("wandb_tags", "")
        if tag is not None and tag not in tags:
            continue

        train_rows = read_csv(run_dir / "metrics.csv")
        
        if not train_rows:
            continue

        # For spiral dataset, there is no validation loader, so we just use the best training loss
        best_val = min(
            train_rows,
            key=lambda row: float(row.get("loss", float("inf")))
        )
        selected_step = int(best_val["step"])
        best_mse = float(best_val["loss"])

        selections.append(
            {
                "run": run_dir.name,
                "step": selected_step,
                "validation_mse": best_mse,
                "num_latents": config.get("num_latents", ""),
                "learning_rate": config.get("lr", "")
            }
        )

        final_train = train_rows[-1]
        
        time_conditioning = config.get("time_conditioning", "polynomial")
        time_features = (
            config.get("time_embedding_dim", 16)
            if time_conditioning == "fourier"
            else config.get("time_order", 4)
        )
        
        if not fast:
            print(f"[{i+1}/{len(run_dirs)}] Evaluating trajectories for {run_dir.name}...")
        s_mean, s_var, s_nll = (None, None, None) if fast else get_trajectory_metrics(run_dir, is_teacher=False)
        rows.append(
            {
                "run_timestamp": run_dir.name,
                "dataset": config.get("dataset", ""),
                "num_latents": config.get("num_latents", ""),
                "sum_layer": config.get("pc_layer", ""),
                "time_conditioning": time_conditioning,
                "time_features": time_features,
                "learning_rate": config.get("lr", ""),
                "density_weight": config.get("density_weight", 0.0),
                "parameters": config.get("parameter_count", ""),
                "parameter_ratio": config.get("student_to_teacher_parameter_ratio", ""),
                "train_seconds": float_or_blank(final_train.get("elapsed_seconds")),
                "final_train_loss": float_or_blank(final_train.get("loss")),
                "best_validation_step": selected_step,
                "best_validation_mse": best_mse,
                "trajectory_mean_mse": s_mean if s_mean is not None else "",
                "trajectory_var_mse": s_var if s_var is not None else "",
                "trajectory_nll_nats": s_nll if s_nll is not None else "",
            }
        )

    best = None
    if selections:
        best = min(
            selections,
            key=lambda item: item["validation_mse"],
        )
    return rows, best

def format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

def main() -> None:
    args = parse_args()
    if not args.fast:
        print("Running ODE trajectory evaluations... this may take a while! Use --fast to skip.")
    rows, best = collect_rows(args.runs_root, fast=args.fast, tag=args.tag)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} tuning rows to {csv_path}")
    print(f"Best validation selection: {best}")

if __name__ == "__main__":
    main()
