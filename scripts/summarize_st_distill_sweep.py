#!/usr/bin/env python3
"""Summarize ST distillation sweep runs into one CSV row per run."""

import argparse
import csv
import json
from pathlib import Path


COLUMNS = (
    "run_name",
    "fold",
    "held_out_slide",
    "num_latents",
    "learning_rate",
    "time_embedding_dim",
    "marginal_nll_weight",
    "batch_size",
    "steps_configured",
    "steps_completed",
    "parameter_count",
    "train_seconds",
    "final_train_loss",
    "best_train_loss",
    "checkpoint_exists",
    "evaluation_complete",
    "media_logged",
    "held_out_emd_normalized",
    "held_out_emd_raw",
    "teacher_emd_raw",
    "generator_emd_raw",
    "held_out_mean_mse",
    "held_out_var_mse",
    "held_out_nll_nats",
    "velocity_mse",
    "relative_velocity_mse",
    "teacher_velocity_rms",
    "student_velocity_rms",
    "ode_steps",
)


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def read_last_training_row(path: Path) -> tuple[dict, float | None]:
    if not path.is_file():
        return {}, None
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}, None
    losses = []
    for row in rows:
        try:
            losses.append(float(row["loss"]))
        except (KeyError, TypeError, ValueError):
            pass
    return rows[-1], min(losses) if losses else None


def media_logged(run_dir: Path) -> bool:
    media_root = run_dir / "wandb"
    return any(
        path.is_file()
        and "media/images/eval/st_he_overlay" in str(path)
        for path in media_root.rglob("*")
    ) if media_root.is_dir() else False


def infer_fold(config: dict, run_name: str) -> tuple[str, str]:
    held_out = str(config.get("st_held_out_slide", ""))
    if held_out:
        return run_name.split("-h", 1)[0].removeprefix("st-"), held_out
    for label, slide in (("t=1_seed=45", "U3"), ("t=2_seed=42", "U4")):
        if label in run_name:
            return label, slide
    return "", ""


def collect_rows(runs_root: Path) -> list[dict]:
    rows = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        config = read_json(run_dir / "config.json")
        if not config or config.get("dataset") != "st":
            continue

        train_last, best_loss = read_last_training_row(run_dir / "metrics.csv")
        evaluation = read_json(run_dir / "eval" / "st_eval_metrics.json")
        fold, held_out = infer_fold(config, run_dir.name)
        checkpoint = run_dir / "checkpoints" / "last.pt"
        overlay = run_dir / "eval" / "st_he_overlay.png"
        row = {
            "run_name": run_dir.name,
            "fold": fold,
            "held_out_slide": held_out,
            "num_latents": config.get("num_latents", ""),
            "learning_rate": config.get("lr", ""),
            "time_embedding_dim": config.get("time_embedding_dim", ""),
            "marginal_nll_weight": config.get("marginal_nll_weight", ""),
            "batch_size": config.get("batch_size", ""),
            "steps_configured": config.get("resolved_steps", config.get("steps", "")),
            "steps_completed": train_last.get("step", ""),
            "parameter_count": config.get("parameter_count", ""),
            "train_seconds": train_last.get("elapsed_seconds", ""),
            "final_train_loss": train_last.get("loss", ""),
            "best_train_loss": "" if best_loss is None else best_loss,
            "checkpoint_exists": checkpoint.is_file(),
            "evaluation_complete": bool(evaluation),
            "media_logged": media_logged(run_dir),
        }
        metric_map = {
            "held_out_emd_normalized": "eval/st_held_out_emd_normalized",
            "held_out_emd_raw": "eval/st_held_out_emd_raw",
            "teacher_emd_raw": "eval/st_teacher_emd_raw",
            "generator_emd_raw": "eval/st_generator_emd_raw",
            "held_out_mean_mse": "eval/st_held_out_mean_mse",
            "held_out_var_mse": "eval/st_held_out_var_mse",
            "held_out_nll_nats": "eval/st_held_out_nll_nats",
            "velocity_mse": "eval/st_velocity_mse",
            "relative_velocity_mse": "eval/st_relative_velocity_mse",
            "teacher_velocity_rms": "eval/st_teacher_velocity_rms",
            "student_velocity_rms": "eval/st_student_velocity_rms",
            "ode_steps": "eval/st_ode_steps",
        }
        row.update({column: evaluation.get(key, "") for column, key in metric_map.items()})
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/runs/ST"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/runs/ST/st_distill_summary.csv"),
    )
    args = parser.parse_args()
    rows = collect_rows(args.runs_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} ST runs to {args.output}")


if __name__ == "__main__":
    main()
