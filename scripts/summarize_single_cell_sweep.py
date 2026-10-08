#!/usr/bin/env python3
"""Summarize cached single-cell distillation runs and held-out ODE EMD.

Reads existing results without launching training or evaluation. Accepts either
a sweep directory or its runs/ directory; direct-sampling EMD is omitted.
"""

import argparse
import csv
import json
import math
from pathlib import Path


COLUMNS = (
    "run_name", "dataset", "dimensions", "heldout_day", "num_latents",
    "time_conditioning", "time_features", "learning_rate", "batch_size", "seed",
    "steps_configured", "steps_completed", "train_seconds", "final_train_loss",
    "best_train_loss", "best_validation_step", "best_validation_mse",
    "training_status", "evaluation_status", "failure_stage", "exit_code", "error",
    "checkpoint_exists", "evaluated_checkpoint_step", "heldout_ode_emd",
    "teacher_heldout_ode_emd", "integration_seconds", "ode_nfe_total",
    "wandb_id", "wandb_url",
)


def read_json(path):
    if not path.is_file():
        return {}
    try:
        result = json.loads(path.read_text())
        return result if isinstance(result, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Skipping unreadable JSON {path}: {exc}")
        return {}


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def training_metrics(path):
    last, best_loss, best_validation = {}, None, {}
    if path.is_file():
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                last = row
                loss = finite(row.get("loss"))
                if loss is not None and (best_loss is None or loss < best_loss):
                    best_loss = loss
                score = finite(row.get("validation_score"))
                if score is not None and (
                    not best_validation or score < float(best_validation["validation_score"])
                ):
                    best_validation = row
    return last, best_loss, best_validation


def collect_rows(runs_root, reference_path=None):
    sweep_root = runs_root.parent if runs_root.name == "runs" else runs_root
    if (runs_root / "runs").is_dir():
        runs_root = runs_root / "runs"
    binding = read_json(sweep_root / "sweep_inputs.json")
    reference_path = reference_path or (
        Path(binding["reference"]) if binding.get("reference") else None
    )
    reference = read_json(reference_path) if reference_path else {}
    if reference_path and not reference:
        raise ValueError(f"Teacher reference is missing or unreadable: {reference_path}")
    if reference and reference.get("model") != "teacher":
        raise ValueError("--reference must contain a separately evaluated teacher")
    teacher_emd = reference.get("heldout_ode_emd", "")
    contract = reference.get("data_contract", {})
    preprocessing = contract.get("preprocessing", {})
    baseline = {
        "run_name": "Teacher Baseline", "dataset": preprocessing.get("dataset", ""),
        "dimensions": preprocessing.get("n_pca_components_used", ""),
        "heldout_day": contract.get("heldout_day", ""),
        "training_status": "reference", "evaluation_status": "complete" if reference else "missing",
        "heldout_ode_emd": teacher_emd, "teacher_heldout_ode_emd": teacher_emd,
        "integration_seconds": reference.get("integration_seconds", ""),
        "ode_nfe_total": reference.get("ode_nfe_total", ""),
    }
    rows = [baseline]
    sweep_status = read_json(sweep_root / "sweep_status.json")
    failures = {item["name"]: item for item in sweep_status.get("failures", [])}
    for run in sorted(runs_root.iterdir()):
        if not run.is_dir():
            continue
        saved = read_json(run / "config.json")
        settings = read_json(sweep_root / "configs" / f"{run.name}.json")
        settings.update(saved.get("training", {}))
        if not settings:
            continue
        model = saved.get("model", {})
        run_contract = saved.get("data_contract", contract)
        prep = run_contract.get("preprocessing", {})
        status = read_json(run / "status.json")
        eval_status = read_json(run / "eval" / "status.json")
        evaluation = read_json(run / "eval" / "metrics.json")
        tracking = read_json(run / "wandb_run.json")
        last, best_loss, best_validation = training_metrics(run / "metrics.csv")
        failure = failures.get(run.name, {})
        train_status = status.get("status", "unknown")
        error = status.get("error", "") or eval_status.get("error", "")
        if failure.get("stage") == "train":
            train_status = "failed"
        if failure and not error:
            log_path = sweep_root / f"{run.name}.log"
            if log_path.is_file():
                for line in log_path.read_text(errors="replace").splitlines():
                    if "Error:" in line:
                        error = line.strip()
        if train_status == "running" and sweep_status.get("status") == "interrupted":
            train_status = "interrupted"
        rows.append({
            "run_name": run.name, "dataset": prep.get("dataset", ""),
            "dimensions": prep.get("n_pca_components_used", ""),
            "heldout_day": run_contract.get("heldout_day", ""),
            "num_latents": settings.get("num_latents", model.get("num_latents", "")),
            "time_conditioning": model.get("time_conditioning", "fourier"),
            "time_features": settings.get("time_embedding_dim", ""),
            "learning_rate": settings.get("lr", ""), "batch_size": settings.get("batch_size", ""),
            "seed": settings.get("seed", ""), "steps_configured": settings.get("steps", ""),
            "steps_completed": status.get("steps", last.get("step", "")),
            "train_seconds": last.get("elapsed_seconds", ""),
            "final_train_loss": last.get("loss", ""),
            "best_train_loss": best_loss if best_loss is not None else "",
            "best_validation_step": best_validation.get("step", ""),
            "best_validation_mse": status.get("best_validation_score", best_validation.get("validation_score", "")),
            "training_status": train_status,
            "evaluation_status": eval_status.get("status", "failed" if failure.get("stage") == "evaluate" else "pending"),
            "failure_stage": failure.get("stage", ""), "exit_code": failure.get("exit_code", ""),
            "error": error, "checkpoint_exists": (run / "checkpoints" / "best.pt").is_file(),
            "evaluated_checkpoint_step": evaluation.get("checkpoint_step", ""),
            "heldout_ode_emd": evaluation.get("heldout_ode_emd", ""),
            "teacher_heldout_ode_emd": evaluation.get("reference_heldout_ode_emd", teacher_emd),
            "integration_seconds": evaluation.get("integration_seconds", ""),
            "ode_nfe_total": evaluation.get("ode_nfe_total", ""),
            "wandb_id": tracking.get("id", ""), "wandb_url": tracking.get("url", ""),
        })
    eligible = [row for row in rows[1:] if row["training_status"] == "complete"
                and row["evaluation_status"] == "complete"
                and finite(row["best_validation_mse"]) is not None]
    best = min(eligible, key=lambda row: float(row["best_validation_mse"])) if eligible else None
    return rows, best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True,
                        help="Sweep directory or its runs/ directory")
    parser.add_argument("--reference", type=Path,
                        help="Teacher metrics.json; defaults to the sweep's recorded reference")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, best = collect_rows(args.runs_root, args.reference)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "metrics.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "best_selection.json").write_text(json.dumps({
        "selection_rule": "lowest observed-validation velocity MSE among completed, evaluated runs",
        "best": best,
    }, indent=2) + "\n")
    print(f"Wrote {len(rows) - 1} student runs and teacher baseline to {output}")
    print(f"Best validation selection: {best['run_name'] if best else 'none'}")


if __name__ == "__main__":
    main()
