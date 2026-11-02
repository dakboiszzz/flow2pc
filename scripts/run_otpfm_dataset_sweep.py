#!/usr/bin/env python3
"""OTP-FM hyperparameter search on shared datasets; preview imports no Torch."""

import argparse
import copy
import csv
import fcntl
import hashlib
import io
import itertools
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from flow2pc.utils.config import load_config, resolve_config  # noqa: E402
from flow2pc.utils.parallel_sweep import ParallelSweep  # noqa: E402
from flow2pc.pretrain.models.otpfm.provenance import verify_source  # noqa: E402
from run_alicfm_checks import atomic_json, gpu_inventory, selected_devices  # noqa: E402


# Search orchestration belongs with the launcher. Complete, independently
# runnable teacher settings live in configs/pretrain/otpfm_<dataset>.yaml.
DEFAULT_SWEEP = {
    "runs_per_dataset": 100,
    "seed": 42,
    "grid_seed": 2026,
    "datasets": {
        "cite5": {"config": "otpfm_cite5.yaml", "ranking_metric": "teacher_heldout_emd"},
        "cite50": {"config": "otpfm_cite50.yaml", "ranking_metric": "teacher_heldout_emd"},
        "cite100": {"config": "otpfm_cite100.yaml", "ranking_metric": "teacher_heldout_emd"},
        "multi5": {"config": "otpfm_multi5.yaml", "ranking_metric": "teacher_heldout_emd"},
        "multi50": {"config": "otpfm_multi50.yaml", "ranking_metric": "teacher_heldout_emd"},
        "multi100": {"config": "otpfm_multi100.yaml", "ranking_metric": "teacher_heldout_emd"},
        "eb": {"config": "otpfm_eb.yaml", "ranking_metric": "teacher_heldout_emd"},
        "st": {"config": "otpfm_st.yaml", "ranking_metric": "teacher_heldout_emd"},
        "cell_tracking": {"config": "otpfm_cell_tracking.yaml", "ranking_metric": "teacher_mean_emd"},
        "knot": {"config": "otpfm_knot.yaml", "ranking_metric": "teacher_mean_emd"},
    },
    "grid": {
        "hidden_dim": [128, 256, 512],
        "num_hidden_layers": [2, 4, 6],
        "lr": [0.0001, 0.0003, 0.001, 0.003],
        "batch_size": [64, 128, 256],
        "strength": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "width_multiplier": [0.25, 0.5, 1.0, 2.0],
        "lambda_type": ["gaussian", "box"],
        "consistency_loss": ["meanflow", "imf"],
        "otp_alpha_mean_scale": [0.5, 1.0],
    },
}


@dataclass
class Run:
    dataset: str
    name: str
    config: dict
    output: Path
    params: dict
    ranking_metric: str
    action: str = "pretrain"
    status: str = "pending"
    error: str = ""
    metrics: dict = field(default_factory=dict)
    tracking_status: str = "unknown"
    retries: int = 0
    not_before: float = 0.0


def scientific_config(config):
    result = copy.deepcopy(config)
    result.pop("output", None)
    result.pop("logging", None)
    result["training"].pop("device", None)
    return result


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def verify_inputs(data_root, aliases):
    """Check files/headers only; no training data arrays or model execution."""
    files = set()
    for alias in aliases:
        if alias.startswith("cite"):
            files.add("single_cell/op_cite_inputs_0.h5ad")
        elif alias.startswith("multi"):
            files.add("single_cell/op_train_multi_targets_0.h5ad")
        elif alias == "eb":
            files.add("single_cell/ebdata_v3.h5ad")
        elif alias == "st":
            for slide in range(2, 6):
                files.add(f"spatial_transcriptomics/aligned/aligned_spots/U{slide}_tumor_coordinates.csv")
            files.add("spatial_transcriptomics/aligned/ref_U5_warped_images/U3_aligned_to_U5.tif")
        elif alias == "cell_tracking":
            files.add("cell_tracking/prepared/mask_cell4_stack.npy")
            for frame in (0, 24, 49, 74, 99, 114):
                files.add(f"cell_tracking/PhC-C2DH-U373/01/t{frame:03d}.tif")
        elif alias == "knot":
            files.add("knot/knot_distribution.py")
        else:
            raise ValueError(f"Add an audited input profile for dataset {alias!r}")
    for relative in sorted(files):
        if not (data_root / relative).is_file():
            raise FileNotFoundError(f"Required experiment input is missing: {data_root / relative}")
    return sorted(files)


def build_runs(args, specification):
    grid = specification["grid"]
    combinations = list(itertools.product(*grid.values()))
    if args.runs_per_dataset > len(combinations):
        raise ValueError("Requested more configurations than the distinct search space")
    selected = random.Random(args.grid_seed).sample(combinations, args.runs_per_dataset)
    presets = {
        alias: load_config(ROOT / "configs/pretrain" / specification["datasets"][alias]["config"])
        for alias in args.datasets
    }
    if any(config["stage"] != "pretrain" or config["model"]["name"] != "otpfm" for config in presets.values()):
        raise ValueError("Each dataset must reference a standalone OTP-FM pretrain config")
    runs = []
    for index, values in enumerate(selected):
        parameters = dict(zip(grid, values, strict=True))
        for alias in args.datasets:
            profile = specification["datasets"][alias]
            config = copy.deepcopy(presets[alias])
            config["seed"] = args.seed
            config["data"]["root"] = str(args.data_root)
            config["training"]["device"] = "cuda:0"
            if args.steps is not None:
                config["training"]["steps"] = args.steps
            native = config["model"]["params"]["native_config"]
            native.update(parameters)
            signature_config = {key: value for key, value in config.items() if key not in ("logging", "output")}
            signature = hashlib.sha256(json.dumps(signature_config, sort_keys=True).encode()).hexdigest()[:12]
            name = (f"{alias}-c{index:03d}-h{native['hidden_dim']}-l{native['num_hidden_layers']}"
                    f"-lr{native['lr']:g}-b{native['batch_size']}-w{native['strength']:g}"
                    f"-tau{native['width_multiplier']:g}-{native['lambda_type']}-{native['consistency_loss']}"
                    f"-s{args.seed}-{signature}")
            output = args.output_root / alias / name
            config["output"]["dir"] = str(output)
            config["logging"].update(mode=args.mode, project="flow2pc", name=f"otpfm-{name}")
            config = resolve_config(config, base=ROOT)
            runs.append(Run(alias, name, config, output, parameters, profile["ranking_metric"]))
    return runs


def inspect_run(run, retry_incomplete=False):
    if not run.output.exists():
        return
    try:
        path = run.output / "resolved_config.yaml"
        if not path.is_file() or scientific_config(yaml.safe_load(path.read_text())) != scientific_config(run.config):
            run.status, run.error = "blocked", "Scientific settings differ or metadata is missing; use a fresh root"
            return
        checkpoint = run.output / "checkpoints/last.pt"
        state = read_json(run.output / "status.json")
        run.error = state.get("error", "")
        if not checkpoint.is_file():
            run.action = "restart"
            run.status = "pending" if retry_incomplete else "blocked"
            run.error = run.error or "Incomplete training has no final checkpoint; explicit --retry-incomplete archives/restarts it"
            return
        manifest = read_json(run.output / "manifest.json")
        if not manifest or not (run.output / "preprocessing.json").is_file():
            run.status, run.error = "blocked", "Saved weights lack restore metadata; preserve them for inspection"
            return
        run.tracking_status = read_json(run.output / "tracking_status.json").get("status", "unknown")
        run.metrics = read_json(run.output / "eval/metrics.json")
        media = read_json(run.output / "eval/media.json")
        protocol = {key: value for key, value in run.config["evaluation"].items() if key not in ("enabled", "image_dir")}
        matching = all(run.metrics.get("protocol", {}).get(key) == value for key, value in protocol.items())
        metric = run.metrics.get(run.ranking_metric)
        finite = isinstance(metric, (int, float)) and math.isfinite(metric)
        rows = run.metrics.get("scores", [])
        selected_rows = [row for row in rows if row.get("held_out")] if run.ranking_metric == "teacher_heldout_emd" else [row for row in rows if row["index"] != 0]
        aggregate = (bool(selected_rows) and all(math.isfinite(row["teacher_emd"]) for row in selected_rows)
                     and finite and math.isclose(sum(row["teacher_emd"] for row in selected_rows) / len(selected_rows), metric, rel_tol=1e-12, abs_tol=1e-12))
        valid = matching and aggregate and media and all(Path(value).is_file() for value in media.values())
        eval_state = read_json(run.output / "eval/status.json")
        infrastructure = any(word in (run.error + eval_state.get("error", "")).lower()
                             for word in ("wandb", "service process", "no space left", "disk quota"))
        if valid and (state.get("status") == "complete" and eval_state.get("status") == "complete" or infrastructure):
            run.status = "complete"
            if infrastructure:
                run.tracking_status = "pending_sync"
            run.error = ""
        else:
            run.action, run.status = "evaluate", "pending"
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        run.status, run.error = "blocked", f"Unreadable saved metadata: {error}"


def archive_incomplete(run):
    if (run.output / "checkpoints/last.pt").exists():
        raise ValueError("Refusing to archive a trained teacher")
    if run.output.exists():
        destination = run.output.parents[1] / "_attempts" / run.dataset / run.name / str(time.time_ns())
        destination.parent.mkdir(parents=True, exist_ok=True)
        run.output.rename(destination)
        print(f"Preserved incomplete attempt: {destination}", flush=True)


def summary(args, runs, active, state):
    rows = []
    for run in runs:
        row = {"dataset": run.dataset, "run": run.name, "status": run.status,
               "ranking_metric": run.ranking_metric, "ranking_score": run.metrics.get(run.ranking_metric),
               "teacher_heldout_emd": run.metrics.get("teacher_heldout_emd"), "teacher_mean_emd": run.metrics.get("teacher_mean_emd"),
               "checkpoint": str(run.output / "checkpoints/last.pt"), "media_manifest": str(run.output / "eval/media.json"),
               "tracking_status": run.tracking_status, "error": run.error, **run.params}
        rows.append(row)
    fields = list(rows[0])
    def write_csv(path, selected):
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
        temporary = path.with_suffix(".csv.tmp")
        try:
            temporary.write_text(stream.getvalue())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    write_csv(args.output_root / "summary.csv", rows)
    for dataset in args.datasets:
        selected = [row for row in rows if row["dataset"] == dataset]
        ranked = sorted((row for row in selected if row["status"] == "complete"), key=lambda row: row["ranking_score"])
        write_csv(args.output_root / dataset / "summary.csv", selected)
        atomic_json(args.output_root / dataset / "best_models.json", {
            "metric": selected[0]["ranking_metric"], "formula": "exact uniform-mass W1, Euclidean costs, full feature dimension",
            "note": "Temporal holdout used for hyperparameter selection" if selected[0]["ranking_metric"] == "teacher_heldout_emd" else "Observed-time reconstruction; no temporal holdout claimed",
            "best": ranked[:10],
        })
    atomic_json(args.output_root / "_launcher/status.json", {
        "status": state, "total": len(runs),
        "counts": {name: sum(run.status == name for run in runs) for name in ("pending", "running", "complete", "failed", "blocked", "interrupted")},
        "active": [{"run": worker["task"].name, "gpu": worker["device"], "pid": worker["process"].pid} for worker in active],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--runs-per-dataset", type=int)
    parser.add_argument("--steps", type=int, help="Optional update-budget override; defaults to each pretrain config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--grid-seed", type=int)
    parser.add_argument("--data-root", type=Path, default=Path("/data/lthung/ali-cfm-data"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/lthung/flow2pc/otp-fm/ali-data-sweep-100"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--max-parallel", type=int, default=16)
    parser.add_argument("--min-free-gb", type=float, default=4)
    parser.add_argument("--min-disk-free-gb", type=float, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--worker-retries", type=int, default=1)
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--retry-incomplete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    specification = copy.deepcopy(DEFAULT_SWEEP)
    args.datasets = args.datasets or list(specification["datasets"])
    for fieldname, key in (("runs_per_dataset", "runs_per_dataset"), ("seed", "seed"), ("grid_seed", "grid_seed")):
        if getattr(args, fieldname) is None:
            setattr(args, fieldname, specification[key])
    if (any(name not in specification["datasets"] for name in args.datasets)
            or len(set(args.datasets)) != len(args.datasets)
            or min(args.runs_per_dataset, args.max_parallel, args.workers_per_gpu, args.threads) < 1
            or args.steps is not None and args.steps < 1
            or not 0 <= args.seed < 2**32 or not 0 <= args.grid_seed < 2**32
            or args.worker_retries < 0 or any(not math.isfinite(value) or value <= 0 for value in (args.min_free_gb, args.min_disk_free_gb))):
        parser.error("Invalid dataset, budget, seed or resource setting")
    args.data_root, args.output_root = args.data_root.resolve(), args.output_root.resolve()
    provenance = verify_source()
    required_files = verify_inputs(args.data_root, args.datasets)
    runs = build_runs(args, specification)
    for run in runs:
        inspect_run(run, args.retry_incomplete)
    inventory = {} if args.summarize_only else gpu_inventory()
    devices = [] if args.summarize_only else selected_devices(args.gpus, inventory)
    if devices:
        args.max_parallel = min(args.max_parallel, len(devices) * args.workers_per_gpu)
    budgets = {run.dataset: run.config["training"]["steps"] for run in runs}
    common_budget = next(iter(budgets.values())) if len(set(budgets.values())) == 1 else budgets
    print(json.dumps({"dry_run": args.dry_run, "total_runs": len(runs), "runs_per_dataset": args.runs_per_dataset,
                      "datasets": args.datasets, "gpus": devices, "max_parallel": args.max_parallel,
                      "requested_updates_per_run": common_budget, "data_root": str(args.data_root),
                      "output_root": str(args.output_root), "project": "flow2pc", "wandb_mode": args.mode,
                      "method": "native W2Inf/direct OTP-FM; original MSE consistency training",
                      "metric": "existing velocity ODE + W1 EMD; not native finite-step W2",
                      "tracking_frames_one_based": [1, 25, 50, 75, 100, 115],
                      "min_disk_free_gb": args.min_disk_free_gb, "verified_source_files": len(provenance["files"]),
                      "required_input_files": len(required_files),
                      "counts": {name: sum(run.status == name for run in runs) for name in ("pending", "complete", "blocked")},
                      "preview_names": [run.name for run in runs[:3]]}, indent=2), flush=True)
    if args.dry_run:
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".sweep.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("An active sweep owns this output root; do not launch duplicates")
        if args.summarize_only:
            summary(args, runs, [], "summary_only")
            return 0
        atomic_json(args.output_root / "sweep_plan.json", {
            "sweep_specification": specification, "source_revision": provenance["upstream_revision"],
            "runs": [{"dataset": run.dataset, "run": run.name, "config": run.config} for run in runs],
        })
        return ParallelSweep(args, runs, repo=ROOT, inventory=gpu_inventory, inspect=inspect_run,
                             report=summary, archive=archive_incomplete).run(devices, lock)


if __name__ == "__main__":
    raise SystemExit(main())
