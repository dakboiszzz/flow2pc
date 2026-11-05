#!/usr/bin/env python3
"""Tune ALI-CFM/OTP-FM on saved Arch/Sphere/LiDAR data; never run distillation.

The controller imports no Torch. --dry-run reads metadata, hashes input files and
resolves configs only; it writes no files and starts no processes.
"""

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

from flow2pc.utils.config import resolve_config  # noqa: E402
from flow2pc.utils.parallel_sweep import ParallelSweep  # noqa: E402
from run_alicfm_checks import atomic_json, gpu_inventory, selected_devices  # noqa: E402

DATASETS = ("arch", "sphere", "lidar")
METHODS = ("alicfm", "otpfm")
GRIDS = {
    "alicfm": {
        "width": (64, 128, 256, 512),
        "generator_lr": (1e-5, 5e-5, 1e-4, 5e-4),
        "cfm_lr": (1e-4, 3e-4, 1e-3, 3e-3),
        "batch_size": (64, 128, 256, 512),
        "coupling": ("none", "border"),
    },
    "otpfm": {
        "hidden_dim": (64, 128, 256, 512),
        "num_hidden_layers": (2, 4, 6, 8),
        "lr": (1e-4, 3e-4, 1e-3, 3e-3),
        "batch_size": (64, 128, 256, 512),
        "consistency_loss": ("meanflow", "imf"),
        "ema_decay": (.99, .999),
    },
}


@dataclass
class Run:
    method: str
    dataset: str
    name: str
    config: dict
    output: Path
    params: dict
    prepared_identity: str
    observed: list
    heldout: list
    action: str = "pretrain"
    status: str = "pending"
    error: str = ""
    metrics: dict = field(default_factory=dict)
    selection: dict = field(default_factory=dict)
    tracking_status: str = "unknown"
    retries: int = 0
    not_before: float = 0.


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def verify_inputs(root, datasets):
    """Audit saved identities/files without generating or evaluating a dataset."""
    result = {}
    for name in datasets:
        directory = root / "prepared" / name / "seed42"
        manifest = read_json(directory / "manifest.json")
        if (manifest.get("format") != "mfm_snapshots_v1" or manifest.get("profile") != name
                or manifest.get("identity") != digest({k: v for k, v in manifest.items() if k != "identity"})):
            raise ValueError(f"Invalid prepared {name} manifest: {directory}")
        expected_times = [0., 1.] if name == "lidar" else [0., .5, 1.]
        expected_holdouts = [] if name == "lidar" else [1]
        if (manifest["model_times"] != expected_times or manifest["heldout_indices"] != expected_holdouts
                or manifest["split_protocol"] != "independent_rows_90_10_observed_only"):
            raise ValueError(f"Unexpected saved MFM experiment split: {name}")
        for filename, expected in manifest["arrays"].items():
            if Path(filename).name != filename:
                raise ValueError("Prepared manifest path escapes its directory")
            value = hashlib.sha256()
            with (directory / filename).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(block)
            if value.hexdigest() != expected:
                raise ValueError(f"Prepared array changed: {directory / filename}")
        result[name] = manifest
    return result


def build_runs(args, metadata):
    selected, presets = {}, {}
    for method in args.methods:
        grid = GRIDS[method]
        combinations = list(itertools.product(*grid.values()))
        if not 1 <= args.runs_per_dataset <= len(combinations):
            raise ValueError(f"{method} has {len(combinations)} distinct active configurations")
        selected[method] = random.Random(args.grid_seed).sample(combinations, args.runs_per_dataset)
        for name in args.datasets:
            relative = f"{method}/{method}_{name}.yaml"
            presets[method, name] = yaml.safe_load((ROOT / "configs/pretrain" / relative).read_text())
    runs = []
    for index in range(args.runs_per_dataset):
        for name in args.datasets:
            for method in args.methods:
                config = copy.deepcopy(presets[method, name])
                params = dict(zip(GRIDS[method], selected[method][index], strict=True))
                config["seed"] = args.seed
                config["data"]["root"] = str(args.data_root / "prepared" / name / "seed42")
                config["training"]["device"] = "cuda:0"
                if method == "alicfm":
                    model, training = config["model"]["params"], config["training"]
                    for key in ("cfm_dim", "interpolant_dim", "disc_dim"):
                        model[key] = params["width"]
                    for key in ("pretain_ot", "interpolant_ot", "cfm_ot"):
                        model[key] = params["coupling"]
                    training.update(interpolant_gen_lr=params["generator_lr"], cfm_lr=params["cfm_lr"],
                                    batch_size=params["batch_size"])
                    if args.generator_steps is not None:
                        training["num_ali_train_steps"] = args.generator_steps
                    if args.steps is not None:
                        training["num_cfm_train_steps"] = args.steps
                    description = f"h{params['width']}-glr{params['generator_lr']:g}-vlr{params['cfm_lr']:g}-{params['coupling']}"
                else:
                    config["model"]["params"]["native_config"].update(params)
                    if args.steps is not None:
                        config["training"]["steps"] = args.steps
                    description = f"h{params['hidden_dim']}-l{params['num_hidden_layers']}-lr{params['lr']:g}-{params['consistency_loss']}"
                signature = digest({key: value for key, value in config.items() if key not in ("logging", "output")})[:12]
                run_name = f"{name}-{method}-c{index:03d}-{description}-b{params['batch_size']}-s{args.seed}-{signature}"
                output = args.output_root / method / name / run_name
                config["output"]["dir"] = str(output)
                config["logging"].update(mode=args.mode, project="flow2pc", name=run_name)
                config = resolve_config(config, base=ROOT)
                saved = metadata[name]
                heldout = saved["heldout_indices"]
                observed = [i for i in range(len(saved["model_times"])) if i not in heldout]
                runs.append(Run(method, name, run_name, config, output, params,
                                saved["identity"], observed, heldout))
    return runs


def scientific_config(config):
    config = copy.deepcopy(config)
    config.pop("logging", None)
    config.pop("output", None)
    config["training"].pop("device", None)
    return config


def inspect_run(run, retry_incomplete=False):
    if not run.output.exists():
        return
    try:
        path = run.output / "resolved_config.yaml"
        if not path.is_file() or scientific_config(yaml.safe_load(path.read_text())) != scientific_config(run.config):
            run.status, run.error = "blocked", "Existing scientific settings differ; use a fresh sweep root"
            return
        if not (run.output / "checkpoints/last.pt").is_file():
            run.action = "restart"
            run.status = "pending" if retry_incomplete else "blocked"
            run.error = read_json(run.output / "status.json").get("error", "Incomplete training; --retry-incomplete archives and retries it")
            return
        run.metrics = read_json(run.output / "eval/metrics.json")
        run.selection = read_json(run.output / "eval/selection.json")
        run.tracking_status = read_json(run.output / "tracking_status.json").get("status", "unknown")
        validation = read_json(run.output / "eval/validation/metrics.json")
        media = read_json(run.output / "eval/media.json")
        score = run.selection.get("score")
        valid = (isinstance(score, (int, float)) and math.isfinite(score)
                 and run.selection.get("prepared_identity") == run.prepared_identity
                 and run.selection.get("source_split") == run.selection.get("reference_split") == "val"
                 and run.selection.get("time_indices") == run.observed
                 and sorted(row["index"] for row in validation.get("scores", [])) == run.observed
                 and math.isclose(score, validation.get("teacher_mean_emd", math.inf), rel_tol=1e-12)
                 and sorted(row["index"] for row in run.metrics.get("scores", []) if row.get("held_out")) == run.heldout
                 and "teacher_geometry" in media and all(Path(value).is_file() for value in media.values()))
        expected = {k: v for k, v in run.config["evaluation"].items() if k not in ("enabled", "image_dir")}
        valid = valid and all(run.metrics.get("protocol", {}).get(k) == v for k, v in expected.items())
        valid = valid and all(read_json(run.output / item).get("status") == "complete"
                              for item in ("eval/status.json", "eval/validation/status.json"))
        if valid:
            run.status, run.error = "complete", ""
        else:
            run.action, run.status = "evaluate", "pending"
            run.error = read_json(run.output / "status.json").get("error", "")
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        run.status, run.error = "blocked", f"Unreadable saved metadata: {error}"


def archive_incomplete(run):
    if (run.output / "checkpoints/last.pt").exists():
        raise ValueError("Refusing to archive a trained teacher")
    if run.output.exists():
        destination = run.output.parents[2] / "_attempts" / run.method / run.dataset / run.name / str(time.time_ns())
        destination.parent.mkdir(parents=True, exist_ok=True)
        run.output.rename(destination)


def summary(args, runs, active, state):
    rows = [{"method": run.method, "dataset": run.dataset, "run": run.name, "status": run.status,
             "validation_emd": run.selection.get("score"),
             "test_heldout_emd": run.metrics.get("teacher_heldout_emd"),
             "evaluation_final_emd": run.metrics.get("teacher_final_emd"),
             "checkpoint": str(run.output / "checkpoints/last.pt"),
             "image": str(run.output / "eval/teacher_geometry.png"),
             "tracking_status": run.tracking_status, "error": run.error, **run.params} for run in runs]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    def csv_file(path, items):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(items)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        try:
            temporary.write_text(stream.getvalue())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    csv_file(args.output_root / "summary.csv", rows)
    for method in args.methods:
        for name in args.datasets:
            selected = [row for row in rows if row["method"] == method and row["dataset"] == name]
            ranked = sorted((row for row in selected if row["status"] == "complete"), key=lambda row: row["validation_emd"])
            destination = args.output_root / method / name
            csv_file(destination / "summary.csv", selected)
            atomic_json(destination / "best_models.json", {
                "selection": "validation_emd; saved observed-time validation rows only",
                "test_score_used_for_selection": False, "best": ranked[:10],
            })
    atomic_json(args.output_root / "_launcher/status.json", {
        "status": state, "total": len(runs),
        "counts": {name: sum(run.status == name for run in runs) for name in ("pending", "running", "complete", "failed", "blocked", "interrupted")},
        "active": [{"run": item["task"].name, "gpu": item["device"], "pid": item["process"].pid} for item in active],
    })


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    result.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    result.add_argument("--runs-per-dataset", type=int, default=256)
    result.add_argument("--data-root", type=Path, default=Path("/data/pqhung_group/pdtung/flow2pc/data/mfm"))
    result.add_argument("--output-root", type=Path, default=Path("/data/lthung/flow2pc/mfm-teachers/sweep-256"))
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--grid-seed", type=int, default=2026)
    result.add_argument("--steps", type=int, help="ALI field updates / OTP requested updates; defaults 30000")
    result.add_argument("--generator-steps", type=int, help="ALI regularizer stage; default 10000")
    result.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    result.add_argument("--workers-per-gpu", type=int, default=2)
    result.add_argument("--max-parallel", type=int, default=16)
    result.add_argument("--threads", type=int, default=4)
    result.add_argument("--min-free-gb", type=float, default=6)
    result.add_argument("--min-disk-free-gb", type=float, default=20)
    result.add_argument("--worker-retries", type=int, default=1)
    result.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    result.add_argument("--retry-incomplete", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--summarize-only", action="store_true")
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if (len(set(args.datasets)) != len(args.datasets) or len(set(args.methods)) != len(args.methods)
            or min(args.runs_per_dataset, args.workers_per_gpu, args.max_parallel, args.threads) < 1
            or any(value is not None and value < 1 for value in (args.steps, args.generator_steps))
            or not 0 <= args.seed < 2**32 or not 0 <= args.grid_seed < 2**32 or args.worker_retries < 0
            or any(not math.isfinite(value) or value <= 0 for value in (args.min_free_gb, args.min_disk_free_gb))):
        cli.error("Invalid dataset, seed, budget or resource setting")
    args.data_root, args.output_root = args.data_root.resolve(), args.output_root.resolve()
    metadata = verify_inputs(args.data_root, args.datasets)
    runs = build_runs(args, metadata)
    for run in runs:
        inspect_run(run, args.retry_incomplete)
    print(json.dumps({"dry_run": args.dry_run, "runs": len(runs), "runs_per_dataset_per_method": args.runs_per_dataset,
                      "datasets": args.datasets, "methods": args.methods, "gpus": args.gpus,
                      "max_parallel": args.max_parallel, "output_root": str(args.output_root),
                      "ranking": "observed-time reserved-row W1 EMD; middle-time test excluded",
                      "project": "flow2pc", "wandb_mode": args.mode,
                      "protocol": "saved MFM splits; two-observation adaptations; interior GAN/potential inactive",
                      "counts": {name: sum(run.status == name for run in runs) for name in ("pending", "complete", "blocked")},
                      "preview_names": [run.name for run in runs[:6]]}, indent=2), flush=True)
    if args.dry_run:
        return 0
    inventory = {} if args.summarize_only else gpu_inventory()
    devices = [] if args.summarize_only else selected_devices(args.gpus, inventory)
    if devices:
        args.max_parallel = min(args.max_parallel, len(devices) * args.workers_per_gpu)
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".sweep.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            cli.error("An active controller/worker owns this sweep root")
        if args.summarize_only:
            summary(args, runs, [], "summary_only")
            return 0
        atomic_json(args.output_root / "sweep_plan.json", {
            "grid_seed": args.grid_seed, "seed": args.seed, "grids": GRIDS,
            "data_identities": {name: value["identity"] for name, value in metadata.items()},
            "runs": [{"method": run.method, "dataset": run.dataset, "name": run.name, "config": run.config} for run in runs],
        })
        return ParallelSweep(args, runs, repo=ROOT, inventory=gpu_inventory,
                             inspect=inspect_run, report=summary, archive=archive_incomplete).run(devices, lock)


if __name__ == "__main__":
    raise SystemExit(main())
