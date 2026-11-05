#!/usr/bin/env python3
"""Prepare/run 512 ALI-CFM teachers per real transport dataset; never distill.

The controller imports no Torch and initializes no CUDA context. --dry-run only
reads config/data headers and saved metadata; it writes nothing and starts no jobs.
"""

import argparse
from collections import deque
import copy
import csv
from dataclasses import dataclass, field
import errno
import fcntl
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from flow2pc.utils.config import load_config  # noqa: E402
from flow2pc.utils.gpu_queue import choose_device, stop_workers  # noqa: E402
from run_alicfm_checks import atomic_json, gpu_inventory, selected_devices  # noqa: E402

GRID = {
    "width": (128, 256),
    "gan_lr": (1e-5, 5e-5, 1e-4, 5e-4),
    "cfm_lr": (1e-4, 1e-3),
    "t_smooth": (0.0, 0.01),
    "correct_coeff": (0.0, 0.001, 0.01, 0.1),
    "coupling": ("border", "mmot"),
    "gan_loss": ("vanilla", "R3GAN"),
}
HOLDOUTS = {"gom": [1, 3, 5, 7], "beijing": [2, 5, 8, 11]}


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


@dataclass
class Run:
    dataset: str
    name: str
    config: dict
    output: Path
    params: dict
    action: str = "pretrain"
    status: str = "pending"
    error: str = ""
    metrics: dict = field(default_factory=dict)
    tracking_status: str = "unknown"
    retries: int = 0
    not_before: float = 0.0


def data_headers(root, dataset):
    """Read safe NPY headers from the NPZ; no data/model execution in preview."""
    path = root / dataset / "marginals.npz"
    expected = ({"samples.npy": [9, 111, 2], "times.npy": [9],
                 "train_indices.npy": [5], "holdout_indices.npy": [4]} if dataset == "gom" else
                {"values.npy": [9409, 1], "offsets.npy": [14], "times.npy": [13],
                 "train_indices.npy": [9], "holdout_indices.npy": [4], "month_indices.npy": [13]})
    shapes = {}
    with zipfile.ZipFile(path) as archive:
        for name, shape in expected.items():
            with archive.open(name) as handle:
                prefix = handle.read(8)
                if prefix[:6] != b"\x93NUMPY" or prefix[6] not in (1, 2):
                    raise ValueError(f"Unsupported NPY header: {path}:{name}")
                length = int.from_bytes(handle.read(2 if prefix[6] == 1 else 4), "little")
                import ast

                header = ast.literal_eval(handle.read(length).decode("latin1"))
            shapes[name] = list(header["shape"])
            if shapes[name] != shape or "O" in header["descr"]:
                raise ValueError(f"Unexpected array shape/dtype: {path}:{name}")
    return {"path": str(path), "shapes": shapes, "heldout_indices": HOLDOUTS[dataset]}


def build_runs(args):
    combinations = list(itertools.product(*GRID.values()))
    if len(combinations) != 512:
        raise ValueError("Sweep must contain 512 distinct configurations per dataset")
    random.Random(2026).shuffle(combinations)
    runs = []
    for values in combinations:
        params = dict(zip(GRID, values, strict=True))
        digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
        for dataset in args.datasets:
            config = load_config(ROOT / f"configs/pretrain/alicfm/alicfm_{dataset}.yaml")
            name = (f"{dataset}-h{params['width']}-glr{params['gan_lr']:g}-vlr{params['cfm_lr']:g}"
                    f"-ts{params['t_smooth']:g}-reg{params['correct_coeff']:g}"
                    f"-{params['coupling']}-{params['gan_loss']}-s{args.seed}-{digest}")
            output = args.output_root / dataset / name
            config["seed"] = args.seed
            model, training = config["model"]["params"], config["training"]
            for key in ("cfm_dim", "interpolant_dim", "disc_dim"):
                model[key] = params["width"]
            for key in ("pretain_ot", "interpolant_ot", "cfm_ot"):
                model[key] = params["coupling"]
            for key in ("t_smooth", "correct_coeff", "gan_loss"):
                model[key] = params[key]
            training.update(
                interpolant_gen_lr=params["gan_lr"], interpolant_disc_lr=params["gan_lr"],
                cfm_lr=params["cfm_lr"], num_ali_pretrain_steps=args.pretrain_steps,
                num_ali_train_steps=args.gan_steps, num_cfm_train_steps=args.cfm_steps,
            )
            config["data"]["root"] = str(args.data_root)
            config["output"]["dir"] = str(output)
            config["logging"].update(project="flow2pc", name=f"alicfm-{name}")
            runs.append(Run(dataset, name, config, output, params))
    return runs


def scientific_config(config):
    value = copy.deepcopy(config)
    for key in ("logging", "output"):
        value.pop(key, None)
    value["training"].pop("device", None)
    return value


def tracking_error(message):
    return any(text in str(message).lower() for text in ("wandb", "service process is busy", "failed to get resume status"))


def prepare(run, retry_incomplete=False):
    if not run.output.exists():
        return
    saved = run.output / "resolved_config.yaml"
    if not saved.is_file() or scientific_config(yaml.safe_load(saved.read_text())) != scientific_config(run.config):
        run.status, run.error = "blocked", "Existing scientific settings differ or metadata is missing; use a fresh sweep root"
        return
    if not (run.output / "checkpoints/last.pt").is_file():
        if retry_incomplete:
            run.action, run.status = "restart", "pending"
            run.error = "Previous incomplete attempt will be archived, then retrained"
        else:
            run.status, run.error = "blocked", "Partial training has no final checkpoint; use --retry-incomplete to archive and retry"
        return
    tracking = read_json(run.output / "tracking_status.json")
    run.tracking_status = tracking.get("status", "online")
    media = read_json(run.output / "eval/media.json")
    run.metrics = read_json(run.output / "eval/metrics.json")
    rows = [r for r in run.metrics.get("scores", []) if r.get("held_out")]
    correct_rows = sorted(r["index"] for r in rows) == HOLDOUTS[run.dataset]
    states = [read_json(run.output / path) for path in ("status.json", "eval/status.json")]
    states_complete = all(s.get("status") == "complete" for s in states)
    only_tracking_failed = all(s.get("status") == "complete" or tracking_error(s.get("error", "")) for s in states)
    expected_protocol = {k: v for k, v in run.config["evaluation"].items() if k not in ("enabled", "image_dir")}
    protocol_matches = all(run.metrics.get("protocol", {}).get(k) == v for k, v in expected_protocol.items())
    finite = correct_rows and all(math.isfinite(r["teacher_emd"]) for r in rows)
    mean_matches = finite and math.isclose(sum(r["teacher_emd"] for r in rows) / 4,
                                         run.metrics.get("teacher_heldout_emd", math.inf), rel_tol=1e-12, abs_tol=1e-12)
    if (states_complete or only_tracking_failed) and mean_matches and protocol_matches and media and all(Path(p).is_file() for p in media.values()):
        run.status = "complete"
        run.error = ""
        if not states_complete:
            run.tracking_status = "pending_sync"
    else:
        run.action = "evaluate"


def inspect_run(run, retry_incomplete=False):
    try:
        prepare(run, retry_incomplete)
    except (OSError, ValueError, yaml.YAMLError) as error:
        run.status, run.error = "blocked", f"Unreadable run metadata: {error}"


def command_for(run, config_path):
    command = [sys.executable, str(ROOT / "main.py"), "--config", str(config_path)]
    if run.action == "evaluate":
        command += ["--evaluate", "teacher"]
    return command


def check_worker_entrypoint():
    """Check the worker's CLI imports without constructing data or models."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--list-models"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src"),
             "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode:
        raise ValueError(
            "Worker CLI startup failed; no jobs were launched:\n"
            + (result.stderr or result.stdout)[-4000:]
        )
    listing = json.loads(result.stdout)
    if listing.get("alicfm", {}).get("status") != "implemented":
        raise ValueError("Worker CLI does not register ALI-CFM; no jobs were launched")


def archive_incomplete(run):
    if not run.output.exists():
        return
    if (run.output / "checkpoints/last.pt").exists():
        raise ValueError("Refusing to archive a trained teacher")
    destination = run.output.parents[1] / "_attempts" / run.dataset / run.name / str(time.time_ns())
    destination.parent.mkdir(parents=True, exist_ok=True)
    run.output.rename(destination)
    print(f"Preserved incomplete attempt at {destination}", flush=True)


def summary(args, runs, active, state):
    rows = []
    for run in runs:
        row = dict(dataset=run.dataset, run=run.name, status=run.status,
                   teacher_heldout_emd=run.metrics.get("teacher_heldout_emd"),
                   teacher_mean_emd=run.metrics.get("teacher_mean_emd"),
                   checkpoint=str(run.output / "checkpoints/last.pt"),
                   media_manifest=str(run.output / "eval/media.json"), tracking_status=run.tracking_status,
                   error=run.error, **run.params)
        scores = {r["index"]: r["teacher_emd"] for r in run.metrics.get("scores", [])}
        row.update({f"holdout_emd_{i}": scores.get(i) for i in HOLDOUTS[run.dataset]})
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
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
        ranked = sorted((r for r in selected if r["status"] == "complete"
                         and isinstance(r["teacher_heldout_emd"], (int, float))
                         and math.isfinite(r["teacher_heldout_emd"])), key=lambda r: r["teacher_heldout_emd"])
        write_csv(args.output_root / dataset / "summary.csv", selected)
        atomic_json(args.output_root / dataset / "best_models.json", {
            "selection_metric": "mean raw-coordinate W1 EMD across four temporal holdouts",
            "heldout_indices": HOLDOUTS[dataset], "best": ranked[:10],
            "selection_note": "Temporal holdouts are used for tuning; this is not an untouched test score",
        })
    atomic_json(args.output_root / "_launcher/status.json", {
        "status": state, "total": len(runs),
        "counts": {s: sum(r.status == s for r in runs) for s in ("pending", "running", "complete", "failed", "blocked")},
        "active": [{"run": w["task"].name, "gpu": w["device"], "pid": w["process"].pid} for w in active],
    })


def storage_error(error):
    return isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT)


def try_summary(args, runs, active, state):
    """Derived reports must not terminate workers when their filesystem fills."""
    try:
        summary(args, runs, active, state)
    except OSError as error:
        if not storage_error(error):
            raise
        print(f"Storage full: summary update deferred; active workers retained: {error}", file=sys.stderr, flush=True)
        return False
    return True


def worker_environment(args, device):
    # The root filesystem's /tmp also filled during this sweep. Keep new worker
    # temporary files and W&B staging/cache on the monitored output filesystem.
    runtime = args.output_root / "_runtime"
    directories = {"TMPDIR": runtime / "tmp", "WANDB_CACHE_DIR": runtime / "wandb-cache",
                   "WANDB_DATA_DIR": runtime / "wandb-staging"}
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           "CUDA_VISIBLE_DEVICES": device["uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
           "OPENBLAS_NUM_THREADS": str(args.threads), "NUMEXPR_NUM_THREADS": str(args.threads),
           "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "FLOW2PC_WANDB_INIT_LOCK": str(args.output_root / ".wandb-init.lock"),
           **{key: str(path) for key, path in directories.items()}}
    env.update(TMP=env["TMPDIR"], TEMP=env["TMPDIR"])
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for key in ("WANDB_RUN_ID", "WANDB_NAME", "WANDB_RESUME"):
        env.pop(key, None)
    return env


def start_worker(args, task, device, inventory, lock):
    if task.action == "restart":
        archive_incomplete(task)
        task.action, task.error = "pretrain", ""
    config_path = args.output_root / "_configs" / task.dataset / f"{task.name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(task.config, sort_keys=False))
    log_path = args.output_root / "_launcher/logs" / f"{task.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = worker_environment(args, inventory[device])
    log = log_path.open("a")
    try:
        process = subprocess.Popen(command_for(task, config_path), cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                   pass_fds=(lock.fileno(),))
    except BaseException:
        log.close()
        raise
    return dict(task=task, device=device, process=process, log=log, started=time.monotonic())


def run_queue(args, runs, devices, lock):
    pending = deque(run for run in runs if run.status == "pending")
    active, interrupted, previous, cooldown = [], False, {}, {}
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, stop)
    next_summary = 0.0
    storage_retry_after, storage_warned = 0.0, False
    try:
        while pending or active:
            if interrupted:
                break
            for worker in active[:]:
                code = worker["process"].poll()
                if code is None:
                    continue
                worker["log"].close()
                active.remove(worker)
                task = worker["task"]
                task.status = "pending"
                inspect_run(task)
                if task.status != "complete":
                    task.status = "failed"
                    recovery = ("saved checkpoint permits evaluation-only recovery" if
                                (task.output / "checkpoints/last.pt").is_file() else
                                "no final checkpoint; training was incomplete and is preserved")
                    task.error = f"Worker exit {code}; inspect _launcher/logs/{task.name}.log; {recovery}"
                    state = read_json(task.output / "status.json")
                    error = state.get("error", "")
                    oom = "out of memory" in error.lower()
                    if (oom or tracking_error(error)) and task.retries < args.worker_retries and not (task.output / "checkpoints/last.pt").exists():
                        task.retries += 1
                        task.status, task.action = "pending", "restart"
                        task.not_before = time.monotonic() + 120
                        cooldown[worker["device"]] = task.not_before
                        pending.append(task)
                print(f"{task.status}: {task.name}", flush=True)
                next_summary = 0.0
            try:
                inventory = gpu_inventory()
            except RuntimeError as error:
                print(f"GPU query unavailable; keeping workers alive and retrying: {error}", flush=True)
                time.sleep(5)
                continue
            free = {d: inventory[d]["free"] if time.monotonic() >= cooldown.get(d, 0) else 0 for d in devices}
            disk_free_gb = shutil.disk_usage(args.output_root).free / 2**30
            storage_ready = disk_free_gb >= args.min_disk_free_gb and time.monotonic() >= storage_retry_after
            if pending and not storage_ready and not storage_warned:
                print(f"Pausing new launches: {disk_free_gb:.1f} GiB free; require {args.min_disk_free_gb:g} GiB and no recent storage-write failure. Existing workers continue.", file=sys.stderr, flush=True)
                storage_warned = True
            elif storage_ready:
                storage_warned = False
            while pending and storage_ready:
                if shutil.disk_usage(args.output_root).free / 2**30 < args.min_disk_free_gb:
                    storage_ready = False
                    break
                device = choose_device(devices, active, free, args.workers_per_gpu, args.min_free_gb, time.monotonic())
                if device is None or len(active) >= args.max_parallel:
                    break
                task = next((r for r in pending if r.not_before <= time.monotonic()), None)
                if task is None:
                    break
                pending.remove(task)
                try:
                    worker = start_worker(args, task, device, inventory, lock)
                except OSError as error:
                    if not storage_error(error):
                        raise
                    pending.appendleft(task)
                    storage_retry_after = time.monotonic() + 30
                    storage_ready = False
                    print(f"Storage full: launch deferred; active workers retained: {error}", file=sys.stderr, flush=True)
                    break
                task.status = "running"
                active.append(worker)
                print(f"Started {task.action}: {task.name}, GPU {device}, pid {worker['process'].pid}", flush=True)
                next_summary = 0.0
            if time.monotonic() >= next_summary:
                if not try_summary(args, runs, active, "running" if storage_ready else "waiting_for_disk"):
                    storage_retry_after = time.monotonic() + 30
                next_summary = time.monotonic() + 30
            time.sleep(2)
    finally:
        aborted = sys.exc_info()[0] is not None
        if active:
            stop_workers(active)
            for worker in active:
                worker["task"].status = "pending"
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        try_summary(args, runs, [], "interrupted" if interrupted else "aborted" if aborted else "finished")
    return 130 if interrupted else (1 if any(r.status in ("failed", "blocked") for r in runs) else 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("gom", "beijing"), default=["gom", "beijing"])
    parser.add_argument("--data-root", type=Path, default=Path("/data/lthung/otp-fm"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/lthung/flow2pc/alicfm-transport/sweep-512"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--min-free-gb", type=float, default=4)
    parser.add_argument("--min-disk-free-gb", type=float, default=20, help="Pause new workers below this many GiB free on the output filesystem")
    parser.add_argument("--worker-retries", type=int, default=1)
    parser.add_argument("--retry-incomplete", action="store_true", help="Preserve partial runs in _attempts before retraining; never overwrite final checkpoints")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrain-steps", type=int, default=2000)
    parser.add_argument("--gan-steps", type=int, default=30000)
    parser.add_argument("--cfm-steps", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    if (len(set(args.datasets)) != len(args.datasets) or args.workers_per_gpu < 1
            or args.max_parallel < 2 or args.threads < 1 or args.pretrain_steps < 0
            or min(args.gan_steps, args.cfm_steps) < 1 or not 0 <= args.seed < 2**32
            or args.worker_retries < 0 or not math.isfinite(args.min_free_gb) or args.min_free_gb <= 0
            or not math.isfinite(args.min_disk_free_gb) or args.min_disk_free_gb <= 0):
        parser.error("Invalid sweep, seed, memory or concurrency settings")
    args.data_root, args.output_root = args.data_root.resolve(), args.output_root.resolve()
    headers = {name: data_headers(args.data_root, name) for name in args.datasets}
    runs = build_runs(args)
    for task in runs:
        inspect_run(task, args.retry_incomplete)
    if not args.summarize_only:
        try:
            check_worker_entrypoint()
        except (ValueError, OSError, subprocess.TimeoutExpired) as error:
            parser.error(str(error))
    inventory = {} if args.summarize_only else gpu_inventory()
    devices = [] if args.summarize_only else selected_devices(args.gpus, inventory)
    args.max_parallel = min(args.max_parallel, len(devices) * args.workers_per_gpu) if devices else 0
    print(json.dumps({"dry_run": args.dry_run, "total_runs": len(runs),
                     "runs_per_dataset": 512, "grid": GRID, "data": headers,
                     "gpus": devices, "workers_per_gpu": args.workers_per_gpu,
                     "max_parallel": args.max_parallel, "output_root": str(args.output_root),
                     "min_disk_free_gb": args.min_disk_free_gb,
                     "worker_runtime_dir": str(args.output_root / "_runtime"),
                     "project": "flow2pc", "preview_names": [r.name for r in runs[:4]],
                     "counts": {s: sum(r.status == s for r in runs) for s in ("pending", "complete", "blocked")}}, indent=2), flush=True)
    if args.dry_run:
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".sweep.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another controller/worker holds this sweep root; do not start duplicates")
        if args.summarize_only:
            summary(args, runs, [], "summary_only")
            return 0
        atomic_json(args.output_root / "sweep_plan.json", {
            "grid": GRID, "selection_metric": "teacher_heldout_emd", "heldout_indices": HOLDOUTS,
            "seed": args.seed, "runs": [{"name": r.name, "dataset": r.dataset, "config": r.config} for r in runs],
        })
        return run_queue(args, runs, devices, lock)


if __name__ == "__main__":
    raise SystemExit(main())
