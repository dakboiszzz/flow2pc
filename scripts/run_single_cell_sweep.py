"""Distill multiple student configurations using a separately evaluated teacher.

One worker occupies each configured device; cpu,cpu provides two CPU workers.
No shell commands are built from configuration text, and no external repository
code is imported. A failed configuration leaves the other workers running.
"""

import argparse
import csv
from collections import deque
import fcntl
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repc.single_cell import (DEFAULT_DATA_DIR, DEFAULT_TEACHER, atomic_json,
                             load_single_cell_data, sha256_file)
from repc.single_cell_tracking import add_tracking_arguments


def automatic_devices(limit):
    """Use visible logical GPUs with at least 8 GiB free, else CPUs.

    Like the ALI-CFM sweep, query PyTorch's logical device indices so scheduler
    CUDA_VISIBLE_DEVICES mappings (including GPU UUIDs) are respected.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return ["cpu"] * limit
        eligible = [f"cuda:{index}" for index in range(torch.cuda.device_count())
                    if torch.cuda.mem_get_info(index)[0] >= 8 * 2**30]
        if eligible:
            return eligible[:limit]
    except (OSError, ValueError, RuntimeError):
        pass
    return ["cpu"] * limit


def load_jobs(grid_path, overrides):
    grid = json.loads(grid_path.read_text())
    common = grid.get("common", {})
    if not isinstance(common, dict):
        raise ValueError("grid common settings must be an object")
    if ("runs" in grid) == ("parameters" in grid):
        raise ValueError("grid requires exactly one of runs or parameters")
    cartesian = "parameters" in grid
    if cartesian:
        parameters = grid["parameters"]
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("grid parameters must be a nonempty object of value lists")
        for key, values in parameters.items():
            if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or key == "name":
                raise ValueError(f"invalid sweep parameter: {key}")
            if not isinstance(values, list) or not values:
                raise ValueError(f"sweep parameter {key} requires a nonempty value list")
            if any(not isinstance(value, (str, int, float, bool, type(None))) for value in values):
                raise ValueError(f"sweep parameter {key} must contain scalar values")
            encoded = [json.dumps(value, sort_keys=True, allow_nan=False) for value in values]
            if len(set(encoded)) != len(encoded):
                raise ValueError(f"sweep parameter {key} contains duplicate values")
        keys = list(parameters)
        # JSON axis order controls the queue; the shipped grid varies capacity first.
        runs = (dict(zip(keys, values)) for values in product(*(parameters[key] for key in keys)))
    else:
        runs = grid["runs"]
        if not isinstance(runs, list) or not runs:
            raise ValueError("grid requires a nonempty runs list")
    jobs, names, job_names, settings_seen = [], set(), set(), set()
    abbreviations = {"num_latents": "h", "time_embedding_dim": "f", "lr": "lr",
                     "batch_size": "b", "seed": "s"}
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("each run must be an object")
        settings = {**common, **{k: v for k, v in run.items() if k != "name"}, **overrides}
        canonical = json.dumps(settings, sort_keys=True, allow_nan=False)
        if cartesian:
            # An override can collapse an axis: launch each resulting configuration once.
            if canonical in settings_seen:
                continue
            ordered = [key for key in abbreviations if key in parameters]
            ordered.extend(key for key in parameters if key not in abbreviations)
            parts = []
            for key in ordered:
                value = settings[key]
                label = format(value, ".8g") if isinstance(value, float) else str(value)
                label = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
                parts.append(f"{abbreviations.get(key, key)}{label}")
            name = "-".join(parts)[:160]
        else:
            name = run.get("name", "")
        if not isinstance(name, str):
            raise ValueError("configuration names must be strings")
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
                or (not cartesian and name in names)):
            raise ValueError("configuration names must be unique, simple directory names")
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()[:8]
        job_name = f"{name}-{fingerprint}"
        if job_name in job_names:
            raise ValueError("configuration directory collision; rename the configuration")
        names.add(name)
        job_names.add(job_name)
        settings_seen.add(canonical)
        jobs.append((job_name, settings))
    return jobs


def summarize(output, run_names=None):
    rows = []
    for run in sorted((output / "runs").glob("*")):
        if run_names is not None and run.name not in run_names:
            continue
        status_path = run / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "unknown"}
        row = {"run": run.name, "status": status["status"],
               "best_validation_score": status.get("best_validation_score", ""),
               "heldout_direct_emd": "", "heldout_ode_emd": "", "reference_heldout_ode_emd": ""}
        config_path = output / "configs" / f"{run.name}.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        row.update({key: config.get(key, "") for key in
                    ("num_latents", "time_embedding_dim", "lr", "batch_size", "seed", "steps")})
        eval_status_path = run / "eval/status.json"
        row["evaluation_status"] = (json.loads(eval_status_path.read_text()).get("status", "unknown")
                                    if eval_status_path.exists() else "pending")
        tracking_path = run / "wandb_run.json"
        tracking = json.loads(tracking_path.read_text()) if tracking_path.exists() else {}
        row.update({"wandb_id": tracking.get("id", ""), "wandb_url": tracking.get("url", "")})
        metrics_path = run / "eval/metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
            row.update({key: metrics[key] for key in ("heldout_direct_emd", "heldout_ode_emd",
                                                     "reference_heldout_ode_emd")})
        rows.append(row)
    if rows:
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    complete = [row for row in rows if row["status"] == "complete"
                and row["evaluation_status"] == "complete" and row["heldout_direct_emd"] != ""]
    if complete:
        winner = min(complete, key=lambda row: row["best_validation_score"])
        atomic_json(output / "selected_student.json", {
            "selection_rule": "lowest observed-validation velocity_mse; NLL and held-out EMD unused",
            "checkpoint": str((output / "runs" / winner["run"] / "checkpoints/best.pt").resolve()),
            "metrics": winner,
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=ROOT / "configs/single_cell/cite5d_intensive_sweep.json")
    parser.add_argument("--teacher-checkpoint", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--reference", type=Path, required=True,
                        help="metrics.json produced by a separate teacher evaluation")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/single_cell/cite5d-intensive")
    parser.add_argument("--devices", default="auto", help="auto, cpu,cpu, or cuda:0,cuda:1")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads per worker")
    parser.add_argument("--eval-samples", type=int,
                        help="must match the teacher reference; omitted inherits its sample count")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--emd-every", type=int, help="periodic direct EMD interval; 0 disables")
    parser.add_argument("--emd-samples", type=int, help="periodic direct EMD sample count")
    add_tracking_arguments(parser, mode_default=None)
    parser.set_defaults(wandb_project=None, wandb_entity=None, wandb_group=None, wandb_service_wait=None)
    parser.add_argument("--dry-run", action="store_true", help="print jobs without evaluation or training")
    args = parser.parse_args()
    if (args.max_parallel < 1 or args.threads < 1
            or (args.eval_samples is not None and args.eval_samples < 0)):
        parser.error("parallelism/threads must be positive; sample count nonnegative")
    overrides = {key: getattr(args, key) for key in (
        "steps", "emd_every", "emd_samples", "wandb_mode", "wandb_project",
        "wandb_entity", "wandb_group", "wandb_service_wait") if getattr(args, key) is not None}
    jobs = load_jobs(args.grid, overrides)
    devices = automatic_devices(args.max_parallel) if args.devices == "auto" else args.devices.split(",")
    devices = [device.strip() for device in devices][:args.max_parallel]
    if not devices or any(not re.fullmatch(r"cpu|cuda:\d+", device) for device in devices):
        parser.error("devices must be cpu or cuda:N")
    if len([x for x in devices if x != "cpu"]) != len({x for x in devices if x != "cpu"}):
        parser.error("assign each GPU once; repeated cpu workers are allowed")
    output = args.output_dir.resolve()
    teacher, data_dir = args.teacher_checkpoint.resolve(), args.data_dir.resolve()
    reference = args.reference.resolve()
    print(json.dumps({"teacher": str(teacher), "reference": str(reference), "devices": devices,
                      "grid": str(args.grid.resolve()), "num_jobs": len(jobs),
                      "jobs": [{"name": name, **settings} for name, settings in jobs]}, indent=2), flush=True)
    if args.dry_run:
        return
    output.mkdir(parents=True, exist_ok=True)
    if not reference.is_file():
        parser.error("teacher reference missing; run scripts/evaluate_single_cell.py first")
    cached = json.loads(reference.read_text())
    identity = load_single_cell_data(teacher, data_dir).identity
    protocol = cached["protocol"]
    cached_id = cached.get("identity", {})
    if cached.get("model") != "teacher" or cached_id.get("checkpoint_sha256") != identity.get("checkpoint_sha256"):
        raise ValueError("reference does not match this teacher/data contract")
    if (protocol["source_index"] != 0 or protocol["solver"] != "dopri5"
            or protocol["metric_space"] != identity["metric_space"]):
        raise ValueError("reference uses an unsupported evaluation protocol")
    if args.eval_samples is not None and args.eval_samples != protocol["num_samples"]:
        parser.error("--eval-samples must match the separate teacher reference")
    common_eval = ["--teacher-checkpoint", str(teacher), "--data-dir", str(data_dir),
                   "--num-samples", str(protocol["num_samples"]), "--seed", str(protocol["seed"]),
                   "--batch-size", str(protocol["batch_size"]), "--tolerance", str(protocol["tolerance"]),
                   "--threads", str(args.threads)]
    # Reject two launchers writing the same checkpoints/W&B IDs concurrently.
    lock = (output / ".sweep.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        parser.error("another sweep is using this output directory")
    binding = {"identity": identity, "reference": str(reference),
               "reference_sha256": sha256_file(reference), "protocol": protocol}
    binding_path = output / "sweep_inputs.json"
    if binding_path.exists():
        old_binding = json.loads(binding_path.read_text())
        if old_binding.get("reference_sha256") != binding["reference_sha256"] or old_binding.get("identity", {}).get("checkpoint_sha256") != binding["identity"]["checkpoint_sha256"]:
            lock.close()
            parser.error("sweep teacher/data/reference changed; use a new --output-dir")
    atomic_json(binding_path, binding)
    atomic_json(output / "sweep_manifest.json", {
        "grid": str(args.grid.resolve()), "grid_sha256": sha256_file(args.grid),
        "num_jobs": len(jobs), "devices": devices,
        "jobs": [{"name": name, "settings": settings} for name, settings in jobs],
    })
    print(f"Using separate teacher reference: {reference}", flush=True)
    queue = deque(jobs)
    available = deque(devices)
    active = []
    failures = []
    run_names = {name for name, _ in jobs}
    interrupted = False
    previous_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in previous_handlers:
        signal.signal(signum, interrupt)
    try:
        while queue or active:
            while queue and available:
                name, settings = queue.popleft()
                device = available.popleft()
                run = output / "runs" / name
                run.mkdir(parents=True, exist_ok=True)
                status_path = run / "status.json"
                completed = status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete"
                eval_path = run / "eval/metrics.json"
                eval_status = run / "eval/status.json"
                if (completed and eval_path.exists() and eval_status.exists()
                        and json.loads(eval_status.read_text()).get("status") == "complete"):
                    print(f"Already complete: {name}", flush=True)
                    available.append(device)
                    continue
                input_config = output / "configs" / f"{name}.json"
                atomic_json(input_config, settings)
                log = (output / f"{name}.log").open("a")
                train_command = [sys.executable, "-u", str(ROOT / "scripts/train_single_cell.py"),
                                 "--config", str(input_config), "--teacher-checkpoint", str(teacher),
                                 "--data-dir", str(data_dir), "--reference", str(reference),
                                 "--output-dir", str(run), "--device", device,
                                 "--threads", str(args.threads), "--resume"]
                eval_command = [sys.executable, "-u", str(ROOT / "scripts/evaluate_single_cell.py"),
                                *common_eval, "--device", device, "--reference", str(reference),
                                "--student-checkpoint", str(run / "checkpoints/best.pt"),
                                "--output-dir", str(run / "eval"), "--wandb-run-dir", str(run),
                                "--wandb-mode", settings.get("wandb_mode", "online"),
                                "--wandb-project", settings.get("wandb_project", "flow2pc"),
                                "--wandb-group", settings.get("wandb_group", "cite5d-distillation"),
                                "--wandb-service-wait", str(settings.get("wandb_service_wait", 300))]
                if settings.get("wandb_entity") or os.environ.get("WANDB_ENTITY"):
                    eval_command.extend(["--wandb-entity", settings.get("wandb_entity") or os.environ["WANDB_ENTITY"]])
                stage = "evaluate" if completed else "train"
                command = eval_command if completed else train_command
                child_env = {**os.environ, "OMP_NUM_THREADS": str(args.threads),
                             "MKL_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": str(args.threads),
                             "FLOW2PC_WANDB_INIT_LOCK": str(output / ".wandb-init.lock"),
                             "WANDB__SERVICE_WAIT": str(settings.get("wandb_service_wait", 300))}
                for key in ("WANDB_RUN_ID", "WANDB_NAME", "WANDB_RESUME"):
                    child_env.pop(key, None)
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           cwd=ROOT, env=child_env, start_new_session=True)
                active.append({"process": process, "device": device, "name": name,
                               "log": log, "stage": stage, "eval_command": eval_command,
                               "env": child_env})
                print(f"Started {name} [{stage}] on {device}; log {output / (name + '.log')}", flush=True)
            for item in active[:]:
                code = item["process"].poll()
                if code is None:
                    continue
                if code == 0 and item["stage"] == "train":
                    item["stage"] = "evaluate"
                    item["process"] = subprocess.Popen(item["eval_command"], stdout=item["log"],
                                                       stderr=subprocess.STDOUT, cwd=ROOT, env=item["env"],
                                                       start_new_session=True)
                    print(f"Evaluating {item['name']}", flush=True)
                    continue
                item["log"].close()
                active.remove(item)
                available.append(item["device"])
                if code:
                    try:
                        os.killpg(item["process"].pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    failures.append({"name": item["name"], "stage": item["stage"], "exit_code": code})
                print(f"Finished {item['name']} [{item['stage']}] exit={code}", flush=True)
                summarize(output, run_names)
            if active:
                time.sleep(1)
    except BaseException:
        interrupted = True
        raise
    finally:
        for signum in previous_handlers:
            signal.signal(signum, signal.SIG_IGN)
        for item in active:
            try:
                os.killpg(item["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for item in active:
            try:
                item["process"].wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(item["process"].pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                item["process"].wait()
            item["log"].close()
        summarize(output, run_names)
        atomic_json(output / "sweep_status.json", {
            "status": "interrupted" if interrupted else "failed" if failures else "complete",
            "reference": str(reference), "failures": failures})
        lock.close()
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
    if failures:
        raise SystemExit(f"Some configurations failed: {failures}")
    print(f"Sweep complete. Results: {output / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
