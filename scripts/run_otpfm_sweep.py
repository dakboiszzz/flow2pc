"""Prepare two frozen OTP-FM references, then sweep Flow2PC students on many GPUs.

Only students are optimized. The default student recipe is shared with ALI;
source-MLE anchoring is an explicit experiment. Pretrained teachers stay frozen.
--dry-run reads identities and prints the plan; it starts no worker or W&B run.
"""

import argparse
from collections import deque
import csv
import fcntl
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

import torch

from repc.otpfm import DEFAULT_OTP_ROOT, select_teacher
from repc.single_cell import atomic_json, sha256_file
from scripts.run_single_cell_sweep import load_jobs
from scripts.run_single_cell_all_sweeps import choose_device, stop_workers


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def gpu_free_memory(devices):
    """Read free bytes without creating a CUDA context in the scheduler."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"nvidia-smi memory query failed: {error}") from error
    physical = {}
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 2:
            continue
        try:
            physical[int(row[0].strip())] = max(0, int(row[1].strip())) * 2**20
        except ValueError:
            # An unavailable GPU can report N/A without invalidating other GPUs.
            continue
    free = {}
    for device in devices:
        try:
            # This helper reads NVML/visibility metadata only. Passing an integer
            # avoids current_device(), and honors numeric/UUID visibility masks.
            physical_index = torch.cuda._get_nvml_device_index(int(device.split(":")[1]))
            free[device] = physical.get(physical_index, 0)
        except (RuntimeError, ValueError, IndexError):
            free[device] = 0
    return free


def build_plan(args):
    plans = {}
    for dataset in args.datasets.split(","):
        if dataset not in ("gom", "beijing") or dataset in plans:
            raise ValueError("datasets must be unique names: gom,beijing")
        teacher, provenance = select_teacher(args.otp_root, dataset)
        overrides = {"wandb_mode": args.wandb_mode, "wandb_project": "flow2pc",
                     "student_recipe": args.student_recipe,
                     "wandb_entity": args.wandb_entity, "wandb_service_wait": 300,
                     "wandb_group": f"otpfm-{dataset}-w2-teacher{teacher['seed']}"}
        if args.steps is not None:
            overrides["steps"] = args.steps
        jobs = load_jobs(args.grid, overrides)
        plans[dataset] = {"teacher": teacher, "provenance": provenance, "jobs": jobs}
    return plans


def task_command(args, task, device):
    dataset, stage, name, settings, retries = task
    reference = (args.reference_root or args.output_root / "reference") / dataset
    common = ["--otp-root", str(args.otp_root), "--device", device, "--threads", str(args.threads)]
    if stage == "prepare":
        command = [sys.executable, "-u", str(ROOT / "scripts/prepare_otpfm.py"), *common,
                   "--dataset", dataset, "--output-dir", str(reference),
                   "--wandb-mode", args.wandb_mode, "--wandb-project", "flow2pc",
                   "--ode-comparison" if args.ode_comparison else "--no-ode-comparison"]
        if args.wandb_entity:
            command.extend(["--wandb-entity", args.wandb_entity])
        return command
    sweep = args.output_root / "sweeps" / dataset
    return [sys.executable, "-u", str(ROOT / "scripts/train_otpfm.py"), *common,
            "--reference-dir", str(reference), "--output-dir", str(sweep / "runs" / name),
            "--config", str(sweep / "configs" / f"{name}.json"), "--resume"]


def summarize(args, plans):
    rows = []
    for dataset, plan in plans.items():
        candidates = []
        for name, settings in plan["jobs"]:
            path = args.output_root / "sweeps" / dataset / "runs" / name
            status = read_json(path / "status.json")
            metrics = read_json(path / "eval/metrics.json")
            if not status:
                continue
            row = {"dataset": dataset, "run": name, "status": status["status"],
                   **{key: settings[key] for key in ("num_latents", "time_embedding_dim", "lr", "batch_size", "seed", "steps")},
                   "best_validation_mse": status.get("best_validation_mse", ""),
                   "selected_checkpoint_step": metrics.get("checkpoint_step", ""),
                   **{key: metrics.get(key, "") for key in (
                       "student_direct_holdout_w2", "teacher_finite_holdout_w2", "interpolant_holdout_w2",
                       "student_ode_holdout_w2", "teacher_ode_holdout_w2", "student_direct_source_w2")}}
            rows.append(row)
            if status["status"] == "complete" and metrics:
                candidates.append(row)
        if candidates:
            winner = min(candidates, key=lambda row: row["best_validation_mse"])
            atomic_json(args.output_root / "sweeps" / dataset / "selected_student.json", {
                "selection_rule": "minimum source-row-validation velocity MSE; holdout W2 unused",
                "checkpoint": str(args.output_root / "sweeps" / dataset / "runs" / winner["run"] / "checkpoints/best.pt"),
                "metrics": winner})
    if rows:
        with (args.output_root / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def run(args, plans, devices):
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = (args.output_root / ".sweep.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = {"grid_sha256": sha256_file(args.grid), "steps_override": args.steps,
               "ode_comparison": args.ode_comparison,
               "reference_root": str(args.reference_root or args.output_root / "reference"),
               "student_code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in
                   [ROOT / "scripts/train_otpfm.py", ROOT / "repc/otpfm_student.py", ROOT / "repc/otpfm_eval.py"]},
               "teachers": {ds: {"checkpoint_sha256": plan["teacher"]["checkpoint_sha256"],
                                  "config_sha256": plan["teacher"]["config_sha256"],
                                  "jobs": plan["jobs"]} for ds, plan in plans.items()}}
    manifest_path = args.output_root / "sweep_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != json.loads(json.dumps(binding)):
        raise ValueError("sweep inputs changed; use a new output root")
    atomic_json(manifest_path, binding)
    pending = {dataset: deque([(dataset, "prepare", "reference", {}, 0)]) for dataset in plans}
    order = deque(plans)
    active, failures = [], []
    interrupted = False

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}

    def status(state):
        atomic_json(args.output_root / "sweep_status.json", {
            "status": state, "failures": failures,
            "pending_tasks": sum(len(queue) for queue in pending.values()),
            "active": [{"dataset": w["task"][0], "stage": w["task"][1], "name": w["task"][2],
                        "pid": w["process"].pid, "device": w["device"]} for w in active]})
        summarize(args, plans)

    try:
        report_at = 0
        memory_error_report_at = 0
        while active or any(pending.values()):
            try:
                free = gpu_free_memory(devices)
            except RuntimeError as error:
                # Keep polling existing workers, pause launches, and retry the
                # monitor instead of killing the entire sweep on a query error.
                free = dict.fromkeys(devices, 0)
                if time.monotonic() >= memory_error_report_at:
                    print(f"GPU monitor unavailable; pausing new workers: {error}", flush=True)
                    memory_error_report_at = time.monotonic() + 30
            while any(pending.values()):
                device = choose_device(devices, active, free, args.workers_per_gpu, args.min_free_gb, time.monotonic())
                if device is None:
                    break
                while not pending[order[0]]:
                    order.rotate(-1)
                task = pending[order[0]].popleft()
                order.rotate(-1)
                dataset, stage, name, settings, retries = task
                if stage == "train":
                    atomic_json(args.output_root / "sweeps" / dataset / "configs" / f"{name}.json", settings)
                log_path = args.output_root / "logs" / f"{dataset}-{name}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("a")
                offset = log.tell()
                env = {**os.environ, "OMP_NUM_THREADS": str(args.threads),
                       "MKL_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": str(args.threads),
                       "FLOW2PC_WANDB_INIT_LOCK": str(args.output_root / ".wandb-init.lock")}
                for key in ("WANDB_RUN_ID", "WANDB_NAME", "WANDB_RESUME"):
                    env.pop(key, None)
                try:
                    process = subprocess.Popen(task_command(args, task, device), stdout=log,
                        stderr=subprocess.STDOUT, cwd=ROOT, env=env, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                active.append({"task": task, "process": process, "device": device, "log": log,
                               "path": log_path, "offset": offset, "started": time.monotonic()})
                print(f"Started {dataset}/{name} [{stage}] on {device}; pid={process.pid}", flush=True)
            for worker in active[:]:
                code = worker["process"].poll()
                if code is None:
                    continue
                active.remove(worker)
                worker["log"].close()
                dataset, stage, name, settings, retries = worker["task"]
                print(f"Finished {dataset}/{name} [{stage}] exit={code}", flush=True)
                if code == 0 and stage == "prepare":
                    for job_name, job_settings in plans[dataset]["jobs"]:
                        output = args.output_root / "sweeps" / dataset / "runs" / job_name
                        if read_json(output / "status.json").get("status") == "complete" and (output / "eval/metrics.json").exists():
                            continue
                        pending[dataset].append((dataset, "train", job_name, job_settings, 0))
                elif code:
                    try:
                        os.killpg(worker["process"].pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    with worker["path"].open(errors="replace") as stream:
                        stream.seek(worker["offset"])
                        error = stream.read().lower()
                    if any(msg in error for msg in ("out of memory", "cublas_status_alloc_failed")) and retries < 1:
                        pending[dataset].append((dataset, stage, name, settings, retries + 1))
                    else:
                        failures.append({"dataset": dataset, "name": name, "stage": stage,
                                         "exit_code": code, "log": str(worker["path"])})
                status("running")
            if time.monotonic() >= report_at:
                status("running")
                print(f"Active workers={len(active)}; pending={sum(len(queue) for queue in pending.values())}", flush=True)
                report_at = time.monotonic() + 30
            if active or any(pending.values()):
                time.sleep(2)
    except BaseException:
        interrupted = True
        raise
    finally:
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            stop_workers(active)
            active.clear()
            status("interrupted" if interrupted else "failed" if failures else "complete")
        finally:
            for sig, previous in handlers.items():
                signal.signal(sig, previous)
            lock.close()
    if failures:
        raise SystemExit(f"Some jobs failed: {args.output_root / 'sweep_status.json'}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="gom,beijing")
    parser.add_argument("--otp-root", type=Path, default=DEFAULT_OTP_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("/data/lthung/flow2pc/otp-fm/ali-recipe"))
    parser.add_argument("--reference-root", type=Path,
                        help="reuse existing frozen-teacher references outside the student output root")
    parser.add_argument("--student-recipe", choices=("ali", "anchored"), default="ali")
    parser.add_argument("--grid", type=Path, default=ROOT / "configs/otpfm/sweep.json")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--min-free-gb", type=float, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--ode-comparison", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers_per_gpu < 1 or args.min_free_gb <= 0 or args.threads < 1 or (args.steps is not None and args.steps < 1):
        parser.error("invalid worker or step counts")
    args.otp_root, args.output_root, args.grid = args.otp_root.resolve(), args.output_root.resolve(), args.grid.resolve()
    if args.reference_root is not None:
        args.reference_root = args.reference_root.resolve()
    return args


def main():
    args = parse_args()
    plans = build_plan(args)
    count = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(count)] if args.devices == "auto" else args.devices.split(",")
    if (len(devices) < 2 or len(set(devices)) != len(devices)
            or any(not re.fullmatch(r"cuda:\d+", device) for device in devices)
            or any(int(device.split(":")[1]) >= count for device in devices)):
        raise SystemExit("select at least two distinct visible CUDA devices; no single-GPU/CPU fallback")
    print(json.dumps({"output_root": str(args.output_root), "wandb_project": "flow2pc",
                      "reference_root": str(args.reference_root or args.output_root / "reference"),
                      "devices": devices, "maximum_parallel_workers": len(devices) * args.workers_per_gpu,
                      "teacher_retraining": False, "dry_run": args.dry_run,
                      "sweeps": [{"dataset": ds, "checkpoint": str(args.otp_root / plan["teacher"]["checkpoint"]),
                                  "teacher_seed": plan["teacher"]["seed"],
                                  "fixed_alpha": plan["teacher"]["fixed_interpolant_alpha"],
                                  "student_runs": len(plan["jobs"]),
                                  "example_wandb_name": f"otpfm-{ds}-{plan['jobs'][0][0]}"}
                                 for ds, plan in plans.items()]}, indent=2), flush=True)
    if not args.dry_run:
        run(args, plans, devices)


if __name__ == "__main__":
    main()
