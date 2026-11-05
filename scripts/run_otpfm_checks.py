#!/usr/bin/env python3
"""Common parallel native/migrated OTP-FM checks, with metadata-only preview."""

import argparse
from collections import deque
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from flow2pc.utils.config import load_config  # noqa: E402
from flow2pc.utils.gpu_queue import choose_device, stop_workers  # noqa: E402
from flow2pc.pretrain.models.otpfm.provenance import verify_source  # noqa: E402
from run_alicfm_checks import atomic_json, gpu_inventory, selected_devices  # noqa: E402


def tasks_for(args):
    tasks = []
    for seed in args.seeds:
        for recipe in args.methods:
            for dataset in args.datasets:
                config = load_config(ROOT / f"configs/pretrain/otpfm/otpfm_{dataset}_{recipe.lower()}.yaml")
                output = args.output_root / args.backend / dataset / recipe / f"seed{seed}"
                config["seed"] = seed
                config["data"]["root"] = str(args.data_root)
                config["output"]["dir"] = str(output)
                native = config["model"]["params"]["native_config"]
                native.update(seed=seed, data_dir=str(args.data_root / dataset))
                if args.epochs is not None:
                    native["epochs"] = args.epochs
                config["logging"].update(mode=args.mode, name=f"otpfm-{args.backend}-{dataset}-{recipe}-h{native['hidden_dim']}-lr{native['lr']:g}-seed{seed}")
                if not (args.data_root / dataset / "marginals.npz").is_file():
                    raise FileNotFoundError(f"Shared data bundle missing: {args.data_root / dataset}")
                if args.backend == "native":
                    required = args.data_root / dataset / ("GoMvortex_data.npy" if dataset == "gom" else "dingling_pm25_13marginals.npy")
                    if not required.is_file():
                        raise FileNotFoundError(f"Native cache missing; downloads disabled: {required}")
                status = "pending"
                if output.exists():
                    marker = output / ("result.json" if args.backend == "native" else "status.json")
                    complete = marker.is_file() and (args.backend == "native" or json.loads(marker.read_text()).get("status") == "complete")
                    saved_path = output / ("config.json" if args.backend == "native" else "resolved_config.yaml")
                    saved = (json.loads(saved_path.read_text()) if args.backend == "native" else yaml.safe_load(saved_path.read_text())) if saved_path.is_file() else None
                    expected = native if args.backend == "native" else config
                    if args.backend == "migrated" and saved is not None:
                        saved = {k: v for k, v in saved.items() if k not in ("logging", "output")}
                        expected = {k: v for k, v in expected.items() if k not in ("logging", "output")}
                    if saved != expected:
                        raise ValueError(f"Saved OTP-FM config differs or is missing: {output}; choose a new --output-root")
                    status = "complete" if complete else "blocked"
                tasks.append({"dataset": dataset, "recipe": recipe, "seed": seed, "output": output,
                              "config": config, "status": status, "name": f"{dataset}-{recipe}-s{seed}"})
    return tasks


def command_for(args, task, config_path):
    if args.backend == "migrated":
        return [sys.executable, str(ROOT / "main.py"), "--config", str(config_path)]
    return [sys.executable, str(ROOT / "scripts/run_native_otpfm_control.py"),
            "--native-root", str(args.native_root), "--dataset", task["dataset"],
            "--recipe", task["recipe"], "--seed", str(task["seed"]), "--device", "cuda:0",
            "--config", str(config_path), "--output", str(task["output"]),
            "--threads", str(args.threads), "--mode", args.mode]


def run_queue(args, tasks, devices, lock):
    pending = deque(task for task in tasks if task["status"] == "pending")
    active, interrupted, previous = [], False, {}
    queue_failed = False
    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, stop)
    def status():
        atomic_json(args.output_root / args.backend / "_launcher/status.json", {
            "status": "interrupted" if interrupted else "failed" if queue_failed else "running" if pending or active else "finished",
            "active": [{"run": w["task"]["name"], "pid": w["process"].pid, "gpu": w["device"]} for w in active],
            "tasks": [{k: task[k] for k in ("name", "status")} for task in tasks],
        })
    try:
        while pending or active:
            if interrupted:
                break
            for worker in active[:]:
                code = worker["process"].poll()
                if code is not None:
                    worker["log"].close()
                    active.remove(worker)
                    worker["task"]["status"] = "complete" if code == 0 else "failed"
                    print(worker["task"]["name"], worker["task"]["status"], flush=True)
            try:
                inventory = gpu_inventory()
            except RuntimeError as error:
                print(f"GPU query unavailable; retaining workers and retrying: {error}", flush=True)
                time.sleep(5)
                continue
            free = {d: inventory[d]["free"] for d in devices}
            while pending and len(active) < args.max_parallel:
                if interrupted:
                    break
                device = choose_device(devices, active, free, 1, args.min_free_gb, time.monotonic())
                if device is None:
                    break
                task = pending.popleft()
                base = args.output_root / args.backend / "_launcher"
                (base / "configs").mkdir(parents=True, exist_ok=True)
                (base / "logs").mkdir(parents=True, exist_ok=True)
                config_path = base / "configs" / f"{task['name']}.{'yaml' if args.backend == 'migrated' else 'json'}"
                config_path.write_text(yaml.safe_dump(task["config"], sort_keys=False) if args.backend == "migrated" else json.dumps(task["config"]["model"]["params"]["native_config"], indent=2))
                log = (base / "logs" / f"{task['name']}.log").open("a")
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": inventory[device]["uuid"],
                       "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONPATH": str(ROOT / "src"),
                       "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                       "FLOW2PC_WANDB_INIT_LOCK": str(args.output_root / ".wandb-init.lock")}
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    env[name] = str(args.threads)
                for name in ("WANDB_RUN_ID", "WANDB_NAME", "WANDB_RESUME"):
                    env.pop(name, None)
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                try:
                    process = subprocess.Popen(command_for(args, task, config_path), cwd=ROOT, env=env,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock.fileno(),))
                except BaseException:
                    log.close()
                    raise
                task["status"] = "running"
                active.append({"task": task, "device": device, "process": process, "log": log, "started": time.monotonic()})
                print(f"Started {args.backend}/{task['name']} GPU {device} pid {process.pid}", flush=True)
            status()
            time.sleep(2)
    except BaseException:
        queue_failed = True
        raise
    finally:
        if active:
            stop_workers(active)
            for worker in active:
                worker["task"]["status"] = "interrupted"
            active.clear()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        status()
    return 130 if interrupted else (1 if any(task["status"] in ("failed", "blocked") for task in tasks) else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("migrated", "native"), default="migrated")
    parser.add_argument("--datasets", nargs="+", default=["gom", "beijing"])
    parser.add_argument("--methods", nargs="+", choices=("W2", "W2Inf", "MMD", "KL"), default=["W2", "W2Inf"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--min-free-gb", type=float, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=Path, default=Path("/data/lthung/otp-fm"))
    parser.add_argument("--native-root", type=Path, default=ROOT.parent / "otp-fm")
    parser.add_argument("--output-root", type=Path, default=Path("/data/lthung/flow2pc/otp-fm/migration-check"))
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--epochs", type=int, help="Explicit debug budget override; changes native curriculum and is not a full reproduction")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (args.max_parallel < 2 or args.threads < 1 or not math.isfinite(args.min_free_gb) or args.min_free_gb <= 0
            or args.epochs is not None and args.epochs < 1
            or any(not 0 <= seed < 2**32 for seed in args.seeds)
            or any(len(set(getattr(args, key))) != len(getattr(args, key)) for key in ("datasets", "methods", "seeds"))):
        parser.error("Invalid concurrency, epoch, seed or repeated task selection")
    args.data_root, args.native_root, args.output_root = args.data_root.resolve(), args.native_root.resolve(), args.output_root.resolve()
    verify_source(args.native_root if args.backend == "native" else None)
    tasks = tasks_for(args)
    inventory = gpu_inventory()
    devices = selected_devices(args.gpus, inventory)
    args.max_parallel = min(args.max_parallel, len(devices))
    print(json.dumps({"dry_run": args.dry_run, "backend": args.backend, "tasks": len(tasks),
                      "gpus": devices, "max_parallel": args.max_parallel, "wandb_project": "flow2pc",
                      "output_root": str(args.output_root), "runs": [{k: t[k] for k in ("name", "status")} for t in tasks]}, indent=2), flush=True)
    if args.dry_run:
        return 0
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".checks.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"An active native/migrated batch holds {args.output_root / '.checks.lock'}; wait for that batch to exit before starting another against this root")
        blocked = [task["name"] for task in tasks if task["status"] == "blocked"]
        if blocked:
            parser.error(f"Preserving incomplete run directories in {args.output_root}: {', '.join(blocked)}. Select a fresh --output-root; no jobs were started")
        return run_queue(args, tasks, devices, lock)


if __name__ == "__main__":
    raise SystemExit(main())
