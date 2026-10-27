"""Prepare one frozen Cell4 reference, then schedule PC students across GPUs."""
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from repc.cell_tracking import ROOT, DEFAULT_ALI_ROOT, EXPECTED_SHA256, input_identity, code_identity
from repc.single_cell import atomic_json, sha256_file
from scripts.run_otpfm_sweep import read_json, gpu_free_memory
from scripts.run_single_cell_sweep import load_jobs
from scripts.run_single_cell_all_sweeps import choose_device, stop_workers


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ali-root", type=Path, default=DEFAULT_ALI_ROOT)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_ALI_ROOT / "changed_files/ali_cfm_model.pth")
    p.add_argument("--expected-sha256", default=EXPECTED_SHA256)
    p.add_argument("--mask-stack", type=Path, default=DEFAULT_ALI_ROOT / "data/cell_tracking/exports/Cell4_masks/mask_cell4_stack.npy")
    p.add_argument("--image-dir", type=Path, default=DEFAULT_ALI_ROOT / "data/cell_tracking/PhC-C2DH-U373/01")
    p.add_argument("--output-root", type=Path, default=Path("/data/lthung/flow2pc/cell_tracking/distillation_sweep"))
    p.add_argument("--grid", type=Path, default=ROOT / "configs/cell_tracking/sweep.json")
    p.add_argument("--devices", default="auto")
    p.add_argument("--workers-per-gpu", type=int, default=2)
    p.add_argument("--min-free-gb", type=float, default=4)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--steps", type=int)
    p.add_argument("--metric-samples", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--ode-tolerance", type=float, default=1e-5)
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if min(args.workers_per_gpu, args.threads, args.metric_samples, args.eval_batch_size) < 1 or args.min_free_gb <= 0 or args.ode_tolerance <= 0 or (args.steps is not None and args.steps < 1):
        p.error("invalid worker/evaluation settings")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_sha256): p.error("invalid checkpoint SHA-256")
    for key in ("ali_root", "checkpoint", "mask_stack", "image_dir", "output_root", "grid"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def build_plan(args):
    overrides = {"wandb_mode": args.wandb_mode, "wandb_project": "flow2pc",
        "wandb_entity": args.wandb_entity, "wandb_service_wait": 300,
        "wandb_group": f"cell-tracking-teacher-{args.expected_sha256[:12]}"}
    if args.steps is not None: overrides["steps"] = args.steps
    jobs = load_jobs(args.grid, overrides)
    allowed = {"num_latents", "time_embedding_dim", "lr", "batch_size", "steps", "seed", "validate_every", "checkpoint_every", "log_every"} | set(overrides)
    for _, settings in jobs:
        if set(settings) - allowed: raise ValueError(f"unsupported student settings: {sorted(set(settings) - allowed)}")
        for k in ("num_latents", "time_embedding_dim", "batch_size", "steps", "validate_every", "checkpoint_every", "log_every"):
            if not isinstance(settings[k], int) or settings[k] < 1: raise ValueError(f"invalid {k}")
        if settings["num_latents"] < 2 or settings["time_embedding_dim"] % 2 or settings["lr"] <= 0 or settings["seed"] < 0:
            raise ValueError("invalid PC configuration")
    identity = input_identity(args.ali_root, args.checkpoint, args.mask_stack, args.image_dir, args.expected_sha256)
    return {"jobs": jobs, "identity": identity}


def task_command(args, task, device):
    stage, name, settings, retries = task
    common = ["--ali-root", str(args.ali_root), "--checkpoint", str(args.checkpoint),
              "--expected-sha256", args.expected_sha256, "--device", device, "--threads", str(args.threads)]
    reference = args.output_root / "reference"
    if stage == "prepare":
        command = [sys.executable, "-u", str(ROOT / "scripts/prepare_cell_tracking.py"), *common,
            "--mask-stack", str(args.mask_stack), "--image-dir", str(args.image_dir), "--output-dir", str(reference),
            "--metric-samples", str(args.metric_samples), "--eval-batch-size", str(args.eval_batch_size),
            "--ode-tolerance", str(args.ode_tolerance), "--wandb-mode", args.wandb_mode, "--wandb-project", "flow2pc"]
        if args.wandb_entity: command.extend(["--wandb-entity", args.wandb_entity])
        return command
    return [sys.executable, "-u", str(ROOT / "scripts/train_cell_tracking.py"), *common,
        "--reference-dir", str(reference), "--output-dir", str(args.output_root / "runs" / name),
        "--config", str(args.output_root / "configs" / f"{name}.json"), "--resume"]


def completed(path):
    return (read_json(path / "status.json").get("status") == "complete" and
        all((path / "eval" / name).is_file() for name in ("metrics.json", "student_overlay.png", "comparison_overlay.png")))


def summarize(args, plan):
    rows = []
    for name, settings in plan["jobs"]:
        run = args.output_root / "runs" / name
        status = read_json(run / "status.json")
        if not status: continue
        metrics = read_json(run / "eval/metrics.json")
        row = {"run": name, "status": status["status"], **{k:settings[k] for k in
            ("num_latents", "time_embedding_dim", "lr", "batch_size", "steps", "seed")},
            "best_selection_score": status.get("best_selection_score", ""),
            **{k:metrics.get(k, "") for k in ("student_display_mean_emd_pixels", "teacher_display_mean_emd_pixels",
                "student_final_emd_pixels", "teacher_final_emd_pixels", "final_paired_rmse_pixels", "late_paired_rmse_pixels")}}
        rows.append(row)
    if rows:
        with (args.output_root / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        candidates = [r for r in rows if r["status"] == "complete"]
        if candidates:
            best = min(candidates, key=lambda r:r["best_selection_score"])
            atomic_json(args.output_root / "selected_student.json", {"selection_rule": "worst global/late/final teacher-state velocity error; real-mask EMD unused",
                "checkpoint": str(args.output_root / "runs" / best["run"] / "checkpoints/best.pt"),
                "overlay": str(args.output_root / "runs" / best["run"] / "eval/student_overlay.png"), "metrics": best})


def run(args, plan, devices):
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = (args.output_root / ".sweep.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    binding = {"grid_sha256": sha256_file(args.grid), "identity": plan["identity"],
        "code_sha256": code_identity(), "jobs": plan["jobs"], "steps_override": args.steps,
        "metric_samples": args.metric_samples, "eval_batch_size": args.eval_batch_size, "ode_tolerance": args.ode_tolerance}
    manifest = args.output_root / "sweep_manifest.json"
    if manifest.exists() and read_json(manifest) != json.loads(json.dumps(binding)):
        raise ValueError("sweep code/inputs changed; choose a new output root")
    atomic_json(manifest, binding)
    pending = deque([("prepare", "reference", {}, 0)])
    active, failures, cooldown = [], [], {}
    interrupted = False

    def interrupt(signum, frame): raise KeyboardInterrupt(f"received signal {signum}")
    handlers = {sig:signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}

    def status(state):
        atomic_json(args.output_root / "sweep_status.json", {"status":state,"pending_tasks":len(pending),"failures":failures,
            "active":[{"stage":w["task"][0],"name":w["task"][1],"pid":w["process"].pid,"device":w["device"]} for w in active]})
        summarize(args, plan)

    try:
        report_at = monitor_report_at = 0
        while active or pending:
            now = time.monotonic()
            try: free = gpu_free_memory(devices)
            except RuntimeError as error:
                free = dict.fromkeys(devices, 0)
                if now >= monitor_report_at:
                    print(f"GPU monitor unavailable; pausing launches: {error}", flush=True); monitor_report_at = now + 30
            free = {d:0 if cooldown.get(d, 0) > now else b for d,b in free.items()}
            while pending:
                device = choose_device(devices, active, free, args.workers_per_gpu, args.min_free_gb, time.monotonic())
                if device is None: break
                task = pending.popleft(); stage, name, settings, retries = task
                if stage == "train": atomic_json(args.output_root / "configs" / f"{name}.json", settings)
                log_path = args.output_root / "logs" / f"{name}.log"; log_path.parent.mkdir(parents=True, exist_ok=True)
                log = log_path.open("a"); offset = log.tell()
                env = {**os.environ, "OMP_NUM_THREADS":str(args.threads), "MKL_NUM_THREADS":str(args.threads),
                    "OPENBLAS_NUM_THREADS":str(args.threads), "FLOW2PC_WANDB_INIT_LOCK":str(args.output_root / ".wandb-init.lock")}
                for key in ("WANDB_RUN_ID", "WANDB_NAME", "WANDB_RESUME"): env.pop(key, None)
                try:
                    process = subprocess.Popen(task_command(args, task, device), stdout=log, stderr=subprocess.STDOUT,
                        cwd=ROOT, env=env, start_new_session=True)
                except BaseException: log.close(); raise
                active.append({"task":task,"process":process,"device":device,"log":log,"path":log_path,
                               "offset":offset,"started":time.monotonic()})
                print(f"Started cell-tracking/{name} [{stage}] on {device}; pid={process.pid}", flush=True)
            for worker in active[:]:
                code = worker["process"].poll()
                if code is None: continue
                active.remove(worker); worker["log"].close()
                stage,name,settings,retries = worker["task"]
                print(f"Finished {name} [{stage}] exit={code}", flush=True)
                if code == 0 and stage == "prepare":
                    pending.extend(("train",n,s,0) for n,s in plan["jobs"] if not completed(args.output_root / "runs" / n))
                elif code:
                    try: os.killpg(worker["process"].pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                    with worker["path"].open(errors="replace") as f: f.seek(worker["offset"]); error=f.read().lower()
                    oom = any(t in error for t in ("out of memory", "cublas_status_alloc_failed"))
                    transient = any(t in error for t in ("context canceled", "context deadline exceeded", "http 500", "http 502", "http 503", "connection reset", "timed out"))
                    if retries < 2 and (oom or transient):
                        if oom: cooldown[worker["device"]] = time.monotonic() + 60
                        pending.append((stage,name,settings,retries+1))
                    else: failures.append({"stage":stage,"name":name,"exit_code":code,"log":str(worker["path"])})
                status("running")
            if time.monotonic() >= report_at:
                status("running"); print(f"Active={len(active)}; pending={len(pending)}; failures={len(failures)}", flush=True)
                report_at = time.monotonic() + 30
            if active or pending: time.sleep(2)
    except BaseException: interrupted = True; raise
    finally:
        for sig in handlers: signal.signal(sig, signal.SIG_IGN)
        try:
            stop_workers(active); active.clear(); status("interrupted" if interrupted else "failed" if failures else "complete")
        finally:
            for sig,handler in handlers.items(): signal.signal(sig, handler)
            lock.close()
    if failures: raise SystemExit(f"Some jobs failed: {args.output_root / 'sweep_status.json'}")


def main():
    args = parse_args(); plan = build_plan(args)
    count = torch.cuda.device_count()
    devices = [f"cuda:{i}" for i in range(count)] if args.devices == "auto" else [d.strip() for d in args.devices.split(",")]
    if len(devices) < 2 or len(set(devices)) != len(devices) or any(not re.fullmatch(r"cuda:\d+",d) or int(d.split(":")[1]) >= count for d in devices):
        raise SystemExit("select at least two distinct visible CUDA devices; no single-GPU fallback")
    print(json.dumps({"dataset":"cell-tracking","student_runs":len(plan["jobs"]),"teacher_retraining":False,
        "checkpoint":str(args.checkpoint),"checkpoint_sha256":args.expected_sha256,"output_root":str(args.output_root),
        "devices":devices,"maximum_parallel_workers":len(devices)*args.workers_per_gpu,"wandb_project":"flow2pc",
        "dry_run":args.dry_run,"example_wandb_name":f"cell-tracking-{plan['jobs'][0][0]}",
        "final_images":["eval/student_overlay.png","eval/comparison_overlay.png"]}, indent=2), flush=True)
    if not args.dry_run: run(args, plan, devices)


if __name__ == "__main__": main()
