"""Six t=1 sweeps sharing a fair, bounded multi-GPU worker pool.

Teacher references are independent prerequisites. As each reference completes,
its students can start while other datasets are still being evaluated. A worker
runs training followed by evaluation, and returns its GPU slot to the shared
pool. --dry-run checks all six bundles and prints the plan without subprocesses,
W&B initialization, output writes, or training/evaluation.
"""

import argparse
from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import dataclass, field
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

from repc.single_cell import atomic_json, sha256_file
from repc.single_cell_tracking import dataset_run_name
from scripts.run_single_cell_sweep import load_jobs, summarize

EXPECTED = {(dataset, dim) for dataset in ("cite", "multi") for dim in (5, 50, 100)}


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


@dataclass
class Dataset:
    tag: str
    teacher: Path
    metadata: dict
    contract: dict
    checkpoint_hash: str
    output: Path
    reference: Path
    jobs: list
    pending: deque = field(default_factory=deque)
    failures: list = field(default_factory=list)


@dataclass
class Task:
    dataset: Dataset
    stage: str
    name: str = "teacher"
    settings: dict = field(default_factory=dict)
    retries: int = 0


def discover_datasets(args):
    """Require the six indexed t=1 combined bundles, never infer seed from paths."""
    index = read_json(args.teachers_root / "index.json")
    entries = [row for row in index["teachers"] if row["heldout_index"] == 1]
    keys = [(row["dataset"], row["dim"]) for row in entries]
    if len(keys) != 6 or set(keys) != EXPECTED:
        raise ValueError("index.json must contain exactly one t=1 teacher for each of the six datasets")
    datasets = []
    for entry in entries:
        folder = args.teachers_root / f"{entry['dataset']}_{entry['dim']}D" / "t=1"
        teacher = folder / "teacher.pth"
        metadata = read_json(folder / "metadata.json")
        digest = sha256_file(teacher)
        if digest != metadata["export_sha256"]["teacher.pth"]:
            raise ValueError(f"export checksum mismatch: {teacher}")
        checkpoint = torch.load(teacher, map_location="cpu", weights_only=False)
        contract = checkpoint["data_contract"]
        prep = contract["preprocessing"]
        if (contract["time_index"] != 1 or contract["heldout_day"] != 3
                or prep["dataset"] != entry["dataset"]
                or prep["n_pca_components_used"] != entry["dim"]
                or metadata["seed"] != entry["seed"]
                or metadata["heldout_index"] != 1
                or metadata["data_contract"] != contract
                or checkpoint.get("time_format") != "global"
                or contract["pseudo_time_convention"] != "equally_spaced_ordered_marginals_on_[0,1]"
                or prep["time_order"] != [2, 3, 4, 7]
                or bool(prep["whiten"]) != (entry["dim"] == 5)):
            raise ValueError(f"teacher index/metadata/data contract mismatch: {teacher}")
        for key in ("ot_cfm_model", "interpolant"):
            if not checkpoint.get(key):
                raise ValueError(f"combined teacher missing {key}: {teacher}")
        del checkpoint
        prefix = f"{entry['dataset']}{entry['dim']}d"
        tag = f"{prefix}-t1-teacher{entry['seed']}"
        overrides = {"wandb_mode": args.wandb_mode, "wandb_project": args.wandb_project,
                     "wandb_group": tag, "wandb_entity": args.wandb_entity,
                     "wandb_service_wait": args.wandb_service_wait}
        overrides.update({key: getattr(args, key) for key in ("steps", "emd_every", "emd_samples")
                          if getattr(args, key) is not None})
        datasets.append(Dataset(tag, teacher, metadata, contract, digest,
                                args.output_root / "sweeps" / tag,
                                args.output_root / "reference" / tag / "metrics.json",
                                load_jobs(args.grid, overrides)))
    return datasets


def evaluation_protocol(args, dataset):
    return {"num_samples": args.eval_samples, "seed": args.eval_seed, "source_index": 0,
            "solver": "dopri5", "tolerance": args.eval_tolerance,
            "batch_size": args.eval_batch_size,
            "metric_space": ("inverse_minmax_standardized_pca" if
                             dataset.contract["preprocessing"]["whiten"] else "inverse_minmax_pca")}


def verify_reference(args, dataset):
    cached = read_json(dataset.reference)
    if (cached.get("model") != "teacher"
            or cached.get("identity", {}).get("checkpoint_sha256") != dataset.checkpoint_hash
            or cached.get("data_contract") != dataset.contract
            or cached.get("protocol") != evaluation_protocol(args, dataset)):
        raise ValueError(f"reference differs from this bundle/protocol; use a new output root: {dataset.reference}")
    return cached


def prepare_students(args, dataset):
    reference = verify_reference(args, dataset)
    binding = {"identity": reference["identity"], "reference": str(dataset.reference),
               "reference_sha256": sha256_file(dataset.reference), "protocol": reference["protocol"]}
    binding_path = dataset.output / "sweep_inputs.json"
    previous = read_json(binding_path)
    if previous and previous != binding:
        raise ValueError(f"sweep inputs changed; use a new output root: {dataset.output}")
    atomic_json(binding_path, binding)
    atomic_json(dataset.output / "sweep_manifest.json", {
        "grid": str(args.grid), "grid_sha256": sha256_file(args.grid),
        "num_jobs": len(dataset.jobs), "teacher_seed": dataset.metadata["seed"],
        "jobs": [{"name": name, "settings": settings} for name, settings in dataset.jobs]})
    for name, settings in dataset.jobs:
        run = dataset.output / "runs" / name
        trained = read_json(run / "status.json").get("status") == "complete"
        evaluation = read_json(run / "eval/metrics.json")
        if trained and read_json(run / "eval/status.json").get("status") == "complete" and evaluation:
            if (evaluation.get("identity") != reference["identity"]
                    or evaluation.get("protocol") != reference["protocol"]
                    or evaluation.get("checkpoint_step") is None):
                raise ValueError(f"cached student evaluation differs: {run}")
            continue
        dataset.pending.append(Task(dataset, "evaluate" if trained else "train", name, settings))


def command_for(args, task, device):
    ds = task.dataset
    common = ["--teacher-checkpoint", str(ds.teacher), "--data-dir", str(args.data_dir),
              "--device", device, "--threads", str(args.threads)]
    tracking = ["--wandb-mode", args.wandb_mode, "--wandb-project", args.wandb_project,
                "--wandb-group", ds.tag, "--wandb-service-wait", str(args.wandb_service_wait)]
    if args.wandb_entity:
        tracking.extend(["--wandb-entity", args.wandb_entity])
    run = ds.output / "runs" / task.name
    if task.stage == "train":
        return [sys.executable, "-u", str(ROOT / "scripts/train_single_cell.py"), *common,
                "--config", str(ds.output / "configs" / f"{task.name}.json"),
                "--reference", str(ds.reference), "--output-dir", str(run), "--resume"]
    command = [sys.executable, "-u", str(ROOT / "scripts/evaluate_single_cell.py"), *common,
               "--num-samples", str(args.eval_samples), "--seed", str(args.eval_seed),
               "--batch-size", str(args.eval_batch_size), "--tolerance", str(args.eval_tolerance),
               *tracking]
    if task.stage == "teacher":
        command.extend(["--output-dir", str(ds.reference.parent)])
    else:
        command.extend(["--reference", str(ds.reference),
                        "--student-checkpoint", str(run / "checkpoints/best.pt"),
                        "--output-dir", str(run / "eval"), "--wandb-run-dir", str(run)])
    return command


def choose_device(devices, active, free_bytes, workers_per_gpu, min_free_gb, now):
    """Balance workers across GPUs, reserving memory while new workers warm up."""
    counts = Counter(worker["device"] for worker in active)
    required = min_free_gb * 2**30
    candidates = []
    for device in devices:
        # Avoid launching a burst before CUDA contexts/allocations become visible.
        warming = sum(worker["device"] == device and now - worker["started"] < 30
                      for worker in active)
        available = free_bytes[device] - warming * required
        if counts[device] < workers_per_gpu and available >= required:
            candidates.append((counts[device], -available, device))
    return min(candidates)[2] if candidates else None


def next_task(datasets):
    """One task per dataset per round; finished datasets do not strand GPUs."""
    for _ in range(len(datasets)):
        dataset = datasets[0]
        datasets.rotate(-1)
        if dataset.pending:
            return dataset.pending.popleft()
    return None


def child_environment(args):
    env = {**os.environ, "OMP_NUM_THREADS": str(args.threads),
           "MKL_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": str(args.threads),
           "FLOW2PC_WANDB_INIT_LOCK": str(args.output_root / ".wandb-init.lock")}
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for key in ("WANDB_RUN_ID", "WANDB_NAME", "WANDB_RESUME"):
        env.pop(key, None)
    return env


def start_worker(args, task, device):
    dataset = task.dataset
    log_path = (dataset.reference.parent / "teacher.log" if task.stage == "teacher"
                else dataset.output / f"{task.name}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if task.stage != "teacher":
        (dataset.output / "runs" / task.name).mkdir(parents=True, exist_ok=True)
        atomic_json(dataset.output / "configs" / f"{task.name}.json", task.settings)
    log = log_path.open("a")
    offset = log.tell()
    try:
        process = subprocess.Popen(command_for(args, task, device), cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   env=child_environment(args), start_new_session=True)
    except BaseException:
        log.close()
        raise
    print(f"Started {dataset.tag}/{task.name} [{task.stage}] on {device}; pid={process.pid}", flush=True)
    return {"task": task, "device": device, "process": process, "log": log,
            "log_path": log_path, "log_offset": offset, "started": time.monotonic()}


def stop_workers(active):
    """Signal every process group before waiting, including SDK child services."""
    for worker in active:
        try:
            os.killpg(worker["process"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    for worker in active:
        try:
            worker["process"].wait(timeout=max(.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(worker["process"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker["process"].wait()
        worker["log"].close()


def write_status(args, datasets, status, active):
    for ds in datasets:
        summarize(ds.output, {name for name, _ in ds.jobs})
        running = any(w["task"].dataset is ds for w in active)
        ds_status = status if status == "interrupted" else (
            "running" if ds.pending or running else "failed" if ds.failures else "complete")
        atomic_json(ds.output / "sweep_status.json", {
            "status": ds_status, "reference": str(ds.reference), "failures": ds.failures})
    atomic_json(args.output_root / "all_sweeps_status.json", {
        "status": status, "pending_tasks": sum(len(ds.pending) for ds in datasets),
        "active": [{"dataset": w["task"].dataset.tag, "name": w["task"].name,
                    "stage": w["task"].stage, "device": w["device"],
                    "pid": w["process"].pid} for w in active],
        "failures": [{"dataset": ds.tag, **failure} for ds in datasets for failure in ds.failures]})


def run(args, datasets, devices):
    active = []
    fair_queue = deque(datasets)
    interrupted = False
    handlers = {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    with ExitStack() as stack:
        for directory, filename in [(args.output_root, ".six-sweeps.lock"),
                                    *((ds.output, ".sweep.lock") for ds in datasets)]:
            directory.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((directory / filename).open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(f"another launcher is using {directory}") from None
        for ds in datasets:
            if ds.reference.exists():
                verify_reference(args, ds)
            if read_json(ds.reference.parent / "status.json").get("status") == "complete":
                prepare_students(args, ds)
            else:
                ds.pending.append(Task(ds, "teacher"))
        handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            last_report = 0
            while active or any(ds.pending for ds in datasets):
                free = {device: torch.cuda.mem_get_info(int(device.split(":")[1]))[0]
                        for device in devices}
                while any(ds.pending for ds in datasets):
                    device = choose_device(devices, active, free, args.workers_per_gpu,
                                           args.min_free_gb, time.monotonic())
                    if device is None:
                        break
                    active.append(start_worker(args, next_task(fair_queue), device))
                for worker in active[:]:
                    code = worker["process"].poll()
                    if code is None:
                        continue
                    worker["log"].close()
                    active.remove(worker)
                    task = worker["task"]
                    ds = task.dataset
                    print(f"Finished {ds.tag}/{task.name} [{task.stage}] exit={code}", flush=True)
                    if code == 0:
                        if task.stage == "teacher":
                            prepare_students(args, ds)
                        elif task.stage == "train":
                            task.stage = "evaluate"
                            # Reuse the same slot; model allocations from training are gone.
                            active.append(start_worker(args, task, worker["device"]))
                    else:
                        # Clean up any remaining SDK children in this worker's session.
                        try:
                            os.killpg(worker["process"].pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        with worker["log_path"].open(errors="replace") as stream:
                            stream.seek(worker["log_offset"])
                            error = stream.read()
                        oom = any(text in error.lower() for text in (
                            "out of memory", "cublas_status_alloc_failed"))
                        if oom and task.retries < args.oom_retries:
                            task.retries += 1
                            ds.pending.append(task)
                            print(f"Requeued {ds.tag}/{task.name} after GPU allocation failure", flush=True)
                        else:
                            ds.failures.append({"name": task.name, "stage": task.stage,
                                                "exit_code": code, "log": str(worker["log_path"])})
                    summarize(ds.output, {name for name, _ in ds.jobs})
                if time.monotonic() - last_report >= 30:
                    write_status(args, datasets, "running", active)
                    print(f"Workers: {len(active)}/{len(devices) * args.workers_per_gpu}; "
                          f"pending: {sum(len(ds.pending) for ds in datasets)}", flush=True)
                    last_report = time.monotonic()
                if active or any(ds.pending for ds in datasets):
                    time.sleep(2)
        except BaseException:
            interrupted = True
            raise
        finally:
            for sig in handlers:
                signal.signal(sig, signal.SIG_IGN)
            try:
                stop_workers(active)
                write_status(args, datasets, "interrupted" if interrupted else
                             "failed" if any(ds.failures for ds in datasets) else "complete", [])
            finally:
                for sig, previous in handlers.items():
                    signal.signal(sig, previous)
    if any(ds.failures for ds in datasets):
        raise SystemExit(f"Some jobs failed; see {args.output_root / 'all_sweeps_status.json'}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teachers-root", type=Path,
                        default=Path("/data/lthung/ali-cfm/best_models/single_cell"))
    parser.add_argument("--data-dir", type=Path, default=Path("/data/lthung/single_cell"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("/data/lthung/flow2pc/outputs/single_cell/best_ali_cfm_sweeps_t1"))
    parser.add_argument("--grid", type=Path, default=ROOT / "configs/single_cell/intensive_sweep.json")
    parser.add_argument("--devices", default="auto", help="all visible GPUs, or e.g. cuda:0,cuda:1,cuda:2")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--min-free-gb", type=float, default=8, help="free GiB required to start a worker")
    parser.add_argument("--oom-retries", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--emd-every", type=int)
    parser.add_argument("--emd-samples", type=int)
    parser.add_argument("--eval-samples", type=int, default=2048)
    parser.add_argument("--eval-seed", type=int, default=2026)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--eval-tolerance", type=float, default=1e-5)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-project", default="flow2pc")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-service-wait", type=float, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (args.workers_per_gpu < 1 or args.min_free_gb <= 0 or args.threads < 1
            or args.oom_retries < 0 or args.eval_samples < 0 or args.eval_batch_size < 1
            or args.eval_tolerance <= 0 or args.wandb_service_wait <= 0
            or (args.steps is not None and args.steps < 1)
            or (args.emd_every is not None and args.emd_every < 0)
            or (args.emd_samples is not None and args.emd_samples < 1)):
        parser.error("invalid worker, sample, step, or evaluation settings")
    for name in ("teachers_root", "data_dir", "output_root", "grid"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def main():
    args = parse_args()
    datasets = discover_datasets(args)
    count = torch.cuda.device_count()
    devices = ([f"cuda:{index}" for index in range(count)] if args.devices == "auto"
               else [device.strip() for device in args.devices.split(",")])
    if (len(set(devices)) != len(devices) or len(devices) < 2
            or any(not re.fullmatch(r"cuda:\d+", device) for device in devices)
            or any(int(device.split(":")[1]) >= count for device in devices)):
        raise SystemExit("select at least two distinct visible CUDA devices; no CPU or single-GPU fallback")
    plan = {"holdout": 1, "num_sweeps": len(datasets), "devices": devices,
            "workers_per_gpu": args.workers_per_gpu,
            "maximum_parallel_workers": len(devices) * args.workers_per_gpu,
            "total_student_runs": sum(len(ds.jobs) for ds in datasets),
            "output_root": str(args.output_root), "dry_run": args.dry_run,
            "sweeps": [{"tag": ds.tag, "teacher": str(ds.teacher),
                        "teacher_seed": ds.metadata["seed"], "checkpoint_sha256": ds.checkpoint_hash,
                        "cfm_width": ds.metadata["cfm"]["w"],
                        "generator_width": ds.metadata["generator"]["h_dim"],
                        "whiten": ds.contract["preprocessing"]["whiten"],
                        "reference": str(ds.reference), "output": str(ds.output),
                        "student_runs": len(ds.jobs),
                        "steps": sorted({settings["steps"] for _, settings in ds.jobs}),
                        "example_wandb_name": dataset_run_name(ds.contract, ds.jobs[0][0])}
                       for ds in datasets]}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    # Hash each original H5AD once before starting any expensive workers.
    manifest_hash = sha256_file(args.data_dir / "data_manifest.json")
    checked = set()
    for ds in datasets:
        if ds.contract["manifest_sha256"] != manifest_hash:
            raise ValueError("data manifest differs from the exported teacher")
        source = ds.contract["source"]
        path = args.data_dir / Path(source["file"]).name
        key = (path, source["sha256"])
        if key not in checked:
            if sha256_file(path) != source["sha256"]:
                raise ValueError(f"source data checksum mismatch: {path}")
            checked.add(key)
    run(args, datasets, devices)


if __name__ == "__main__":
    main()
