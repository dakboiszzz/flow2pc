"""Reuse the proven shared GPU queue for all migrated teacher/student checks.

The controller imports no Torch. GPU discovery and memory monitoring use
nvidia-smi; workers see one assigned GPU as cuda:0. --dry-run writes nothing
and starts no workers, W&B runs, training or evaluation.
"""

import argparse
import copy
import csv
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))

from flow2pc.utils.config import load_config  # noqa: E402
from flow2pc.utils.gpu_queue import choose_device, next_task, stop_workers  # noqa: E402

TAGS = (
    "cite5",
    "cite50",
    "cite100",
    "multi5",
    "multi50",
    "multi100",
    "eb",
    "st",
    "cell_tracking",
    "knot",
)


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def gpu_inventory():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"nvidia-smi query failed: {error}") from error
    inventory = {}
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 3:
            continue
        index, uuid, free = [value.strip() for value in row]
        if not index.isdigit():
            continue
        inventory[index] = {
            "uuid": uuid,
            "free": max(0, int(free)) * 2**20 if free.isdigit() else 0,
        }
    return inventory


def selected_devices(value, inventory):
    def resolve(token):
        token = token.strip()
        if token in inventory:
            return token
        matches = [
            index
            for index, row in inventory.items()
            if token and row["uuid"].startswith(token)
        ]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"Unknown/ambiguous physical GPU index or UUID: {token}")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    allowed = (
        list(inventory)
        if visible is None
        else [resolve(token) for token in visible.split(",") if token.strip()]
    )
    devices = (
        allowed if value == "auto" else [resolve(token) for token in value.split(",")]
    )
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise ValueError(
            "Select at least two distinct GPUs; no CPU or single-GPU fallback"
        )
    if set(devices) - set(allowed):
        raise ValueError(
            "Requested GPUs are outside the inherited CUDA_VISIBLE_DEVICES allocation"
        )
    return devices


@dataclass
class Dataset:
    tag: str
    output: Path
    pending: deque = field(default_factory=deque)
    failures: list = field(default_factory=list)
    lock: object = None
    done: bool = False


@dataclass
class Task:
    dataset: Dataset
    stage: str
    retries: int = 0


def overrides(args, dataset, phase):
    result = {
        "output.dir": str(dataset.output / phase),
        "logging.project": "flow2pc",
        "logging.mode": "online",
        "evaluation.enabled": True,
    }
    if phase == "teacher":
        result.update({"data.root": str(args.data_root), "training.device": "cuda:0"})
    else:
        result.update(
            {
                "teacher.checkpoint": str(
                    dataset.output / "teacher/checkpoints/last.pt"
                ),
                "distillation.device": "cuda:0",
                "distillation.resume": True,
            }
        )
    return [f"{key}={json.dumps(value)}" for key, value in result.items()]


def expected_config(args, dataset, phase):
    stage = "pretrain" if phase == "teacher" else "distill"
    suffix = "" if phase == "teacher" else "_hclt"
    return load_config(
        ROOT / f"configs/{stage}/alicfm/alicfm_{dataset.tag}{suffix}.yaml",
        overrides(args, dataset, phase),
    )


def scientific_config(config):
    result = copy.deepcopy(config)
    for key in ("output", "logging", "evaluation"):
        result.pop(key, None)
    for key in ("training", "distillation"):
        if key in result:
            result[key].pop("device", None)
            result[key].pop("resume", None)
    return result


def compatible_config(args, dataset, phase):
    path = dataset.output / phase / "resolved_config.yaml"
    expected = expected_config(args, dataset, phase)
    if not path.is_file():
        return expected, None
    saved = yaml.safe_load(path.read_text())
    if scientific_config(saved) != scientific_config(expected):
        raise ValueError(
            f"Saved {dataset.tag}/{phase} scientific settings differ; use a new output root"
        )
    return expected, saved


def trained(directory, phase):
    if phase == "teacher":
        return (directory / "checkpoints/last.pt").is_file()
    status = read_json(directory / "status.json")
    return status.get("status") == "complete" and all(
        (directory / f"checkpoints/{name}.pt").is_file() for name in ("best", "last")
    )


def complete(args, dataset, phase):
    directory = dataset.output / phase
    expected, saved = compatible_config(args, dataset, phase)
    if not trained(directory, phase) or saved is None:
        return False
    metrics = read_json(directory / "eval/metrics.json")
    expected_protocol = {
        k: v
        for k, v in expected["evaluation"].items()
        if k not in ("enabled", "image_dir")
    }
    if any(
        metrics.get("protocol", {}).get(k) != v for k, v in expected_protocol.items()
    ):
        return False
    media = read_json(directory / "eval/media.json")
    return (
        read_json(directory / "status.json").get("status") == "complete"
        and read_json(directory / "eval/status.json").get("status") == "complete"
        and (directory / "eval/metrics.json").is_file()
        and read_json(directory / "wandb_run.json").get("project") == "flow2pc"
        and bool(media)
        and all(Path(path).is_file() for path in media.values())
    )


def prepare_dataset(args, dataset):
    # No models/data are loaded by the scheduler. Workers verify artifact hashes.
    teachers_only = getattr(args, "teachers_only", False)
    for phase in (("teacher",) if teachers_only else ("teacher", "student")):
        compatible_config(args, dataset, phase)
        directory = dataset.output / phase
        if directory.exists() and not (directory / "checkpoints/last.pt").is_file():
            raise ValueError(
                f"Partial {dataset.tag}/{phase} output has no resumable checkpoint: {directory}. Preserve it and select a fresh output root for this dataset."
            )
    teacher_done = complete(args, dataset, "teacher")
    if teachers_only:
        if teacher_done and not args.rerun_evaluation:
            dataset.done = True
        else:
            dataset.pending.append(Task(dataset, "teacher"))
        return
    if (
        teacher_done
        and complete(args, dataset, "student")
        and not args.rerun_evaluation
    ):
        dataset.done = True
        return
    dataset.pending.append(
        Task(
            dataset,
            "student" if teacher_done and not args.rerun_evaluation else "teacher",
        )
    )


def prepare_or_record_failure(args, dataset):
    dataset.pending.clear()
    dataset.failures.clear()
    dataset.done = False
    try:
        prepare_dataset(args, dataset)
    except (ValueError, OSError, yaml.YAMLError) as error:
        dataset.failures.append({"stage": "prepare", "reason": str(error)})


def command_for(args, task):
    if getattr(args, "teachers_only", False) and task.stage != "teacher":
        raise ValueError("Teacher-only launcher cannot execute a student stage")
    directory = task.dataset.output / task.stage
    action = (
        ("evaluate-teacher" if trained(directory, "teacher") else "pretrain")
        if task.stage == "teacher"
        else ("evaluate-student" if trained(directory, "student") else "distill")
    )
    command = ["bash", str(ROOT / "scripts/run_alicfm.sh"), action, task.dataset.tag]
    for value in overrides(args, task.dataset, task.stage):
        command.extend(("--set", value))
    return command


def start_worker(args, task, device, inventory):
    log_path = (
        args.output_root / "_launcher/logs" / f"{task.dataset.tag}-{task.stage}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")
    offset = log.tell()
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": inventory[device]["uuid"],
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "FLOW2PC_PYTHON": sys.executable,
        "OMP_NUM_THREADS": str(args.threads),
        "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads),
        "PYTHONUNBUFFERED": "1",
        "FLOW2PC_WANDB_INIT_LOCK": str(args.output_root / ".wandb-init.lock"),
    }
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for key in ("WANDB_RUN_ID", "WANDB_NAME", "WANDB_RESUME"):
        env.pop(key, None)
    try:
        process = subprocess.Popen(
            command_for(args, task),
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            pass_fds=(task.dataset.lock.fileno(),),
        )
    except BaseException:
        log.close()
        raise
    print(
        f"Started {task.dataset.tag}/{task.stage} on GPU {device}; pid={process.pid}; log={log_path}",
        flush=True,
    )
    return {
        "task": task,
        "device": device,
        "process": process,
        "log": log,
        "log_path": log_path,
        "log_offset": offset,
        "started": time.monotonic(),
        "reservation_bytes": (
            args.teacher_min_free_gb
            if task.stage == "teacher"
            else args.student_min_free_gb
        )
        * 2**30,
    }


def write_status(args, datasets, active, state, free):
    rows = []
    running = {worker["task"].dataset.tag for worker in active}
    for dataset in datasets:
        teacher = read_json(dataset.output / "teacher/eval/metrics.json")
        student = read_json(dataset.output / "student/eval/metrics.json")
        rows.append(
            {
                "dataset": dataset.tag,
                "status": "failed"
                if dataset.failures
                else "complete"
                if dataset.done
                else "running"
                if dataset.tag in running
                else "pending",
                "teacher_mean_emd": teacher.get("teacher_mean_emd"),
                "student_mean_emd": student.get("student_mean_emd"),
                "teacher_heldout_emd": teacher.get("teacher_heldout_emd"),
                "student_heldout_emd": student.get("student_heldout_emd"),
                "teacher_media": str(dataset.output / "teacher/eval/media.json"),
                "student_media": str(dataset.output / "student/eval/media.json"),
            }
        )
    atomic_json(
        args.output_root / "_launcher/status.json",
        {
            "status": state,
            "teachers_only": getattr(args, "teachers_only", False),
            "wandb_project": "flow2pc",
            "pending_tasks": sum(len(ds.pending) for ds in datasets),
            "free_gib": {d: round(b / 2**30, 2) for d, b in free.items()},
            "active": [
                {
                    "dataset": w["task"].dataset.tag,
                    "stage": w["task"].stage,
                    "gpu": w["device"],
                    "pid": w["process"].pid,
                }
                for w in active
            ],
            "datasets": rows,
            "failures": [
                {"dataset": ds.tag, **failure}
                for ds in datasets
                for failure in ds.failures
            ],
        },
    )
    path = args.output_root / "_launcher/summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args, datasets, devices):
    active, fair_queue, cooldown = [], deque(datasets), {}
    interrupted, handlers, free = False, {}, {}

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    with ExitStack() as stack:
        args.output_root.mkdir(parents=True, exist_ok=True)
        for directory, filename, dataset in [
            (args.output_root, ".parallel-checks.lock", None),
            *((ds.output, ".pipeline.lock", ds) for ds in datasets),
        ]:
            directory.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((directory / filename).open("a"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError(
                    f"Another check launcher is using {directory}"
                ) from None
            if dataset is not None:
                dataset.lock = lock
                prepare_or_record_failure(args, dataset)
        handlers = {
            sig: signal.signal(sig, interrupt)
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            report_at = monitor_report_at = 0
            while active or any(ds.pending for ds in datasets):
                now = time.monotonic()
                for worker in active[:]:
                    code = worker["process"].poll()
                    if code is None:
                        continue
                    active.remove(worker)
                    worker["log"].close()
                    task, dataset = worker["task"], worker["task"].dataset
                    if code == 0 and complete(args, dataset, task.stage):
                        if getattr(args, "teachers_only", False):
                            dataset.done = True
                        elif task.stage == "teacher":
                            if args.rerun_evaluation or not complete(
                                args, dataset, "student"
                            ):
                                dataset.pending.append(Task(dataset, "student"))
                            else:
                                dataset.done = True
                        else:
                            dataset.done = True
                    else:
                        try:
                            os.killpg(worker["process"].pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        with worker["log_path"].open(errors="replace") as handle:
                            handle.seek(0, 2)
                            handle.seek(
                                max(worker["log_offset"], handle.tell() - 131072)
                            )
                            error = handle.read().lower()
                        oom = any(
                            text in error
                            for text in ("out of memory", "cublas_status_alloc_failed")
                        )
                        directory = dataset.output / task.stage
                        resumable = (
                            not directory.exists()
                            or (directory / "checkpoints/last.pt").is_file()
                        )
                        if oom and task.retries < args.oom_retries and resumable:
                            task.retries += 1
                            cooldown[worker["device"]] = now + 60
                            dataset.pending.append(task)
                            print(
                                f"Requeued {dataset.tag}/{task.stage} after OOM; GPU cooldown 60s",
                                flush=True,
                            )
                        else:
                            dataset.failures.append(
                                {
                                    "stage": task.stage,
                                    "exit_code": code,
                                    "reason": "missing completed evaluation"
                                    if code == 0
                                    else "worker failed",
                                    "resumable": resumable,
                                    "log": str(worker["log_path"]),
                                }
                            )
                    print(
                        f"Finished {dataset.tag}/{task.stage}: exit={code}", flush=True
                    )
                try:
                    inventory = gpu_inventory()
                    free = {
                        d: inventory.get(d, {}).get("free", 0)
                        if cooldown.get(d, 0) <= now
                        else 0
                        for d in devices
                    }
                except RuntimeError as error:
                    free = dict.fromkeys(devices, 0)
                    if now >= monitor_report_at:
                        print(
                            f"GPU monitor unavailable; pausing launches: {error}",
                            flush=True,
                        )
                        monitor_report_at = now + 30
                while (
                    any(ds.pending for ds in datasets)
                    and len(active) < args.max_parallel
                ):
                    launched = False
                    for _ in range(len(datasets)):
                        if len(active) >= args.max_parallel:
                            break
                        task = next_task(fair_queue)
                        if task is None:
                            break
                        required = (
                            args.teacher_min_free_gb
                            if task.stage == "teacher"
                            else args.student_min_free_gb
                        )
                        launch_time = time.monotonic()
                        # The recovered selector reserves one current threshold per
                        # warming worker. Correct its input for differing teacher/
                        # student reservations while leaving that proven body intact.
                        adjusted = {
                            d: free[d]
                            + sum(
                                required * 2**30 - w["reservation_bytes"]
                                for w in active
                                if w["device"] == d and launch_time - w["started"] < 30
                            )
                            for d in devices
                        }
                        device = choose_device(
                            devices,
                            active,
                            adjusted,
                            args.workers_per_gpu,
                            required,
                            launch_time,
                        )
                        if device is None:
                            task.dataset.pending.appendleft(task)
                            continue
                        active.append(start_worker(args, task, device, inventory))
                        launched = True
                    if not launched:
                        break
                if now >= report_at:
                    write_status(args, datasets, active, "running", free)
                    print(
                        f"Active={len(active)}/{args.max_parallel}; pending={sum(len(ds.pending) for ds in datasets)}; free GiB={ {d: round(b / 2**30, 1) for d, b in free.items()} }",
                        flush=True,
                    )
                    report_at = now + 30
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
                active.clear()
                write_status(
                    args,
                    datasets,
                    active,
                    "interrupted"
                    if interrupted
                    else "failed"
                    if any(ds.failures for ds in datasets)
                    else "complete",
                    free,
                )
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
    return 1 if any(ds.failures for ds in datasets) else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=TAGS, default=list(TAGS))
    parser.add_argument(
        "--gpus",
        default="auto",
        help="Physical nvidia-smi indices/UUIDs, e.g. 0,1,2,3; auto respects CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--max-parallel", type=int)
    parser.add_argument("--teacher-min-free-gb", type=float, default=4)
    parser.add_argument("--student-min-free-gb", type=float, default=8)
    parser.add_argument(
        "--min-free-gb", type=float, help="Override both stage memory thresholds"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--oom-retries", type=int, default=1)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("FLOW2PC_DATA_ROOT", "/data/lthung/ali-cfm-data")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "FLOW2PC_CHECKS_ROOT", "/data/lthung/flow2pc/framework/migration-check"
            )
        ),
    )
    parser.add_argument("--rerun-evaluation", action="store_true")
    parser.add_argument(
        "--teachers-only", action="store_true",
        help="Run teacher training/evaluation only; never queue distillation",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.min_free_gb is not None:
        args.teacher_min_free_gb = args.student_min_free_gb = args.min_free_gb
    if (
        args.workers_per_gpu < 1
        or args.threads < 1
        or args.oom_retries < 0
        or args.max_parallel is not None
        and args.max_parallel < 1
        or any(
            not math.isfinite(v) or v <= 0
            for v in (args.teacher_min_free_gb, args.student_min_free_gb)
        )
        or len(set(args.datasets)) != len(args.datasets)
    ):
        parser.error(
            "Invalid GPU concurrency, memory, thread, retry or dataset settings"
        )
    args.data_root, args.output_root = (
        args.data_root.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
    )
    if not args.data_root.is_dir():
        parser.error(f"Data root does not exist: {args.data_root}")
    inventory = gpu_inventory()
    devices = selected_devices(args.gpus, inventory)
    args.max_parallel = min(
        args.max_parallel or len(devices) * args.workers_per_gpu,
        len(devices) * args.workers_per_gpu,
    )
    if args.max_parallel < 2:
        parser.error("Choose at least two parallel slots for this multi-GPU launcher")
    datasets = [Dataset(tag, args.output_root / tag) for tag in args.datasets]
    for dataset in datasets:
        prepare_or_record_failure(args, dataset)
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "teachers_only": args.teachers_only,
                "wandb_project": "flow2pc",
                "output_root": str(args.output_root),
                "data_root": str(args.data_root),
                "gpus": {
                    d: {
                        "uuid": inventory[d]["uuid"],
                        "free_gib": round(inventory[d]["free"] / 2**30, 2),
                    }
                    for d in devices
                },
                "maximum_parallel_workers": args.max_parallel,
                "workers_per_gpu": args.workers_per_gpu,
                "teacher_min_free_gib": args.teacher_min_free_gb,
                "student_min_free_gib": args.student_min_free_gb,
                "datasets": [
                    {
                        "tag": ds.tag,
                        "already_complete": ds.done,
                        "next_stage": ds.pending[0].stage if ds.pending else None,
                        "errors": ds.failures,
                    }
                    for ds in datasets
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return (
        int(any(ds.failures for ds in datasets))
        if args.dry_run
        else run(args, datasets, devices)
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, RuntimeError, OSError, yaml.YAMLError) as error:
        print(f"parallel checks: {error}", file=sys.stderr)
        raise SystemExit(2)
