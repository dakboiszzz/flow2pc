"""Eight-GPU student sweep with one shared frozen MNIST latent flow teacher.

Each GPU runs one training/evaluation subprocess at a time. Finished slots
immediately take another configuration. No model runs until main is invoked.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
from itertools import product
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARTIFACT_ROOT = Path("/data/lthung/flow2pc")

DEFAULTS = {
    "steps": 20000, "lr": 5e-4, "batch_size": 32, "seed": 0,
    "num_latents": 64, "time_embedding_dim": 16, "pc_structure": "hclt",
    "pc_layer": "dense", "time_conditioning": "fourier", "time_order": 4,
    "leaf_components": 1, "num_components": 256, "hclt_samples": 20000,
    "validate_every": 1000, "checkpoint_every": 5000, "log_every": 100,
    "weight_decay": 0.01, "gradient_clip": 1.0,
}


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else {}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mnist_run_name(name):
    return name if name.startswith("mnist") else f"mnist-{name}"


def student_run_name(args, job, stage):
    return f"mnist-z{args.latent_dim}-{job['name']}-steps{job['settings']['steps']}-{stage}"


def validate_settings(settings):
    unknown = set(settings) - DEFAULTS.keys()
    if unknown:
        raise ValueError(f"unsupported sweep settings: {sorted(unknown)}")
    for key, default in DEFAULTS.items():
        value = settings[key]
        if isinstance(default, int):
            minimum = 0 if key == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        elif isinstance(default, float):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"invalid {key}")
            if value < 0 or (key != "weight_decay" and value == 0):
                raise ValueError(f"invalid {key}")
    if settings["pc_structure"] not in ("hclt", "gaussian"):
        raise ValueError("pc_structure must be hclt or gaussian")
    if settings["pc_layer"] not in ("dense", "monarch"):
        raise ValueError("pc_layer must be dense or monarch")
    if settings["time_conditioning"] not in ("polynomial", "fourier"):
        raise ValueError("time_conditioning must be polynomial or fourier")
    if settings["time_embedding_dim"] < 2 or settings["time_embedding_dim"] % 2:
        raise ValueError("time_embedding_dim must be positive and even")
    if settings["num_latents"] < 2 or settings["hclt_samples"] < 2:
        raise ValueError("HCLT states and topology sample count must be at least two")
    if settings["pc_layer"] == "monarch" and math.isqrt(settings["num_latents"])**2 != settings["num_latents"]:
        raise ValueError("Monarch requires a square number of hidden states")


def load_jobs(path, overrides=None):
    grid = read_json(path)
    common = grid.get("common", {})
    if not isinstance(common, dict) or ("parameters" in grid) == ("runs" in grid):
        raise ValueError("grid needs common settings and exactly one of parameters or runs")
    if "parameters" in grid:
        parameters = grid["parameters"]
        if not isinstance(parameters, dict) or not parameters:
            raise ValueError("parameters must be a nonempty object")
        for key, values in parameters.items():
            if key not in DEFAULTS or not isinstance(values, list) or not values:
                raise ValueError(f"invalid parameter axis: {key}")
            encoded = [json.dumps(value, sort_keys=True, allow_nan=False) for value in values]
            if len(set(encoded)) != len(encoded):
                raise ValueError(f"duplicate axis values: {key}")
        keys = list(parameters)
        runs = [dict(zip(keys, values)) for values in product(*(parameters[key] for key in keys))]
    else:
        runs = grid["runs"]
        if not isinstance(runs, list) or not runs:
            raise ValueError("runs must be a nonempty list")
    jobs, seen, names = [], set(), set()
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("each run must be an object")
        settings = {**DEFAULTS, **common, **{key: value for key, value in run.items() if key != "name"},
                    **(overrides or {})}
        validate_settings(settings)
        canonical = json.dumps(settings, sort_keys=True, allow_nan=False)
        if canonical in seen:
            continue
        seen.add(canonical)
        name = run.get("name") or (
            f"h{settings['num_latents']}-f{settings['time_embedding_dim']}-lr{settings['lr']:g}"
            f"-b{settings['batch_size']}-s{settings['seed']}-{settings['pc_layer']}"
        )
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("job names must be simple directory names")
        name = f"{name[:150]}-{hashlib.sha256(canonical.encode()).hexdigest()[:10]}"
        if name in names:
            raise ValueError("configuration name collision")
        names.add(name)
        jobs.append({"name": name, "settings": settings})
    return jobs


def parse_devices(value, max_parallel, verify=True):
    if value == "auto":
        devices = [f"cuda:{index}" for index in range(max_parallel)]
    else:
        devices = [item.strip() for item in value.split(",")]
    if (not devices or any(not re.fullmatch(r"cuda:\d+|cpu", device) for device in devices)
            or len(devices) != max_parallel):
        raise ValueError("provide exactly max-parallel devices, e.g. cuda:0,...,cuda:7")
    gpus = [device for device in devices if device != "cpu"]
    if len(set(gpus)) != len(gpus):
        raise ValueError("each GPU may appear only once; one worker is assigned to each GPU")
    if verify and gpus:
        import torch

        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if any(int(device.split(":")[1]) >= count for device in gpus):
            raise ValueError(f"requested {devices}, but only {count} logical CUDA devices are visible; check CUDA_VISIBLE_DEVICES")
    return devices


def require_external(path):
    path = Path(path).resolve()
    if ARTIFACT_ROOT.resolve() not in path.parents:
        raise ValueError(f"runtime artifacts must be under {ARTIFACT_ROOT}: {path}")
    return path


def ready_device(devices, min_free_gb):
    """Honor logical CUDA numbering and leave busy GPUs for their current jobs."""
    for device in devices:
        if device == "cpu" or min_free_gb == 0:
            return device
        import torch

        free, _ = torch.cuda.mem_get_info(int(device.split(":")[1]))
        if free >= min_free_gb * 1024**3:
            return device
    return None


def estimated_memory_gb(settings=None, dim=196):
    """A launch budget, not a measured peak: parameter states plus graph headroom."""
    if not settings or settings["pc_structure"] != "hclt":
        return 2
    states = settings["num_latents"]
    features = (settings["time_embedding_dim"] if settings["time_conditioning"] == "fourier"
                else settings["time_order"])
    transitions = (dim - 1) * (states**2 if settings["pc_layer"] == "dense"
                         else 3 * states * math.isqrt(states))
    leaves = dim * states * settings["leaf_components"] * 2 * features
    parameters = transitions + leaves + states
    # Four FP32 copies cover weights, gradients, and Adam moments; another
    # eight allow for differentiating the velocity and temporary dense layers.
    return max(2, math.ceil(1 + 12 * 4 * parameters / 1024**3))


def memory_requirement(args, settings=None):
    return estimated_memory_gb(settings, getattr(args, "latent_dim", 196)) if args.min_free_gb is None else args.min_free_gb


def choose_pending_job(args, queue, available):
    # Keep each just-trained student's evaluation on its reserved GPU.
    reserved = {job["preferred_device"] for job in queue if job.get("preferred_device")}
    for job in queue:
        preferred = job.get("preferred_device")
        if preferred in available:
            return job, preferred
    free = {}
    for device in available:
        if device in reserved:
            continue
        if device == "cpu" or args.min_free_gb == 0:
            free[device] = float("inf")
        else:
            import torch

            free[device] = torch.cuda.mem_get_info(int(device.split(":")[1]))[0] / 1024**3
    # Skip a large queued student if a smaller one fits an idle GPU. Among
    # fitting slots, use the least free memory to leave roomy GPUs for large PCs.
    for job in queue:
        if job.get("preferred_device"):
            continue
        required = memory_requirement(args, job["settings"])
        fitting = [device for device in free if free[device] >= required]
        if fitting:
            return job, min(fitting, key=lambda device: free[device])
    return None


def wait_for_device(args, devices, settings=None):
    next_notice = 0
    required = memory_requirement(args, settings)
    while True:
        device = ready_device(devices, required)
        if device is not None:
            return device
        if time.monotonic() >= next_notice:
            logging.info("Waiting for a GPU with %.1f GiB free", required)
            next_notice = time.monotonic() + 30
        time.sleep(5)


def fresh_path(path):
    path = Path(path)
    candidate, index = path, 0
    while candidate.exists():
        index += 1
        candidate = path.with_name(f"{path.name}-{index:03d}")
    return candidate


def child_environment(args, runtime):
    runtime.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for key in ("WANDB_RUN_ID", "WANDB_NAME", "WANDB_RESUME", "WANDB_SWEEP_ID"):
        env.pop(key, None)
    for key, directory in (
        ("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"), ("XDG_CONFIG_HOME", "config"),
        ("MPLCONFIGDIR", "matplotlib"), ("WANDB_DIR", "wandb"),
        ("WANDB_DATA_DIR", "wandb-data"), ("WANDB_CACHE_DIR", "wandb-cache"),
        ("WANDB_CONFIG_DIR", "wandb-config"), ("PYTHONPYCACHEPREFIX", "pycache"),
    ):
        directory = runtime / directory
        directory.mkdir(exist_ok=True)
        env[key] = str(directory)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(args.threads)
    env.update(PYTHONDONTWRITEBYTECODE="1", WANDB__SERVICE_WAIT="300", WANDB_INIT_TIMEOUT="300",
               WANDB_HTTP_TIMEOUT="120", FLOW2PC_WANDB_INIT_LOCK=str(args.output_dir / ".wandb-init.lock"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def common_arguments(args, device):
    values = ["--bundle-dir", str(args.bundle_dir), "--data-dir", str(args.data_dir),
              "--eqvae-root", str(args.eqvae_root), "--vae-checkpoint", str(args.vae_checkpoint),
              "--device", device, "--threads", str(args.threads), "--num-workers", "0",
              "--wandb-project", "flow2pc", "--wandb-group", args.wandb_group,
              "--wandb-mode", args.wandb_mode, "--ode-steps", "100", "--image-every", "0"]
    if args.wandb_entity:
        values += ["--wandb-entity", args.wandb_entity]
    return values


def evaluation_arguments(args, shared, device, checkpoint, output, name, split="validation"):
    return [sys.executable, "-B", "-u", str(ROOT / "train_mnist_latent.py"), "evaluate",
            *common_arguments(args, device), "--checkpoint", str(checkpoint),
            "--teacher-checkpoint", shared["teacher_checkpoint"],
            "--classifier-checkpoint", shared["classifier_checkpoint"],
            "--split", split, "--num-generated", str(args.eval_samples if split == "validation" else args.test_samples),
            "--batch-size", str(args.eval_batch_size), "--seed", str(args.eval_seed),
            "--kid-subsets", str(args.kid_subsets), "--kid-subset-size", str(args.kid_subset_size),
            "--is-splits", str(args.is_splits), "--metric-chunk-size", str(args.metric_chunk_size),
            "--output-dir", str(output), "--wandb-run-name", mnist_run_name(name)]


def terminate(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
    except ProcessLookupError:
        pass


def run_sync(args, command, label):
    log_path = args.output_dir / "bootstrap" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Preparing %s; log: %s", label, log_path)
    with log_path.open("a") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=child_environment(args, log_path.parent / f".{label}-runtime"),
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait()
        finally:
            terminate(process)
    if code:
        raise RuntimeError(f"{label} failed (exit {code}); see {log_path}")


def prepare_shared(args, device):
    # Only this coordinator prepares shared artifacts, before student workers
    # start. Workers never race to export latents or train the evaluator.
    from repc.mnist_latent import load_latent_bundle
    from repc.mnist_latent_eval import resolve_classifier_checkpoint
    from train_mnist_latent import load_latent_teacher
    import torch

    common = common_arguments(args, device)
    if not (args.bundle_dir / "manifest.json").is_file():
        output = fresh_path(args.output_dir / "bootstrap/export")
        run_sync(args, [sys.executable, "-B", "-u", str(ROOT / "train_mnist_latent.py"), "export",
                        *common, "--download", "--batch-size", "256", "--output-dir", str(output)], "export")
    codes, contract = load_latent_bundle(args.bundle_dir)
    del codes
    cached = read_json(args.output_dir / "shared_inputs.json")
    teacher = args.teacher_checkpoint
    teacher_record = read_json(args.output_dir / "bootstrap/teacher.json")
    if teacher is None and teacher_record:
        teacher = Path(teacher_record["checkpoint"])
        if file_hash(teacher) != teacher_record["sha256"]:
            raise ValueError("shared teacher changed; use a fresh output directory")
    if teacher is None:
        output = fresh_path(args.output_dir / "bootstrap/teacher")
        run_sync(args, [sys.executable, "-B", "-u", str(ROOT / "train_mnist_latent.py"), "flow", *common,
                        "--seed", str(args.teacher_seed), "--steps", str(args.teacher_steps),
                        "--batch-size", str(args.teacher_batch_size), "--lr", str(args.teacher_lr),
                        "--hidden-dim", str(args.teacher_hidden_dim), "--num-blocks", str(args.teacher_blocks),
                        "--output-dir", str(output), "--wandb-run-name", mnist_run_name(f"{args.wandb_group}-teacher")], "teacher")
        teacher = output / "checkpoints/best.pt"
        atomic_json(args.output_dir / "bootstrap/teacher.json", {"checkpoint": str(teacher), "sha256": file_hash(teacher)})
    teacher_model, _ = load_latent_teacher(teacher, contract, torch.device("cpu"))
    del teacher_model
    classifier = args.classifier_checkpoint
    classifier_record = read_json(args.output_dir / "bootstrap/classifier.json")
    if classifier is None and classifier_record:
        classifier = Path(classifier_record["checkpoint"])
        if file_hash(classifier) != classifier_record["sha256"]:
            raise ValueError("shared classifier changed; use a fresh output directory")
    try:
        classifier = resolve_classifier_checkpoint(classifier)
    except FileNotFoundError:
        if classifier is not None:
            raise
        output = fresh_path(args.output_dir / "bootstrap/classifier")
        command = [sys.executable, "-B", "-u", str(ROOT / "train_mnist_classifier.py"),
                   "--data-dir", str(args.data_dir), "--output-dir", str(output),
                   "--device", device, "--seed", "1", "--epochs", "14", "--num-workers", "0",
                   "--wandb-project", "flow2pc", "--wandb-mode", args.wandb_mode,
                   "--wandb-run-name", mnist_run_name(f"{args.wandb_group}-classifier")]
        if args.wandb_entity:
            command += ["--wandb-entity", args.wandb_entity]
        run_sync(args, command, "classifier")
        classifier = output / "checkpoints/last.pt"
        atomic_json(args.output_dir / "bootstrap/classifier.json", {"checkpoint": str(classifier), "sha256": file_hash(classifier)})
    checkpoint = torch.load(classifier, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "mnist_classifier":
        raise ValueError("expected a trained MNIST classifier")
    del checkpoint
    shared = {"teacher_checkpoint": str(teacher.resolve()), "teacher_sha256": file_hash(teacher),
              "classifier_checkpoint": str(classifier.resolve()), "classifier_sha256": file_hash(classifier),
              "bundle_manifest_sha256": contract["manifest_sha256"]}
    if cached and shared != cached:
        raise ValueError("sweep teacher/classifier/data changed; use a new output directory")
    atomic_json(args.output_dir / "shared_inputs.json", shared)
    reference_record = read_json(args.output_dir / "teacher_reference.json")
    if not reference_record or not Path(reference_record["metrics"]).is_file():
        output = fresh_path(args.output_dir / "teacher-validation")
        run_sync(args, evaluation_arguments(args, shared, device, teacher, output,
                                            f"{args.wandb_group}-teacher-validation"), "teacher-validation")
        atomic_json(args.output_dir / "teacher_reference.json", {"metrics": str(output / "metrics.json")})
    return shared


def job_state(args, job):
    return read_json(args.output_dir / "runs" / job["name"] / "status.json")


def set_job_state(args, job, **updates):
    state = {**job_state(args, job), **updates}
    atomic_json(args.output_dir / "runs" / job["name"] / "status.json", state)
    return state


def pending_jobs(args, jobs):
    pending, failures = deque(), []
    for definition in jobs:
        job = dict(definition)
        state = job_state(args, job)
        if state.get("status") == "complete" and Path(state.get("eval_output", ""), "metrics.json").is_file():
            continue
        train_output = Path(state["train_output"]) if state.get("train_output") else None
        trained = (train_output is not None and read_json(train_output / "training_summary.json").get("status") == "complete"
                   and (train_output / "checkpoints/best.pt").is_file())
        if state.get("status") == "failed" and not args.retry_failed:
            failures.append(job["name"])
            continue
        if trained:
            job.update(stage="evaluate", attempt=state["attempt"], train_output=train_output)
        else:
            attempt = state.get("attempt", -1) + 1
            attempt_root = args.output_dir / "runs" / job["name"] / f"attempt{attempt:03d}"
            job.update(stage="train", attempt=attempt, train_output=attempt_root / "train")
        pending.append(job)
    return pending, failures


def start_worker(args, shared, job, device):
    attempt_root = job["train_output"].parent
    attempt_root.mkdir(parents=True, exist_ok=True)
    stage = job["stage"]
    if stage == "train":
        command = [sys.executable, "-B", "-u", str(ROOT / "train_mnist_latent.py"), "distill",
                   *common_arguments(args, device), "--teacher-checkpoint", shared["teacher_checkpoint"],
                   "--output-dir", str(job["train_output"]), "--wandb-run-name", student_run_name(args, job, "train")]
        for key, value in job["settings"].items():
            command.extend([f"--{key.replace('_', '-')}", str(value)])
    else:
        job["eval_output"] = fresh_path(attempt_root / "validation")
        command = evaluation_arguments(args, shared, device, job["train_output"] / "checkpoints/best.pt",
                                        job["eval_output"], student_run_name(args, job, "validation"))
    log_path = fresh_path(attempt_root / f"{stage}.log")
    stream = log_path.open("a")
    process = None
    try:
        process = subprocess.Popen(command, cwd=ROOT, env=child_environment(args, attempt_root / f".{stage}-runtime"),
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        set_job_state(args, job, status="running", stage=stage, attempt=job["attempt"], device=device,
                      pid=process.pid, train_output=str(job["train_output"]),
                      eval_output=str(job.get("eval_output", "")), log=str(log_path), started_at=time.time())
    except BaseException:
        if process is not None:
            terminate(process)
        stream.close()
        raise
    logging.info("%s %s on %s (PID %d); log %s", stage, job["name"], device, process.pid, log_path)
    return {"job": job, "device": device, "process": process, "log": stream}


def summarize(args, jobs):
    rows = []
    for job in jobs:
        state = job_state(args, job)
        metrics = read_json(Path(state["eval_output"]) / "metrics.json") if state.get("eval_output") else {}
        training = read_json(Path(state["train_output"]) / "training_summary.json") if state.get("train_output") else {}
        row = {"run": job["name"], "status": state.get("status", "pending"),
               "stage": state.get("stage", ""), "device": state.get("device", ""),
               "attempt": state.get("attempt", ""), **job["settings"],
               "best_validation_score": training.get("best_validation_score", ""),
               "checkpoint": str(Path(state["train_output"]) / "checkpoints/best.pt") if state.get("train_output") else "",
               "train_wandb_id": training.get("wandb_run_id", ""),
               "eval_output": state.get("eval_output", "")}
        for label in ("teacher", "student"):
            for metric in ("fid_mnist", "kid_mnist_mean", "inception_score_mnist_mean",
                           "precision_mnist", "recall_mnist", "generation_seconds", "parameter_count"):
                row[f"{label}/{metric}"] = metrics.get(f"{label}/{metric}", "")
        rows.append(row)
    temporary = args.output_dir / "summary.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output_dir / "summary.csv")
    complete = [row for row in rows if row["status"] == "complete" and row["best_validation_score"] != ""]
    if not complete:
        return None
    winner = min(complete, key=lambda row: (row["best_validation_score"], row["run"]))
    selected = {"selection_rule": "lowest validation teacher/student velocity MSE; test metrics unused",
                "checkpoint": winner["checkpoint"], "run": winner["run"], "metrics": winner}
    atomic_json(args.output_dir / "selected_student.json", selected)
    return selected


def run_pool(args, shared, jobs, devices):
    queue, failures = pending_jobs(args, jobs)
    available, active = deque(devices), []
    next_notice = 0
    try:
        while queue or active:
            while queue and available:
                assignment = choose_pending_job(args, queue, available)
                if assignment is None:
                    if time.monotonic() >= next_notice:
                        logging.info("GPU memory gate: %d active, %d queued; budget=%s",
                                     len(active), len(queue), args.min_free_gb if args.min_free_gb is not None else "per student")
                        next_notice = time.monotonic() + 30
                    break
                job, device = assignment
                queue.remove(job)
                available.remove(device)
                try:
                    active.append(start_worker(args, shared, job, device))
                except Exception as error:
                    set_job_state(args, job, status="failed", stage=job["stage"], error=str(error))
                    failures.append(job["name"])
                    available.append(device)
                    logging.exception("Could not launch %s", job["name"])
            for worker in list(active):
                code = worker["process"].poll()
                if code is None:
                    continue
                active.remove(worker)
                worker["log"].close()
                job, device = worker["job"], worker["device"]
                stage = job["stage"]
                expected = (job["train_output"] / "training_summary.json" if stage == "train"
                            else job["eval_output"] / "metrics.json")
                if code or not expected.is_file():
                    set_job_state(args, job, status="failed", exit_code=code,
                                  error=f"{stage} failed or did not produce {expected}")
                    failures.append(job["name"])
                    available.append(device)
                    logging.error("%s failed: %s (exit %d); other workers continue", stage, job["name"], code)
                elif stage == "train":
                    set_job_state(args, job, status="trained")
                    job["stage"] = "evaluate"
                    job["preferred_device"] = device
                    # Retain this GPU slot until the same student's evaluation
                    # finishes; another worker cannot accidentally share it.
                    queue.appendleft(job)
                    available.appendleft(device)
                else:
                    set_job_state(args, job, status="complete", finished_at=time.time())
                    available.append(device)
                    logging.info("Completed %s on %s", job["name"], device)
                summarize(args, jobs)
            if queue or active:
                time.sleep(1)
    finally:
        for worker in active:
            try:
                os.killpg(worker["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for worker in active:
            terminate(worker["process"])
            worker["log"].close()
            set_job_state(args, worker["job"], status="interrupted")
        summarize(args, jobs)
    return failures


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, default=ROOT / "configs/mnist_latent/full_sweep.json")
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("LATENT_SWEEP_DIR"))
    parser.add_argument("--teacher-checkpoint", type=Path, default=os.environ.get("LATENT_TEACHER_CHECKPOINT") or None)
    parser.add_argument("--classifier-checkpoint", type=Path, default=os.environ.get("LATENT_CLASSIFIER_CHECKPOINT") or None)
    parser.add_argument("--bundle-dir", type=Path, default=os.environ.get("LATENT_BUNDLE_DIR", str(ARTIFACT_ROOT / "latents/mnist-eqvae-seed0")))
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("LATENT_DATA_DIR", str(ARTIFACT_ROOT / "data/mnist")))
    parser.add_argument("--eqvae-root", type=Path, default=os.environ.get("EQVAE_ROOT", str(ROOT.parent / "eqvae")))
    parser.add_argument("--vae-checkpoint", type=Path, default=os.environ.get("LATENT_VAE_CHECKPOINT", "/data/lthung/eqvae/mnist-eqvae-seed0/best.pt"))
    parser.add_argument("--devices", default="auto", help="logical CUDA devices; auto uses cuda:0,...,cuda:7")
    parser.add_argument("--max-parallel", type=int, default=8)
    parser.add_argument("--min-free-gb", type=float,
                        help="override automatic per-student launch budgets with a fixed free-GiB threshold; 0 disables")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--steps", type=int, help="override every student's training budget")
    parser.add_argument("--eval-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=10000)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--teacher-steps", type=int, default=30000)
    parser.add_argument("--teacher-seed", type=int, default=0)
    parser.add_argument("--teacher-lr", type=float, default=5e-4)
    parser.add_argument("--teacher-hidden-dim", type=int, default=512)
    parser.add_argument("--teacher-blocks", type=int, default=6)
    parser.add_argument("--teacher-batch-size", type=int, default=256)
    parser.add_argument("--kid-subsets", type=int, default=100)
    parser.add_argument("--kid-subset-size", type=int, default=1000)
    parser.add_argument("--is-splits", type=int, default=10)
    parser.add_argument("--metric-chunk-size", type=int, default=256)
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=os.environ.get("WANDB_MODE", "online"))
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="expand grid without writes, CUDA checks, or subprocesses")
    args = parser.parse_args(argv)
    for name in ("max_parallel", "threads", "eval_batch_size", "teacher_steps", "teacher_hidden_dim",
                 "teacher_blocks", "teacher_batch_size", "kid_subsets", "kid_subset_size", "is_splits", "metric_chunk_size"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if min(args.eval_samples, args.test_samples) < 4 or args.kid_subset_size < 2:
        parser.error("use at least four samples and two samples per KID subset")
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be positive")
    if args.min_free_gb is not None and (not math.isfinite(args.min_free_gb) or args.min_free_gb < 0):
        parser.error("min-free-gb must be finite and nonnegative")
    if not math.isfinite(args.teacher_lr) or args.teacher_lr <= 0 or min(args.eval_seed, args.teacher_seed) < 0:
        parser.error("invalid teacher learning rate or seed")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    args.output_dir = require_external(args.output_dir or ARTIFACT_ROOT / "outputs/mnist-latent/sweeps" / timestamp)
    args.bundle_dir, args.data_dir = require_external(args.bundle_dir), require_external(args.data_dir)
    args.eqvae_root, args.vae_checkpoint = args.eqvae_root.resolve(), args.vae_checkpoint.resolve()
    args.teacher_checkpoint = args.teacher_checkpoint.resolve() if args.teacher_checkpoint else None
    args.classifier_checkpoint = args.classifier_checkpoint.resolve() if args.classifier_checkpoint else None
    args.wandb_group = args.wandb_group or f"mnist-latent-sweep-{args.output_dir.name}"
    return args


def main(argv=None):
    args = parse_args(argv)
    jobs = load_jobs(args.grid, {"steps": args.steps} if args.steps is not None else {})
    devices = parse_devices(args.devices, args.max_parallel, verify=not args.dry_run)
    print(json.dumps({"num_jobs": len(jobs), "devices": devices, "workers_per_gpu": 1,
                      "image_policy": "end_only", "ode_steps": 100, "min_free_gb": args.min_free_gb,
                      "memory_policy": "per-student estimates" if args.min_free_gb is None else "fixed override",
                      "teacher": str(args.teacher_checkpoint) if args.teacher_checkpoint else "train once before sweep",
                      "output": str(args.output_dir), "grid": str(args.grid)}, indent=2), flush=True)
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / ".sweep.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise SystemExit("another sweep already uses this output directory")
    env = child_environment(args, args.output_dir / ".coordinator-runtime")
    os.environ.update(env)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(args.output_dir / "sweep.log"), logging.StreamHandler()])
    binding = {"jobs": jobs, "grid_sha256": file_hash(args.grid),
               "bundle_dir": str(args.bundle_dir), "data_dir": str(args.data_dir),
               "eqvae_root": str(args.eqvae_root), "vae_checkpoint": str(args.vae_checkpoint),
               "teacher_request": str(args.teacher_checkpoint) if args.teacher_checkpoint else None,
               "classifier_request": str(args.classifier_checkpoint) if args.classifier_checkpoint else None,
               "teacher_training": {key: value for key, value in vars(args).items() if key.startswith("teacher_") and not isinstance(value, Path)},
               "protocol": {key: getattr(args, key) for key in ("eval_samples", "test_samples", "eval_batch_size", "eval_seed",
                                                               "kid_subsets", "kid_subset_size", "is_splits", "metric_chunk_size")},
               "ode_steps": 100, "image_policy": "end_only", "wandb_group": args.wandb_group}
    previous = read_json(args.output_dir / "sweep_manifest.json")
    if previous and previous != binding:
        lock.close()
        raise SystemExit("sweep settings changed; use a fresh --output-dir")
    atomic_json(args.output_dir / "sweep_manifest.json", binding)
    for job in jobs:
        atomic_json(args.output_dir / "configs" / f"{job['name']}.json", job["settings"])
    old_handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    for signum in old_handlers:
        signal.signal(signum, interrupted)
    try:
        atomic_json(args.output_dir / "sweep_status.json", {"status": "preparing", "devices": devices})
        shared = prepare_shared(args, wait_for_device(args, devices))
        args.latent_dim = read_json(args.bundle_dir / "manifest.json")["dim"]
        atomic_json(args.output_dir / "sweep_status.json", {"status": "running", "devices": devices, "num_jobs": len(jobs)})
        failures = run_pool(args, shared, jobs, devices)
        selected = summarize(args, jobs)
        if selected:
            selected_hash = file_hash(selected["checkpoint"])
            record = read_json(args.output_dir / "selected_test.json")
            if record.get("checkpoint_sha256") != selected_hash or not Path(record.get("metrics", "")).is_file():
                output = fresh_path(args.output_dir / "selected-test")
                run_sync(args, evaluation_arguments(args, shared, wait_for_device(args, devices, selected["metrics"]), selected["checkpoint"], output,
                                                    f"{args.wandb_group}-selected-test", split="test"), "selected-test")
                atomic_json(args.output_dir / "selected_test.json", {"checkpoint_sha256": selected_hash,
                                                                      "metrics": str(output / "metrics.json")})
        atomic_json(args.output_dir / "sweep_status.json", {"status": "failed" if failures else "complete",
                                                            "devices": devices, "num_jobs": len(jobs), "failures": failures})
        if failures:
            raise SystemExit(f"{len(failures)} configurations failed; see summary.csv. Rerun with the same --output-dir and --retry-failed.")
        logging.info("Sweep complete: %s", args.output_dir / "summary.csv")
    except KeyboardInterrupt:
        atomic_json(args.output_dir / "sweep_status.json", {"status": "interrupted", "devices": devices})
        raise SystemExit(130)
    except Exception as error:
        atomic_json(args.output_dir / "sweep_status.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        lock.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    main()
