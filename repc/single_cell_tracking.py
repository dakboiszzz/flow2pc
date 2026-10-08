"""W&B tracking shared by the teacher evaluator and each student worker."""

from contextlib import contextmanager
import csv
import fcntl
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

from repc.single_cell import atomic_json


def dataset_run_name(contract, name):
    """Prefix new runs with their dataset and PCA dimension; resumes keep names."""
    prep = contract["preprocessing"]
    prefix = f"{prep['dataset']}{prep['n_pca_components_used']}d-"
    return name if name.startswith(prefix) else prefix + name


def add_tracking_arguments(parser, mode_default="online"):
    parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"), default=mode_default)
    parser.add_argument("--wandb-project", default="flow2pc")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default="cite5d-distillation")
    parser.add_argument("--wandb-service-wait", type=float, default=300)


@contextmanager
def startup_lock(output):
    """Serialize only W&B initialization; training remains concurrent."""
    path = Path(os.environ.get("FLOW2PC_WANDB_INIT_LOCK", str(output.parent / ".wandb-init.lock")))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def tracking_settings(wandb, timeout):
    settings = {"x_service_wait": timeout, "init_timeout": timeout}
    # GraphQL has a separate request timeout; increasing init_timeout alone
    # does not fix the "context deadline exceeded" resume-status failure.
    fields = getattr(wandb.Settings, "model_fields", {})
    graphql = {"x_graphql_timeout_seconds": min(60., timeout),
               "x_graphql_retry_max": 3, "x_graphql_retry_wait_min_seconds": 2.,
               "x_graphql_retry_wait_max_seconds": 10.}
    settings.update({key: value for key, value in graphql.items() if key in fields})
    return wandb.Settings(**settings)


def retryable_startup_error(wandb, error):
    message = str(error).lower()
    if any(text in message for text in (
        "401", "403", "permission denied", "invalid api key", "unauthorized", "forbidden")):
        return False
    if isinstance(error, wandb.errors.AuthenticationError):
        # W&B also uses AuthenticationError for verification transport and
        # IPC timeouts. The exception class alone does not imply a bad key.
        return any(text in message for text in (
            "service process is busy", "did not respond in time", "timeout", "timed out",
            "context deadline exceeded", "connection refused", "connection reset",
            "failed to connect", "network is unreachable", "temporary failure"))
    return True


def reset_failed_startup(wandb):
    try:
        # Each worker owns its SDK singleton; this resets that worker's failed
        # service without affecting other GPU workers or their W&B runs.
        wandb.teardown()
    except (Exception, SystemExit) as error:
        print(f"W&B startup cleanup reported {error}; retrying initialization", flush=True)


def init_tracking(args, output, config, job_type, name=None):
    if args.wandb_mode == "disabled":
        return None
    # _verify_login constructs its service RPC timeout from this environment
    # variable, separately from the Settings passed to wandb.init().
    os.environ.setdefault("WANDB_HTTP_TIMEOUT", "120")
    os.environ.setdefault("WANDB__SERVICE_WAIT", str(args.wandb_service_wait))
    os.environ.setdefault("WANDB_INIT_TIMEOUT", str(args.wandb_service_wait))
    import json
    import wandb

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "wandb_run.json"
    saved = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
    if saved and (saved["project"] != args.wandb_project
                  or saved["requested_entity"] != args.wandb_entity
                  or saved["group"] != args.wandb_group
                  or saved["mode"] != args.wandb_mode):
        raise ValueError("W&B destination/mode changed; use a new output directory")
    run_id = saved["id"] if saved else uuid4().hex[:16]
    run_name = saved["name"] if saved else name or output.name
    initialized = saved.get("initialized", True) if saved else False
    previous_attempt = bool(saved and saved.get("startup_attempts", 0))
    metadata = {"id": run_id, "name": run_name, "project": args.wandb_project,
                "requested_entity": args.wandb_entity, "entity": saved.get("entity") if saved else None,
                "group": args.wandb_group, "mode": args.wandb_mode,
                "url": saved.get("url") if saved else None, "initialized": initialized}
    print(f"W&B startup queued: project={args.wandb_project}, run={run_id}", flush=True)
    with startup_lock(output):
        for attempt in range(3):
            # Fresh runs need no resume-status request. Retry/resume with the
            # same ID in case an earlier request already created the remote run.
            resume = ("allow" if args.wandb_mode == "online"
                      and (initialized or previous_attempt or attempt > 0) else None)
            metadata.update({"startup_status": "starting", "startup_attempts": attempt + 1})
            atomic_json(metadata_path, metadata)
            try:
                run = wandb.init(
                    id=run_id, resume=resume, project=args.wandb_project,
                    entity=args.wandb_entity, group=args.wandb_group, name=run_name,
                    job_type=job_type, mode=args.wandb_mode, dir=str(output.resolve()),
                    config=config, settings=tracking_settings(wandb, args.wandb_service_wait),
                )
                break
            except wandb.errors.CommError as exc:
                metadata.update({"startup_status": "failed", "startup_error": str(exc)})
                atomic_json(metadata_path, metadata)
                if not retryable_startup_error(wandb, exc) or attempt == 2:
                    raise
                reset_failed_startup(wandb)
                delay = 5 * (2 ** attempt)
                print(f"W&B communication failure ({attempt + 1}/3): {exc}; "
                      f"retrying the same run in {delay}s", flush=True)
                time.sleep(delay)
    atomic_json(metadata_path, {"id": run.id, "name": run.name, "project": args.wandb_project,
                               "requested_entity": args.wandb_entity, "entity": run.entity,
                               "group": args.wandb_group, "mode": args.wandb_mode,
                               "url": run.url, "initialized": True, "startup_status": "ready"})
    for prefix, step in (("train", "train/step"),
                         ("validation", "train/step"), ("monitor", "train/step"),
                         ("eval", "eval/checkpoint_step")):
        run.define_metric(step)
        run.define_metric(f"{prefix}/*", step_metric=step)
    return run


def log_training(run, row):
    if run is None:
        return
    metrics = {"train/step": row["step"]}
    for key, value in row.items():
        if key == "step":
            continue
        name = "validation/" + key.removeprefix("validation_") if key.startswith("validation_") else "train/" + key
        metrics[name] = value
    # Use custom axes rather than W&B's internal step: resume/final
    # evaluation can otherwise cause smaller checkpoint steps to be discarded.
    run.log(metrics)


def emd_scalars(result, prefix):
    metrics = {}
    for score in result["scores"]:
        day = score["day"]
        for key in ("ode_emd", "direct_emd", "normalized_nll_nats_per_dim",
                    "num_reference", "num_predicted"):
            if key in score:
                metrics[f"{prefix}/day_{day}/{key}"] = score[key]
    for key in ("heldout_direct_emd", "heldout_ode_emd", "reference_heldout_ode_emd",
                "integration_seconds", "ode_nfe_total", "direct_sampling_seconds", "seconds",
                "num_samples"):
        if key in result:
            metrics[f"{prefix}/{key}"] = result[key]
    for key, value in result.get("validation", {}).items():
        metrics[f"{prefix}/validation/{key}"] = value
    if "reference_heldout_ode_emd" in result:
        baseline = result["reference_heldout_ode_emd"]
        for method in ("direct", "ode"):
            score = result[f"heldout_{method}_emd"]
            metrics[f"{prefix}/heldout_{method}_emd_minus_teacher"] = score - baseline
            if baseline > 0:
                metrics[f"{prefix}/heldout_{method}_emd_ratio_to_teacher"] = score / baseline
    return metrics


def log_files(run, label, files):
    """Upload attachments through the run file stream, without artifact APIs."""
    if run is None:
        return
    run_directory = Path(run.dir).resolve()
    uploaded = 0
    for name, path in files.items():
        source = Path(path).resolve()
        if source.is_file():
            destination = run_directory / "attachments" / label / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source != destination.resolve():
                shutil.copy2(source, destination)
            run.save(str(destination), base_path=str(run_directory), policy="now")
            uploaded += 1
    run.summary[f"uploads/{label}/files"] = uploaded


def log_evaluation(run, result, output, checkpoint_step=0):
    if run is None:
        return
    import wandb

    prefix = "eval" if result["model"] == "student" else "teacher"
    metrics = emd_scalars(result, prefix)
    if result["model"] == "student":
        metrics["eval/checkpoint_step"] = checkpoint_step
    # Tables implicitly call log_artifact(), which creates a second public
    # API client and triggers the busy-service verification failure in 0.30.
    # Publish scalar EMD first, then images and ordinary CSV/JSON/run files.
    run.log(metrics)
    run.summary.update(emd_scalars(result, prefix))
    table_path = output / "marginal_metrics.csv"
    with table_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["scores"][0]))
        writer.writeheader()
        writer.writerows(result["scores"])
    files = {"metrics.json": output / "metrics.json", "marginal_metrics.csv": table_path}
    images = {}
    for name in ("ode_marginals.png", "direct_marginals.png"):
        path = output / name
        if path.exists():
            images[f"{prefix}/{path.stem}"] = wandb.Image(str(path))
            files[name] = path
    if images:
        if result["model"] == "student":
            images["eval/checkpoint_step"] = checkpoint_step
        run.log(images)
    log_files(run, result["model"] + "-evaluation", files)
    run.summary[f"{prefix}/status"] = "complete"
