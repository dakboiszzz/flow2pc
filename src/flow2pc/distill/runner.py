"""Frozen-teacher distillation runners for ALI-CFM and MFM."""

import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from flow2pc.data.metric_registry import load_dataset
from flow2pc.data.restore import restore_dataset
from flow2pc.pretrain.models.alicfm.query_policy import resolve_sampler
from flow2pc.distill.sampling import sample_ali_batch, sample_mfm_batch, sample_mmfm_batch, velocity_mse
from flow2pc.distill.student import (
    build_optimizer,
    build_student,
    build_student_from_config,
    initialize_source,
    learn_topology,
    student_config,
)
from flow2pc.distill.validation import (
    prepare_validation,
    temporal_validation_contract,
    validate,
    validate_metric,
)
from flow2pc.utils.tracking import LocalLogger, log_population_metrics, metric_logger_config
from flow2pc.pretrain.models.mfm.checkpoint_adapter import load_teacher as load_mfm_teacher
from flow2pc.pretrain.runner import load_teacher
from flow2pc.utils.checkpoint import atomic_json, atomic_path, identity, restore_best_checkpoint, sha256
from flow2pc.utils.experiment import ExperimentConfig
from flow2pc.utils.experiment_protocols import dataset_tag, validation_scope
from flow2pc.utils.seed import seed_everything
from flow2pc.utils.mfm_config import validate_config, validate_resume_training


def prepare(config):
    source = config["teacher"]
    settings = config["distillation"]
    if config["student"]["name"] != "hclt":
        raise ValueError("This runner retains the real-data HCLT student")
    teacher = load_teacher(
        source["checkpoint"],
        settings.get("device", "cpu"),
        legacy_metadata=source.get("legacy_metadata"),
        generator_checkpoint=source.get("generator_checkpoint"),
    )
    dataset = restore_dataset(teacher.manifest["dataset"], teacher.preprocessing)
    saved_sampler = resolve_sampler(teacher.manifest, settings["sampler"])
    return teacher, dataset, saved_sampler


def save(path, payload):
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)


def _run_ali(config):
    teacher, dataset, sampler = prepare(config)
    from flow2pc.evaluation.figures import validate_resources

    validate_resources(dataset, config.get("evaluation", {"enabled": False}))
    cfg = config["distillation"]
    device = next(teacher.parameters()).device
    output = Path(config["output"]["dir"])
    if output.exists() and not cfg.get("resume", False):
        raise FileExistsError(
            "Output directory exists; use a fresh output.dir or distillation.resume: true"
        )
    if output.exists() and not (output / "checkpoints/last.pt").is_file():
        raise ValueError(
            "Partial output has no resume checkpoint; choose a fresh output directory"
        )
    seed_everything(config["seed"])
    pools = [
        dataset.get_marginal(i, "train").materialize()
        for i in range(len(dataset.metadata.model_times))
    ]
    validation_pools = [
        dataset.get_marginal(i, "val").materialize() for i in range(len(pools))
    ]
    clamp_derivative = dataset.metadata.name == "cell_tracking"
    binding = {
        "teacher_checkpoint": str(Path(config["teacher"]["checkpoint"]).resolve()),
        "teacher_sha256": sha256(config["teacher"]["checkpoint"]),
        "legacy_metadata_sha256": sha256(config["teacher"]["legacy_metadata"])
        if config["teacher"].get("legacy_metadata")
        else None,
        "generator_sha256": sha256(config["teacher"]["generator_checkpoint"])
        if config["teacher"].get("generator_checkpoint")
        else None,
        "legacy_provenance": teacher.manifest.get("legacy_provenance"),
        "dataset": dataset.contract(),
        "student": config["student"],
        "seed": config["seed"],
        "settings": {k: v for k, v in cfg.items() if k not in ("resume", "device")},
        "sampler": sampler,
        "objective": "teacher_velocity_MSE_only",
        "clamp_scale_derivative": clamp_derivative,
    }
    temporal_validation = temporal_validation_contract(dataset, cfg)
    if temporal_validation is not None:
        # Do not compare a resumed row-validation best score with the new
        # temporal criterion. Runs without holdouts retain their old binding.
        binding["validation_protocol"] = temporal_validation
    last_path = output / "checkpoints/last.pt"
    previous = (
        torch.load(last_path, map_location="cpu", weights_only=True)
        if last_path.exists()
        else None
    )
    if previous is not None:
        if previous["binding"].get("validation_protocol") != temporal_validation:
            raise ValueError(
                "Resume validation protocol changed; use a fresh output.dir "
                "rather than comparing scores from different validation protocols"
            )
        if previous["binding"] != binding:
            raise ValueError("Resume teacher/data/model/training binding changed")
        if previous.get("best_step") is not None:
            selected_path = output / "checkpoints/best.pt"
            if not selected_path.is_file():
                raise ValueError("Resume selected weights are missing: restore checkpoints/best.pt")
            selected = torch.load(selected_path, map_location="cpu", weights_only=True)
            if (selected.get("binding") != binding
                    or selected.get("step") != previous["best_step"]
                    or selected.get("best_selection_score") != previous["best_selection_score"]):
                raise ValueError("Resume best.pt does not match the saved selection history")
    parents = (
        previous["model_config"]["parents"]
        if previous
        else (
            [-1]
            if dataset.dim == 1
            else [-1, 0]
            if dataset.dim == 2
            else learn_topology([pools[i] for i in dataset.observed])
        )
    )
    student = build_student(
        parents, config["student"]["params"], clamp_derivative=clamp_derivative
    ).to(device)
    optimizer, scheduler = build_optimizer(student, cfg["lr"], cfg["steps"])
    rng = torch.Generator().manual_seed(config["seed"] + 1000)
    best, best_step, start = float("inf"), None, 1
    if previous:
        student.load_state_dict(previous["state_dict"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        rng.set_state(previous["rng"])
        torch.set_rng_state(previous["torch_rng"])
        if device.type == "cuda" and previous.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(previous["cuda_rng"], device)
        best, best_step, start = (
            previous["best_selection_score"],
            previous["best_step"],
            previous["step"] + 1,
        )
    else:
        initialize_source(
            student, pools[0], torch.Generator().manual_seed(config["seed"] + 20000)
        )
    groups, energy = prepare_validation(
        teacher, dataset, validation_pools, cfg, config["seed"] + 100000, sampler
    )
    output.mkdir(parents=True, exist_ok=True)
    with atomic_path(output / "resolved_config.yaml") as temporary:
        temporary.write_text(yaml.safe_dump(config, sort_keys=False))
    model_config = {
        "name": "hclt",
        "parents": parents,
        **config["student"]["params"],
        "fourier_zero_center": False,
        "time_conditioning": "fourier",
        "clamp_scale_derivative": clamp_derivative,
    }
    atomic_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "artifact_type": "student",
            **binding,
            "model_config": model_config,
            "binding_sha256": identity(binding),
        },
    )
    atomic_json(output / "status.json", {"status": "running"})
    logger = None
    try:
        logger = LocalLogger(output, config, append=previous is not None)

        def checkpoint(step):
            return {
                "state_dict": {
                    k: v.detach().cpu() for k, v in student.state_dict().items()
                },
                "binding": binding,
                "model_config": model_config,
                "step": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": rng.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else None,
                "best_selection_score": best,
                "best_step": best_step,
            }

        # Allow an OOM on the first update to resume without abandoning this run.
        if previous is None:
            save(last_path, checkpoint(0))
        for step in range(start, cfg["steps"] + 1):
            student.train()
            batch = sample_ali_batch(
                teacher, dataset, pools, cfg["batch_size"], rng, sampler
            )
            optimizer.zero_grad(set_to_none=True)
            loss = velocity_mse(student, teacher, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite distillation velocity MSE")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                student.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            row = {
                "train/step": step,
                "train/velocity_mse": float(loss.detach()),
                "train/gradient_norm_before_clip": float(grad),
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }
            if (
                step == 1
                or step % cfg.get("validate_every", 1000) == 0
                or step == cfg["steps"]
            ):
                scores = validate(
                    student, groups, energy, cfg.get("validation_batch_size", 128)
                )
                if not all(torch.isfinite(torch.tensor(v)) for v in scores.values()):
                    raise FloatingPointError("Non-finite validation velocity error")
                row.update({"validation/" + k: v for k, v in scores.items()})
                if scores["selection_score"] < best:
                    best, best_step = scores["selection_score"], step
                    save(output / "checkpoints/best.pt", checkpoint(step))
                if cfg.get("validation_population_metrics", True):
                    from flow2pc.evaluation.metric_flow import evaluate as population_evaluate, from_ali_dataset
                    view = from_ali_dataset(dataset)
                    scope = validation_scope(dataset, cfg)
                    diagnostics = population_evaluate(student, teacher, view, {
                        "samples": cfg.get("validation_batch_size", 128),
                        "source_split": "all" if scope == "heldout_times" else "validation",
                        "reference_split": "validation",
                        "time_indices": view.heldout_indices if scope == "heldout_times" else view.observed_indices,
                        "metric_space": "dataset", "solver": "dopri5", "seed": 2026,
                    }, device)
                    diagnostics["checkpoint_step"] = step
                    log_population_metrics(logger, diagnostics, namespace="validation")
                print(
                    f"step={step} velocity_mse={float(loss.detach()):.6g} validation_score={scores['selection_score']:.6g}",
                    flush=True,
                )
            if (
                step == 1
                or step % cfg.get("checkpoint_every", 1000) == 0
                or step == cfg["steps"]
                or "validation/selection_score" in row
            ):
                save(last_path, checkpoint(step))
            if (
                step == 1
                or step % cfg.get("log_every", 100) == 0
                or "validation/selection_score" in row
            ):
                logger.log(row)
        if config.get("evaluation", {}).get("enabled", False):
            from flow2pc.evaluation.runner import evaluate
            from flow2pc.evaluation.metric_flow import _numerical_failure

            selected = torch.load(
                output / "checkpoints/best.pt", map_location="cpu", weights_only=True
            )
            student.load_state_dict(selected["state_dict"], strict=True)
            reference = (
                Path(config["teacher"]["checkpoint"]).resolve().parent.parent / "eval"
            )
            try:
                evaluate(
                    teacher,
                    dataset,
                    config["evaluation"],
                    output / "eval",
                    student=student,
                    logger=logger,
                    checkpoint_step=selected["step"],
                    reference_dir=reference,
                )
            except (FloatingPointError, RuntimeError, AssertionError) as error:
                if not _numerical_failure(error):
                    raise
                atomic_json(output / "eval/ode_error.json", {"error": str(error)})
                logger.log({"eval/historical_ode_error": str(error)})
        if config.get("evaluation", {}).get("enabled", False):
            from flow2pc.evaluation.metric_flow import evaluate as population_evaluate, from_ali_dataset
            population_report = population_evaluate(student, teacher, from_ali_dataset(dataset), {
                "samples": config["evaluation"].get("metric_samples", 512) or max(len(pool) for pool in pools),
                "batch_size": config["evaluation"].get("batch_size", 64),
                "metric_space": "dataset", "solver": config["evaluation"].get("solver", "dopri5"),
                "tolerance": config["evaluation"].get("tolerance", 1e-5),
                "max_nfe": config["evaluation"].get("max_nfe", 50000),
                "ode_substeps": max(1, math.ceil(config["evaluation"].get("ode_steps", 100) / (len(dataset.metadata.model_times) - 1))),
                "source_split": {"val": "validation"}.get(config["evaluation"].get("source_split"), config["evaluation"].get("source_split", "all")),
                "reference_split": {"val": "validation"}.get(config["evaluation"].get("reference_split"), config["evaluation"].get("reference_split", "all")),
                "time_indices": config["evaluation"].get("time_indices", list(range(len(dataset.metadata.model_times)))),
                "seed": config["evaluation"].get("seed", 2026),
            }, device, output / "eval/population")
            population_report["checkpoint_step"] = selected["step"]
            atomic_json(output / "eval/population/evaluation.json", population_report)
            # Historical rollout scores/overlays remain available; name the
            # shared benchmark explicitly instead of silently replacing them.
            metrics_path = output / "eval/metrics.json"
            if metrics_path.is_file():
                metrics = json.loads(metrics_path.read_text())
                metrics["population_diagnostics"] = "population/evaluation.json"
                atomic_json(metrics_path, metrics)
            log_population_metrics(logger, population_report, output=output / "eval/population")
            existing_media = output / "eval/media.json"
            media = json.loads(existing_media.read_text()) if existing_media.is_file() else {}
            media.update({"pc_ode_" + key: value for key, value in
                          json.loads((output / "eval/population/media.json").read_text()).items()})
            atomic_json(existing_media, media)
        if teacher.training or any(p.requires_grad for p in teacher.parameters()):
            raise RuntimeError("Frozen teacher contract was changed")
        if logger.run is not None:
            import wandb

            artifact = wandb.Artifact(
                f"{dataset.metadata.name}-hclt-student", type="model"
            )
            for name in (
                "checkpoints/best.pt",
                "checkpoints/last.pt",
                "manifest.json",
                "resolved_config.yaml",
            ):
                artifact.add_file(str(output / name))
            logger.run.log_artifact(artifact)
        logger.close()
        logger = None
        atomic_json(
            output / "status.json",
            {
                "status": "complete",
                "steps": cfg["steps"],
                "best_selection_score": best,
                "best_step": best_step,
            },
        )
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "failed", "error": str(error)})
        if logger is not None:
            logger.close(failed=True)
        raise
    return output / "checkpoints/best.pt"


METRIC_FIELDS = ["step", "loss", "velocity_mse", "gradient_norm", "validation_loss"]


def metrics_path(output):
    """Preserve legacy CSVs and append only to a matching velocity-only schema."""
    path = output / "metrics.csv"
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as stream:
            header = next(csv.reader(stream))
        if "source_nll_per_dimension" in header:
            path = output / "metrics_velocity.csv"
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as stream:
            if next(csv.reader(stream)) != METRIC_FIELDS:
                raise ValueError(f"Incompatible metrics header: {path}")
    return path


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dict__"):
        return _json(vars(value))
    return value


def _identity(teacher, data):
    if teacher.metadata.get("method") == "mmfm":
        return {
            "method": "mmfm", "flow": teacher.metadata["flow_sha256"],
            "data": data.source_info["snapshots_sha256"], "split": data.split_indices,
            "normalization": data.normalization, "heldout": data.heldout_indices,
            "times": data.times.tolist(), "matcher": teacher.metadata["matcher"],
            "query_protocol": teacher.metadata["query_protocol"],
        }
    return {
        "flow": teacher.metadata["flow_sha256"],
        "geopath": teacher.metadata["geopath_sha256"],
        "data": data.source_info["snapshots_sha256"],
        "split": data.split_indices,
        "normalization": data.normalization,
        "heldout": data.heldout_indices,
    }


def _rng():
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def load_resources(config, device):
    """Load explicitly supplied teacher files, then its matching data coordinates."""
    if config.teacher["name"] == "mmfm":
        from flow2pc.pretrain.models.mmfm.adapter import load_teacher, load_teacher_data

        teacher = load_teacher({**config.teacher.get("options", {}),
                                "checkpoint_path": config.teacher["checkpoint_path"],
                                "dataset": config.dataset["name"]}, device)
        return teacher, load_teacher_data(teacher, config.dataset)
    training = config.training
    seed = int(training.get("seed", 42))
    teacher_options = {
        **config.teacher.get("options", {}),
        "name": config.teacher["name"],
        "checkpoint_path": config.teacher["checkpoint_path"],
        "dataset": config.dataset["name"],
    }
    teacher = load_mfm_teacher(teacher_options, device)
    data_options = {
        **config.dataset.get("options", {}),
        "name": config.dataset["name"],
        "batch_size": int(training.get("batch_size", 128)),
        "seed": seed,
    }
    if config.dataset.get("data_dir") is not None:
        data_options["data_dir"] = config.dataset["data_dir"]
    data = load_dataset(data_options, {"teacher_metadata": teacher.metadata})
    return teacher, data


def _run_metric(config):
    from flow2pc.evaluation.metric_flow import evaluate

    validate_config(config)
    is_mmfm = config.teacher["name"] == "mmfm"
    sample_batch = sample_mmfm_batch if is_mmfm else sample_mfm_batch
    training = config.training
    seed = int(training.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    name = training.get("device", "auto")
    device = torch.device(
        "cuda"
        if name == "auto" and torch.cuda.is_available()
        else "cpu"
        if name == "auto"
        else name
    )
    output = Path(
        training.get(
            "output_dir",
            f"outputs/runs/metric/{config.dataset['name']}_{config.student.get('pc_structure', 'hclt')}",
        )
    )
    resume_path = training.get("resume_checkpoint")
    if output.exists() and any(output.iterdir()) and not resume_path:
        raise FileExistsError(f"Refusing to overwrite nonempty run: {output}")
    teacher, data = load_resources(config, device)
    identity = _identity(teacher, data)
    validation_protocol = {"version": 2, "scope": validation_scope(data, training),
                           "heldout_indices": list(data.heldout_indices),
                           "selection": "velocity_MSE_only"}
    checkpoint = (
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_path
        else None
    )
    if checkpoint and checkpoint.get("identity") != identity:
        raise ValueError(
            "Resume teacher, data, normalization, or split differs from checkpoint"
        )
    if checkpoint and checkpoint.get("validation_protocol", {**validation_protocol, "scope": "observed_rows"}) != validation_protocol:
        raise ValueError("Resume validation protocol changed; use a fresh output directory")
    if checkpoint and checkpoint.get("student_settings") != config.student:
        raise ValueError("Resume student configuration differs from checkpoint")
    if checkpoint:
        validate_resume_training(checkpoint, training)
    model_config = (
        checkpoint["model_config"]
        if checkpoint
        else student_config(config.student, data, seed)
    )
    student = build_student_from_config(model_config).to(device)
    if not checkpoint and data.source_info.get("format") == "mfm_snapshots_v1":
        from flow2pc.distill.student import initialize_snapshot_source

        initialize_snapshot_source(
            student, data.train_marginals[0], torch.Generator().manual_seed(seed + 20000)
        )
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=float(training.get("lr", 0.001))
    )
    start, best = 0, math.inf
    if checkpoint:
        student.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start, best = checkpoint["step"], checkpoint["best_validation_loss"]
        _restore_rng(checkpoint["rng"])
    total = int(training.get("steps", 20000))
    if start >= total:
        raise ValueError("steps must exceed the resumed checkpoint step")
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    if checkpoint:
        restore_best_checkpoint(checkpoint, resume_path, output)
    manifest = {
        "source_info": data.source_info,
        "normalization": data.normalization,
        "split_indices": data.split_indices,
        "times": data.times.tolist(),
    }
    (output / "data_manifest.json").write_text(
        json.dumps(_json(manifest), indent=2) + "\n"
    )
    (output / "config.json").write_text(
        json.dumps(
            _json(
                {
                    "experiment": asdict(config),
                    "teacher_metadata": teacher.metadata,
                    "model_config": model_config,
                }
            ),
            indent=2,
        )
        + "\n"
    )
    if model_config["name"] == "hclt":
        torch.save(
            {
                "parents": model_config["parents"],
                "identity": identity,
                "estimator": model_config["topology_estimator"],
            },
            output / "topology.pt",
        )
    logger = LocalLogger(output, metric_logger_config(asdict(config),
                         method="mmfm" if is_mmfm else "mfm", stage="distill",
                         dataset=dataset_tag(data)),
                         append=checkpoint is not None)

    def save(filename, step):
        with atomic_path(output / "checkpoints" / filename) as temporary:
            torch.save({
                "model_type": "mmfm_pc_distill" if is_mmfm else "metric_pc_distill",
                "model_config": model_config,
                "student_settings": config.student,
                "state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "best_validation_loss": best,
                "validation_protocol": validation_protocol,
                "identity": identity,
                "rng": _rng(),
                "training_config": asdict(config),
                "teacher_metadata": teacher.metadata,
                "data_manifest": manifest,
            }, temporary)

    log_path = metrics_path(output)
    needs_header = not log_path.exists() or log_path.stat().st_size == 0
    batch = int(training.get("batch_size", 128))
    try:
        atomic_json(output / "status.json", {"status": "running"})
        if checkpoint is None:
            save("last.pt", 0)
        with log_path.open("a", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=METRIC_FIELDS,
            )
            if needs_header:
                writer.writeheader()
            for step in range(start + 1, total + 1):
                student.train()
                optimizer.zero_grad(set_to_none=True)
                query = sample_batch(teacher, data, batch, device)
                loss = velocity_mse(student, teacher, query)
                components = {"velocity_mse": loss.detach().item()}
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(),
                    float(training.get("gradient_clip", 1.0)),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                validation = None
                if (
                    step == 1
                    or step % int(training.get("validate_every", 1000)) == 0
                    or step == total
                ):
                    student.eval()
                    validation = validate_metric(student, teacher, data, training,
                                                 device, sample_batch)
                    if validation < best:
                        best = validation
                        save("best.pt", step)
                    if training.get("validation_population_metrics", True):
                        scope = validation_scope(data, training)
                        settings = {**config.evaluation,
                                    "samples": int(training.get("validation_batch_size", batch)),
                                    "source_split": "all" if scope == "heldout_times" else "validation",
                                    "reference_split": "validation",
                                    "time_indices": data.heldout_indices if scope == "heldout_times" else data.observed_indices}
                        diagnostics = evaluate(student, teacher, data, settings, device)
                        diagnostics["checkpoint_step"] = step
                        log_population_metrics(logger, diagnostics, namespace="validation")
                if (
                    step % int(training.get("log_every", 100)) == 0
                    or validation is not None
                    or step == total
                ):
                    row = {
                        "step": step,
                        "loss": loss.item(),
                        **components,
                        "gradient_norm": float(norm),
                        "validation_loss": validation,
                    }
                    writer.writerow(row)
                    stream.flush()
                    logger.log({k: v for k, v in row.items() if v is not None})
                    print(
                        f"step {step}/{total}: loss={loss.item():.6g}, validation={validation}",
                        flush=True,
                    )
                if (
                    step % int(training.get("checkpoint_every", 5000)) == 0
                    or step == total
                    or validation is not None
                ):
                    save(f"step_{step}.pt", step)
                    save("last.pt", step)
        if config.evaluation.get("enabled", True):
            selected_path = output / "checkpoints/best.pt"
            selected = torch.load(selected_path, map_location="cpu", weights_only=False)
            if selected["identity"] != identity:
                raise ValueError("Selected student has a different teacher/data identity")
            student.load_state_dict(selected["state_dict"], strict=True)
            report = evaluate(student, teacher, data, config.evaluation, device, output)
            report.update(checkpoint_step=selected["step"], checkpoint=str(selected_path.resolve()))
            log_population_metrics(logger, report, output=output)
            (output / "evaluation.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
        atomic_json(output / "status.json", {"status": "complete", "best_validation_loss": best})
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "failed", "error": str(error)})
        logger.close(failed=True)
        raise
    else:
        logger.close()
    return output


def run(config):
    """Dispatch distillation using the existing method-specific config contract."""
    if isinstance(config, ExperimentConfig):
        return _run_metric(config)
    if isinstance(config, dict):
        return _run_ali(config)
    raise TypeError(f"Unsupported distillation config type: {type(config).__name__}")
