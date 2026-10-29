"""Shared frozen-ALI-teacher distillation; only student velocity MSE is optimized."""

from pathlib import Path

import torch
import yaml

from flow2pc.data.restore import restore_dataset
from flow2pc.distill.sampling import sample_batch, velocity_mse
from flow2pc.distill.student import (
    build_optimizer,
    build_student,
    initialize_source,
    learn_topology,
)
from flow2pc.distill.validation import prepare_validation, validate
from flow2pc.pretrain.models.alicfm.trainer import LocalLogger
from flow2pc.pretrain.runner import load_teacher
from flow2pc.utils.checkpoint import atomic_json, atomic_path, identity, sha256
from flow2pc.utils.seed import seed_everything


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
    saved_sampler = teacher.manifest.get("distillation_sampler")
    if saved_sampler is None:
        # Compatibility with the already-created CITE/Multi framework manifests.
        saved_sampler = (
            "mmot_middle" if dataset.metadata.name in ("cite", "multi", "eb") else None
        )
    configured = settings["sampler"]["name"]
    if configured != "teacher_interpolant" or settings["sampler"]["params"]:
        raise ValueError(
            "Use teacher_interpolant with empty params; its coupling is bound to the teacher manifest"
        )
    if saved_sampler not in ("mmot_middle", "border_ot", "independent"):
        raise ValueError(
            "Teacher manifest does not declare a supported query distribution"
        )
    return teacher, dataset, saved_sampler


def save(path, payload):
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)


def run(config):
    teacher, dataset, sampler = prepare(config)
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
    last_path = output / "checkpoints/last.pt"
    previous = (
        torch.load(last_path, map_location="cpu", weights_only=True)
        if last_path.exists()
        else None
    )
    if previous is not None and previous["binding"] != binding:
        raise ValueError("Resume teacher/data/model/training binding changed")
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
    if start > cfg["steps"]:
        atomic_json(
            output / "status.json",
            {
                "status": "complete",
                "steps": cfg["steps"],
                "best_selection_score": best,
                "best_step": best_step,
            },
        )
        return output / "checkpoints/best.pt"
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

        for step in range(start, cfg["steps"] + 1):
            student.train()
            batch = sample_batch(
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
