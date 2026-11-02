"""Distill Metric Flow Matching using grouped configs and native geopaths."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from flow2pc.data.metric_registry import load_dataset, validate_combination
from flow2pc.evaluation.metric_flow import evaluate, evaluation_seed
from flow2pc.pretrain.models.mfm.checkpoint_adapter import load_teacher


from flow2pc.utils.mfm_config import validate_config


def build_student(model_config):
    from flow2pc.pc.hclt_pc import TimeDependentHCLTPC
    from flow2pc.pc.gaussian_pc import TimeDependentGaussianPC

    if model_config["name"] == "gaussian":
        return TimeDependentGaussianPC(
            model_config["num_components"],
            model_config["dim"],
            model_config["time_order"],
            polynomial_zero_center=False,
        )
    return TimeDependentHCLTPC(
        model_config["parents"],
        num_latents=model_config["num_latents"],
        time_order=model_config["time_order"],
        sum_layer="dense",
        time_conditioning="fourier",
        time_embedding_dim=model_config["time_embedding_dim"],
        time_feature_seed=model_config.get("time_feature_seed", 0),
        fourier_zero_center=False,
    )


def student_config(settings, data, seed):
    structure = settings.get("pc_structure", "hclt")
    config = {
        "name": structure,
        "dim": data.dim,
        "sample_shape": list(data.sample_shape),
        "time_order": int(settings.get("time_order", 4)),
    }
    if structure == "gaussian":
        config.update(
            num_components=int(settings.get("num_components", 64)),
            polynomial_zero_center=False,
        )
    else:
        if settings.get("topology", "learned") == "balanced":
            parents = [-1] + [(i - 1) // 2 for i in range(1, data.dim)]
        else:
            from flow2pc.pc.metric_topology import parents_from_mutual_information

            matrix = torch.cat(
                [data.train_marginals[i].flatten(1) for i in data.observed_indices]
            )
            indices = torch.randperm(len(matrix))[
                : int(settings.get("topology_samples", 2048))
            ]
            matrix = matrix[indices].double()
            centered = matrix - matrix.mean(0)
            norm = centered.square().sum(0).sqrt().clamp_min(1e-12)
            standardized = centered / norm
            correlations = (standardized.T @ standardized).clamp(-1 + 1e-8, 1 - 1e-8)
            information = -0.5 * torch.log1p(-correlations.square())
            information.fill_diagonal_(0)
            parents = (
                [-1]
                if data.dim == 1
                else parents_from_mutual_information(information.float())
            )
        config.update(
            parents=parents,
            num_latents=int(settings.get("num_latents", 64)),
            sum_layer="dense",
            time_conditioning="fourier",
            fourier_zero_center=False,
            time_embedding_dim=int(settings.get("time_embedding_dim", 64)),
            time_feature_seed=seed,
            topology=settings.get("topology", "learned"),
            topology_estimator="balanced_fixed"
            if settings.get("topology") == "balanced"
            else "gaussian_mutual_information",
        )
    return config


def _draw(values, count, device):
    return values[torch.randint(len(values), (count,))].to(device)


def distillation_loss(
    student,
    teacher,
    data,
    batch_size,
    device,
    source_nll_weight=1.0,
    *,
    validation=False,
    interval_index=None,
):
    frames = data.validation_marginals if validation else data.train_marginals
    observed = data.observed_indices
    interval = (
        int(torch.randint(len(observed) - 1, ()).item())
        if interval_index is None
        else interval_index
    )
    left, right = observed[interval : interval + 2]
    x0, x1 = (
        _draw(frames[left], batch_size, device),
        _draw(frames[right], batch_size, device),
    )
    x0, x1 = teacher.path_sampler.couple(x0, x1)
    times, samples = teacher.path_sampler.sample(
        x0, x1, float(data.times[left]), float(data.times[right])
    )
    with torch.no_grad():
        target = teacher.velocity(samples, times).flatten(1)
    if not torch.isfinite(target).all() or not torch.isfinite(samples).all():
        raise FloatingPointError("Nonfinite metric teacher velocity/geopath points")
    mse = nn.functional.mse_loss(student.velocity(samples.flatten(1), times), target)
    source = _draw(frames[0], batch_size, device).flatten(1)
    source_nll = (
        -student.log_prob(source, torch.zeros(batch_size, device=device)).mean()
        / data.dim
    )
    loss = mse + source_nll_weight * source_nll
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite metric distillation loss")
    return loss, {
        "velocity_mse": mse.detach().item(),
        "source_nll_per_dimension": source_nll.detach().item(),
    }


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
    training = config.training
    seed = int(training.get("seed", 42))
    teacher_options = {
        **config.teacher.get("options", {}),
        "name": config.teacher["name"],
        "checkpoint_path": config.teacher["checkpoint_path"],
        "dataset": config.dataset["name"],
    }
    teacher = load_teacher(teacher_options, device)
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


def run(config):
    validate_config(config)
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
    checkpoint = (
        torch.load(resume_path, map_location="cpu", weights_only=False)
        if resume_path
        else None
    )
    if checkpoint and checkpoint.get("identity") != identity:
        raise ValueError(
            "Resume teacher, data, normalization, or split differs from checkpoint"
        )
    if checkpoint and checkpoint.get("student_settings") != config.student:
        raise ValueError("Resume student configuration differs from checkpoint")
    if checkpoint and checkpoint.get("source_nll_weight") != float(
        training.get("source_nll_weight", 1.0)
    ):
        raise ValueError("Resume source NLL weight differs from checkpoint")
    model_config = (
        checkpoint["model_config"]
        if checkpoint
        else student_config(config.student, data, seed)
    )
    student = build_student(model_config).to(device)
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
    tracker = None
    if config.wandb.get("mode", "disabled") != "disabled":
        import wandb

        tracker = wandb.init(
            project=config.wandb.get("project", "flow2pc"),
            mode=config.wandb["mode"],
            group=config.wandb.get("group", "metric-flow"),
            name=config.wandb.get("run_name"),
            config=_json(asdict(config)),
            dir=str(output.resolve()),
        )

    def save(filename, step):
        torch.save(
            {
                "model_type": "metric_pc_distill",
                "model_config": model_config,
                "student_settings": config.student,
                "state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step,
                "best_validation_loss": best,
                "identity": identity,
                "rng": _rng(),
                "source_nll_weight": float(training.get("source_nll_weight", 1.0)),
                "training_config": asdict(config),
                "teacher_metadata": teacher.metadata,
                "data_manifest": manifest,
            },
            output / "checkpoints" / filename,
        )

    log_path = output / "metrics.csv"
    needs_header = not log_path.exists()
    batch = int(training.get("batch_size", 128))
    weight = float(training.get("source_nll_weight", 1.0))
    try:
        with log_path.open("a", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "step",
                    "loss",
                    "velocity_mse",
                    "source_nll_per_dimension",
                    "gradient_norm",
                    "validation_loss",
                ],
            )
            if needs_header:
                writer.writeheader()
            for step in range(start + 1, total + 1):
                student.train()
                optimizer.zero_grad(set_to_none=True)
                loss, components = distillation_loss(
                    student, teacher, data, batch, device, weight
                )
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(),
                    float(training.get("gradient_clip", 1.0)),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                validation = None
                if (
                    step % int(training.get("validate_every", 1000)) == 0
                    or step == total
                ):
                    student.eval()
                    with torch.no_grad(), evaluation_seed(1234, device):
                        validation = float(
                            np.mean(
                                [
                                    distillation_loss(
                                        student,
                                        teacher,
                                        data,
                                        batch,
                                        device,
                                        weight,
                                        validation=True,
                                        interval_index=index,
                                    )[0].item()
                                    for index in range(len(data.observed_indices) - 1)
                                ]
                            )
                        )
                    if validation < best:
                        best = validation
                        save("best.pt", step)
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
                    if tracker:
                        tracker.log(
                            {k: v for k, v in row.items() if v is not None}, step=step
                        )
                    print(
                        f"step {step}/{total}: loss={loss.item():.6g}, validation={validation}",
                        flush=True,
                    )
                if (
                    step % int(training.get("checkpoint_every", 5000)) == 0
                    or step == total
                ):
                    save(f"step_{step}.pt", step)
                    save("last.pt", step)
        if config.evaluation.get("enabled", True):
            report = evaluate(student, teacher, data, config.evaluation, device, output)
            (output / "evaluation.json").write_text(
                json.dumps(report, indent=2, allow_nan=False) + "\n"
            )
    finally:
        if tracker:
            tracker.finish()
    return output


def main():
    from flow2pc.cli import main as config_main

    config_main()


if __name__ == "__main__":
    main()
