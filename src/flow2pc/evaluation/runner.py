"""Post-training teacher/student ODE evaluation; independent of PC implementations."""

import csv
import hashlib
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from flow2pc.data.base import heldout_indices
from flow2pc.evaluation.figures import TRACKING_FRAMES, save_figures, validate_resources
from flow2pc.evaluation.ode import (
    emd,
    fixed_subset,
    metric_coordinates,
    raw_coordinates,
    rollout,
)
from flow2pc.utils.checkpoint import atomic_json, atomic_path, identity


def population(dataset, index, split):
    if split == "all":
        if hasattr(dataset, "clouds"):
            return dataset.clouds[index]
        if dataset.preprocessing.get("kind") == "all_marginals_sequential_float32":
            return dataset.get_marginal(index, "all").materialize()
        return torch.cat(
            [
                dataset.get_marginal(index, name).materialize()
                for name in ("train", "val", "test")
            ]
        )
    return dataset.get_marginal(index, split).materialize()


def save_tensor(path, payload):
    with atomic_path(path) as temporary:
        torch.save(payload, temporary)


def evaluate(
    teacher,
    dataset,
    settings,
    output,
    *,
    student=None,
    logger=None,
    checkpoint_step=None,
    reference_dir=None,
):
    """Compare ODE flows from identical empirical source particles and model times.

    Settings explicitly record all-population versus held-out-row evaluation.
    Teacher/student observation EMD and paired flow error are separate metrics.
    """
    if settings.get("protocol") == "native_otpfm":
        raise ValueError("Use the OTP-FM native finite-interval evaluator for native_otpfm; shared ODE metrics are a separate protocol")
    validate_resources(dataset, settings)
    source = fixed_subset(
        population(dataset, 0, settings["source_split"]),
        settings["num_samples"],
        settings["seed"],
    )
    if not len(source):
        raise ValueError("Evaluation source split is empty")
    times = torch.tensor(dataset.metadata.model_times, dtype=torch.float32)
    protocol = {k: v for k, v in settings.items() if k not in ("enabled", "image_dir")}
    protocol.update(
        source_index=0,
        initial_law="same_observed_source_cells_for_teacher_and_student",
        source_count=len(source),
        model_times=list(dataset.metadata.model_times),
        source_sha256=hashlib.sha256(source.contiguous().numpy().tobytes()).hexdigest(),
        emd_formula="uniform-mass W1 with Euclidean costs; full feature dimension",
        units=(
            "standardized_feature_coordinates"
            if settings["metric_space"] == "historical"
            and dataset.preprocessing.get("whiten")
            else "raw_feature_coordinates"
        ),
    )
    binding = identity(
        {
            "teacher": teacher.manifest,
            "dataset": dataset.contract(),
            "protocol": protocol,
        }
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output / "status.json", {"status": "running", "binding_sha256": binding}
    )
    try:
        teacher_path, teacher_nfe = None, None
        cache_directories = [Path(reference_dir)] if reference_dir is not None else []
        cache_directories.append(output)
        for directory in cache_directories:
            cache = directory / "teacher_predictions.pt"
            if cache.is_file():
                cached = torch.load(cache, map_location="cpu", weights_only=True)
                if cached.get("binding_sha256") == binding:
                    teacher_path, teacher_nfe = cached["teacher_ode"], cached["nfe"]
                    if (
                        teacher_path.shape != (len(times), len(source), dataset.dim)
                        or not torch.isfinite(teacher_path).all()
                    ):
                        raise ValueError("Invalid cached teacher rollout")
                    break
        if teacher_path is None:
            teacher_path, teacher_nfe = rollout(teacher, source, times, settings)
        save_tensor(
            output / "teacher_predictions.pt",
            {
                "binding_sha256": binding,
                "teacher_ode": teacher_path,
                "times": times,
                "nfe": teacher_nfe,
            },
        )
        student_path, student_nfe = None, None
        if student is not None:
            student_path, student_nfe = rollout(
                student, source, times, settings, student=True
            )
            save_tensor(
                output / "predictions.pt",
                {
                    "binding_sha256": binding,
                    "times": times,
                    "student_ode": student_path,
                    "teacher_ode": teacher_path,
                },
            )
        indices = (
            list(TRACKING_FRAMES)
            if dataset.metadata.name == "cell_tracking"
            else list(range(len(times)))
        )
        rows = []
        for index in indices:
            real = fixed_subset(
                population(dataset, index, settings["reference_split"]),
                settings["metric_samples"],
                settings["seed"] + index + 1,
            )
            if not len(real):
                continue
            predicted_teacher = fixed_subset(
                teacher_path[index], settings["metric_samples"], settings["seed"] + 1000
            )
            real = metric_coordinates(
                real, dataset.preprocessing, settings["metric_space"]
            )
            predicted_teacher = metric_coordinates(
                predicted_teacher, dataset.preprocessing, settings["metric_space"]
            )
            row = {
                "index": index,
                "physical_time": dataset.metadata.physical_times[index],
                "model_time": float(times[index]),
                "held_out": index in heldout_indices(dataset),
                "num_reference": len(real),
                "num_predicted": len(predicted_teacher),
                "teacher_emd": emd(real, predicted_teacher),
            }
            if student_path is not None:
                predicted_student = fixed_subset(
                    student_path[index],
                    settings["metric_samples"],
                    settings["seed"] + 1000,
                )
                predicted_student = metric_coordinates(
                    predicted_student, dataset.preprocessing, settings["metric_space"]
                )
                row.update(
                    student_emd=emd(real, predicted_student),
                    student_teacher_emd=emd(predicted_teacher, predicted_student),
                )
            rows.append(row)
        noninitial = [row for row in rows if row["index"] != 0]
        if not noninitial:
            raise ValueError(
                "No noninitial evaluation populations in the selected reference split"
            )
        scalars = {
            "teacher_mean_emd": float(np.mean([r["teacher_emd"] for r in noninitial])),
            "teacher_final_emd": next(
                (r["teacher_emd"] for r in rows if r["index"] == len(times) - 1), None
            ),
            "teacher_ode_nfe": teacher_nfe,
        }
        heldout_rows = [row for row in rows if row["held_out"]]
        heldout = next(iter(heldout_rows), None)
        if heldout is not None:
            scalars["teacher_heldout_emd"] = float(np.mean([r["teacher_emd"] for r in heldout_rows]))
        if student_path is not None:
            difference = raw_coordinates(
                student_path, dataset.preprocessing
            ) - raw_coordinates(teacher_path, dataset.preprocessing)
            paired_rmse = difference.square().sum(-1).mean(-1).sqrt()
            late = times >= 0.8
            scalars.update(
                student_mean_emd=float(np.mean([r["student_emd"] for r in noninitial])),
                student_final_emd=next(
                    (r["student_emd"] for r in rows if r["index"] == len(times) - 1),
                    None,
                ),
                student_teacher_mean_emd=float(
                    np.mean([r["student_teacher_emd"] for r in noninitial])
                ),
                teacher_student_final_paired_rmse_raw=float(paired_rmse[-1]),
                teacher_student_late_paired_rmse_raw=float(paired_rmse[late].mean()),
                teacher_student_max_paired_rmse_raw=float(paired_rmse.max()),
                student_ode_nfe=student_nfe,
            )
            if heldout is not None:
                scalars["student_heldout_emd"] = float(np.mean([r["student_emd"] for r in heldout_rows]))
        if checkpoint_step is not None:
            scalars["checkpoint_step"] = checkpoint_step
        result = {
            "schema_version": 1,
            "model": "student" if student is not None else "teacher",
            "binding_sha256": binding,
            "protocol": protocol,
            "scores": rows,
            **scalars,
        }
        atomic_json(output / "metrics.json", result)
        with atomic_path(output / "marginal_metrics.csv") as temporary:
            with temporary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        figures = save_figures(dataset, teacher_path, student_path, output, settings)
        atomic_json(
            output / "media.json",
            {key: str(path.resolve()) for key, path in figures.items()},
        )
        # Local computation is complete before optional remote publication.
        atomic_json(
            output / "status.json", {"status": "complete", "binding_sha256": binding}
        )
        if logger is not None:
            logger.log(
                {
                    f"eval/{key}": value
                    for key, value in scalars.items()
                    if value is not None
                }
            )
            if logger.run is not None:
                # Already-running teachers may hold the earlier logger class.
                guard = logger.remote("evaluation publication") if hasattr(logger, "remote") else nullcontext()
                with guard:
                    publish_evaluation(logger.run, rows, scalars, figures, output)
        return result
    except BaseException as error:
        atomic_json(output / "status.json", {"status": "failed", "error": str(error)})
        raise


def publish_evaluation(run, rows, scalars, figures, output):
    """Publish already-saved evaluation outputs; no model/data execution."""
    output = Path(output)
    if run is None:
        return
    import wandb

    run.log({
        "eval/marginal_metrics": wandb.Table(columns=list(rows[0]), data=[list(row.values()) for row in rows]),
        **{f"eval/{key}": wandb.Image(str(path)) for key, path in figures.items()},
        **{f"eval/marginal_{row['index']}/{key}": value for row in rows
           for key, value in row.items() if key.endswith("emd")},
    })
    run.summary.update({f"eval/{key}": value for key, value in scalars.items() if value is not None})
    artifact = wandb.Artifact(f"{run.id}-evaluation", type="evaluation")
    artifacts = [output / "metrics.json", output / "marginal_metrics.csv",
                 output / "media.json", output / "teacher_predictions.pt", *figures.values()]
    if (output / "predictions.pt").is_file():
        artifacts.append(output / "predictions.pt")
    for path in artifacts:
        artifact.add_file(str(path))
    run.log_artifact(artifact)
