"""Teacher comparison on saved MFM splits; test populations never rank runs."""

import json
from pathlib import Path

import numpy as np
import torch

from flow2pc.evaluation.ode import fixed_subset, raw_coordinates, rollout
from flow2pc.evaluation.runner import evaluate as evaluate_ode, population
from flow2pc.utils.checkpoint import atomic_json


def evaluate(teacher, dataset, settings, output, *, logger=None, **kwargs):
    output = Path(output)
    validation_settings = {**settings, "source_split": "val", "reference_split": "val",
                           "time_indices": list(dataset.observed),
                           "num_samples": 512, "metric_samples": 512}
    validation = evaluate_ode(teacher, dataset, validation_settings, output / "validation")
    result = evaluate_ode(teacher, dataset, settings, output, logger=logger, **kwargs)
    selection = {
        "schema_version": 1, "metric": "validation_teacher_mean_emd",
        "score": validation["teacher_mean_emd"],
        "source_split": "val", "reference_split": "val",
        "time_indices": list(dataset.observed),
        "excluded_test_times": list(dataset.heldout_indices),
        "prepared_identity": dataset.source["prepared_identity"],
        "coordinate_space": settings["metric_space"],
        "test_heldout_emd": result.get("teacher_heldout_emd"),
        "formula": "uniform-mass exact W1; Euclidean costs; all dimensions",
        "note": "Ranks final trained teachers across configurations; no temporal test score selects a run",
    }
    atomic_json(output / "selection.json", selection)
    media_path = output / "media.json"
    media = json.loads(media_path.read_text())
    media.update(geometry_figure(teacher, dataset, settings, output))
    atomic_json(media_path, media)
    if logger is not None:
        logger.log({"validation/teacher_emd": selection["score"],
                    "protocol/interior_observed_populations": max(0, len(dataset.observed) - 2)})
        if logger.run is not None:
            import wandb

            with logger.remote("MFM geometry comparison"):
                logger.run.log({"eval/teacher_geometry": wandb.Image(media["teacher_geometry"])})
                artifact = wandb.Artifact(f"{logger.run.id}-mfm-benchmark", type="evaluation")
                for relative in ("selection.json", "media.json", "teacher_geometry.png",
                                 "geometry_trajectories.npz", "validation/metrics.json"):
                    artifact.add_file(str(output / relative))
                logger.run.log_artifact(artifact)
    return result


@torch.no_grad()
def geometry_figure(teacher, dataset, settings, output):
    """Dense ODE curves and observed populations; LiDAR includes terrain geometry."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = fixed_subset(population(dataset, 0, settings["source_split"]), 256, settings["seed"])
    grid = torch.linspace(0, 1, 101)
    paths, _ = rollout(teacher, source, grid, settings)
    raw_path = raw_coordinates(paths, dataset.preprocessing).numpy()
    raw = [raw_coordinates(cloud, dataset.preprocessing).numpy() for cloud in dataset.clouds]
    dims = dataset.dim
    fig = plt.figure(figsize=(7 * (len(raw) + 1), 6))
    projection = "3d" if dims == 3 else None
    axes = [fig.add_subplot(1, len(raw) + 1, i + 1, projection=projection) for i in range(len(raw) + 1)]
    terrain = None
    bounds_parts = [*raw, raw_path.reshape(-1, dims)]
    if dataset.name == "lidar":
        terrain = raw_coordinates(torch.cat(dataset.metric_clouds), dataset.preprocessing).numpy()
        terrain = terrain[::max(1, len(terrain) // 10000)]
        bounds_parts.append(terrain)
    bounds_values = np.concatenate(bounds_parts)
    lo, hi = bounds_values.min(0), bounds_values.max(0)
    padding = np.maximum((hi - lo) * .05, .05)
    for index, ax in enumerate(axes[:-1]):
        cloud = raw[index][::max(1, len(raw[index]) // 1500)]
        prediction = raw_path[round(dataset.metadata.model_times[index] * 100)]
        ax.scatter(*cloud[:, :dims].T, s=4, alpha=.3, label="Observed")
        ax.scatter(*prediction[:, :dims].T, s=5, alpha=.5, label="Teacher ODE")
        suffix = "TEST: never trained" if index in dataset.heldout_indices else "observed time"
        ax.set_title(f"t={dataset.metadata.model_times[index]:g} | {suffix}")
        ax.legend(fontsize=8)
    ax = axes[-1]
    if terrain is not None:
        ax.scatter(*terrain.T, c=terrain[:, 2], cmap="terrain", s=1, alpha=.15)
    for trajectory in raw_path.transpose(1, 0, 2)[:64]:
        ax.plot(*trajectory.T, color="crimson", alpha=.4, linewidth=.7)
    ax.scatter(*raw_path[0].T, s=6, color="green", label="Start")
    ax.scatter(*raw_path[-1].T, s=6, color="purple", label="End")
    ax.set_title("101-time ODE trajectories" + (" / terrain" if dataset.name == "lidar" else ""))
    ax.legend(fontsize=8)
    for ax in axes:
        ax.set_xlim(lo[0] - padding[0], hi[0] + padding[0])
        ax.set_ylim(lo[1] - padding[1], hi[1] + padding[1])
        ax.set_xlabel("x (raw)")
        ax.set_ylabel("y (raw)")
        if dims == 3:
            ax.set_zlim(lo[2] - padding[2], hi[2] + padding[2])
            ax.set_zlabel("z (raw)")
            ax.set_box_aspect(hi - lo + 2 * padding)
        else:
            ax.set_aspect("equal", adjustable="box")
    fig.suptitle(f"{dataset.name} | {teacher.manifest['method']} | saved MFM splits")
    fig.tight_layout()
    destination = Path(output) / "teacher_geometry.png"
    fig.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(fig)
    np.savez_compressed(Path(output) / "geometry_trajectories.npz", times=grid.numpy(), paths_raw=raw_path)
    return {"teacher_geometry": str(destination.resolve())}
