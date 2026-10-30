"""Recovered marginal, microscopy/H&E overlay and knot trajectory figures."""

from pathlib import Path

import numpy as np
import torch

from flow2pc.data.base import heldout_indices
from flow2pc.evaluation.ode import metric_coordinates, raw_coordinates

TRACKING_FRAMES = (0, 16, 32, 48, 65, 81, 97, 114)


def image_directory(dataset, settings):
    if settings.get("image_dir"):
        return Path(settings["image_dir"])
    relative = {
        "st": "spatial_transcriptomics/aligned/ref_U5_warped_images",
        "cell_tracking": "cell_tracking/PhC-C2DH-U373/01",
    }
    return dataset.root / relative[dataset.metadata.name]


def validate_resources(dataset, settings):
    if not settings.get("enabled", False):
        return
    name = dataset.metadata.name
    metric_coordinates(
        torch.zeros(1, dataset.dim), dataset.preprocessing, settings["metric_space"]
    )
    if name in ("st", "cell_tracking"):
        directory = image_directory(dataset, settings)
        labels = (
            (
                [dataset.labels[dataset.heldout]]
                if dataset.heldout is not None
                else dataset.labels
            )
            if name == "st"
            else TRACKING_FRAMES
        )
        for label in labels:
            path = directory / (
                f"{label}_aligned_to_U5.tif" if name == "st" else f"t{label:03d}.tif"
            )
            if not path.is_file():
                raise FileNotFoundError(
                    f"Evaluation background missing: {path}; set evaluation.image_dir to the original clean images"
                )


def save_figures(dataset, teacher_path, student_path, output, settings):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    models = [("teacher", teacher_path)]
    if student_path is not None:
        models.append(("student", student_path))
    files = {}

    def save(fig, key, filename):
        path = output / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        files[key] = path

    name = dataset.metadata.name
    if name == "cell_tracking":
        from PIL import Image

        directory = image_directory(dataset, settings)
        backgrounds = []
        for index in TRACKING_FRAMES:
            with Image.open(directory / f"t{index:03d}.tif") as handle:
                backgrounds.append(np.asarray(handle).copy())

        def panel(axis, index, path, label, image):
            real = raw_coordinates(dataset.clouds[index], dataset.preprocessing).numpy()
            prediction = raw_coordinates(path[index], dataset.preprocessing).numpy()
            axis.imshow(image, cmap="gray", origin="upper")
            axis.scatter(real[:, 0], real[:, 1], s=1, alpha=0.12, color="cyan")
            axis.scatter(
                prediction[:, 0], prediction[:, 1], s=2, alpha=0.35, color="red"
            )
            axis.set(
                xlim=(0, image.shape[1]),
                ylim=(image.shape[0], 0),
                title=f"{label} | frame {index + 1}",
            )
            axis.axis("off")

        for label, path in models:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            for axis, index, background in zip(
                axes.flat, TRACKING_FRAMES, backgrounds, strict=True
            ):
                panel(axis, index, path, label, background)
            fig.suptitle(
                "Cell tracking | cyan: observed mask, red: velocity-driven ODE"
            )
            save(fig, f"{label}_overlay", f"{label}_overlay.png")
        if student_path is not None:
            fig, axes = plt.subplots(2, 8, figsize=(26, 7))
            for row, (label, path) in enumerate(models):
                for col, (index, background) in enumerate(
                    zip(TRACKING_FRAMES, backgrounds, strict=True)
                ):
                    panel(axes[row, col], index, path, label, background)
            fig.suptitle("Teacher / student | identical observed source particles")
            save(fig, "teacher_student_overlay", "comparison_overlay.png")
    elif name == "st":
        from PIL import Image

        directory = image_directory(dataset, settings)
        indices = (
            [dataset.heldout]
            if dataset.heldout is not None
            else list(range(len(dataset.clouds)))
        )
        for index in indices:
            with Image.open(
                directory / f"{dataset.labels[index]}_aligned_to_U5.tif"
            ) as handle:
                background = np.asarray(handle.convert("RGB")).copy()
            real = raw_coordinates(dataset.clouds[index], dataset.preprocessing).numpy()

            def panel(axis, coordinates, label):
                axis.imshow(background, origin="upper")
                # Preserve the original CSV convention: column 1 = x, 0 = y.
                axis.scatter(
                    coordinates[:, 1],
                    coordinates[:, 0],
                    s=10,
                    alpha=0.7,
                    facecolors="none",
                    edgecolors="red",
                    linewidths=0.8,
                )
                axis.set(
                    xlim=(0, background.shape[1]),
                    ylim=(background.shape[0], 0),
                    title=label,
                )
                axis.axis("off")

            suffix = "" if len(indices) == 1 else f"_{dataset.labels[index]}"
            for label, path in models:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
                panel(axes[0], real, f"Observed {dataset.labels[index]}")
                panel(
                    axes[1],
                    raw_coordinates(path[index], dataset.preprocessing).numpy(),
                    label,
                )
                save(fig, f"{label}_overlay{suffix}", f"{label}_overlay{suffix}.png")
            if student_path is not None:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
                panel(axes[0], real, "Ground truth")
                for axis, (label, path) in zip(axes[1:], models, strict=True):
                    panel(
                        axis,
                        raw_coordinates(path[index], dataset.preprocessing).numpy(),
                        label + " ODE",
                    )
                fig.suptitle(
                    f"ST {dataset.labels[index]} | teacher / student from identical U2 cells"
                )
                save(fig, f"st_he_overlay{suffix}", f"st_he_overlay{suffix}.png")
    elif name == "knot":
        real = [
            raw_coordinates(x, dataset.preprocessing).numpy() for x in dataset.clouds
        ]

        def panel(axis, path, label):
            for index, cloud in enumerate(real):
                axis.scatter(
                    cloud[:, 0],
                    cloud[:, 1],
                    color=plt.cm.viridis(index / (len(real) - 1)),
                    alpha=0.15,
                    s=6,
                )
            values = raw_coordinates(path, dataset.preprocessing).numpy()
            for trajectory in values.transpose(1, 0, 2)[:100]:
                axis.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    color="crimson",
                    alpha=0.5,
                    linewidth=1,
                )
            axis.scatter(
                values[0, :100, 0],
                values[0, :100, 1],
                color="forestgreen",
                s=20,
                label="Start",
            )
            axis.scatter(
                values[-1, :100, 0],
                values[-1, :100, 1],
                color="purple",
                s=20,
                label="End",
            )
            axis.set(
                xlim=(-5, 5),
                ylim=(-3, 3),
                aspect="equal",
                title=f"{label} ODE path",
                xlabel="X",
                ylabel="Y",
            )
            axis.legend()

        for label, path in models:
            fig, axis = plt.subplots(figsize=(8, 8))
            panel(axis, path, label)
            save(fig, f"{label}_trajectory", f"{label}_trajectories.png")
        if student_path is not None:
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            for axis, (label, path) in zip(axes, models, strict=True):
                panel(axis, path, label)
            save(fig, "comparison_trajectories", "comparison_trajectories.png")
    elif name in ("gom", "beijing"):
        count, columns = len(dataset.clouds), 4
        raw = [raw_coordinates(x, dataset.preprocessing).numpy() for x in dataset.clouds]
        for label, path in models:
            fig, axes = plt.subplots(
                (count + columns - 1) // columns, columns,
                figsize=(16, 3.5 * ((count + columns - 1) // columns)), squeeze=False,
            )
            for index, axis in enumerate(axes.flat):
                if index >= count:
                    axis.axis("off")
                    continue
                prediction = raw_coordinates(path[index], dataset.preprocessing).numpy()
                if name == "beijing":
                    edges = np.histogram_bin_edges(np.concatenate((raw[index][:, 0], prediction[:, 0])), bins=40)
                    axis.hist(raw[index][:, 0], bins=edges, density=True, histtype="step", label="Observed")
                    axis.hist(prediction[:, 0], bins=edges, density=True, histtype="step", label=label)
                    axis.set(xlabel="PM2.5 (µg/m³)", ylabel="Density")
                else:
                    axis.scatter(raw[index][:, 0], raw[index][:, 1], s=8, alpha=0.4, label="Observed")
                    axis.scatter(prediction[:, 0], prediction[:, 1], s=8, alpha=0.4, label=label)
                    axis.set(xlabel="Source coordinate 1", ylabel="Source coordinate 2", aspect="equal")
                suffix = "held out" if index in heldout_indices(dataset) else "observed time"
                axis.set_title(f"{dataset.labels[index]} | {suffix}")
            axes[0, 0].legend()
            fig.suptitle(f"{name.upper()} | {label} velocity-driven ODE from earliest observed population")
            fig.tight_layout()
            save(fig, f"{label}_marginals", f"{label}_ode_marginals.png")
            if name == "beijing":
                fig, axis = plt.subplots(figsize=(13, 5))
                for quantile, color in zip((0.1, 0.5, 0.9), ("tab:blue", "tab:orange", "tab:green"), strict=True):
                    observed = [np.quantile(x[:, 0], quantile) for x in raw]
                    predicted = [np.quantile(raw_coordinates(x, dataset.preprocessing).numpy()[:, 0], quantile) for x in path]
                    axis.plot(range(count), observed, "o--", color=color, label=f"Observed q={quantile:g}")
                    axis.plot(range(count), predicted, "-", color=color, label=f"{label} q={quantile:g}")
                for index in heldout_indices(dataset):
                    axis.axvspan(index - 0.15, index + 0.15, color="gray", alpha=0.15)
                axis.set_xticks(range(count), dataset.labels, rotation=45, ha="right")
                axis.set(ylabel="PM2.5 (µg/m³)", title="Population quantiles; shaded times held out of teacher training")
                axis.legend(ncol=3)
                fig.tight_layout()
                save(fig, f"{label}_quantiles", f"{label}_quantiles.png")
    else:
        count = (
            len(dataset.clouds)
            if hasattr(dataset, "clouds")
            else len(dataset.metadata.model_times)
        )

        def panel(axis, index, path, label):
            cloud = (
                dataset.clouds[index]
                if hasattr(dataset, "clouds")
                else torch.cat(
                    [
                        dataset.get_marginal(index, split).materialize()
                        for split in ("train", "val", "test")
                    ]
                )
            )
            real = metric_coordinates(
                cloud, dataset.preprocessing, settings["metric_space"]
            )
            predicted = metric_coordinates(
                path[index], dataset.preprocessing, settings["metric_space"]
            )
            real = real[:: max(1, len(real) // 1500)]
            axis.scatter(real[:, 0], real[:, 1], s=2, alpha=0.3, label="Observed")
            axis.scatter(predicted[:, 0], predicted[:, 1], s=2, alpha=0.3, label=label)
            time_label = (
                dataset.labels[index]
                if hasattr(dataset, "labels")
                else f"Day {dataset.metadata.physical_times[index]:g}"
            )
            axis.set_title(
                time_label + (" (held out)" if index in heldout_indices(dataset) else "")
            )
            axis.set_xlabel("Coordinate 1 (metric space)")

        for label, path in models:
            fig, axes = plt.subplots(
                1,
                count,
                figsize=(4 * count, 4),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            for index, axis in enumerate(axes[0]):
                panel(axis, index, path, label)
            axes[0, 0].set_ylabel("Coordinate 2 (metric space)")
            axes[0, -1].legend(markerscale=3)
            fig.tight_layout()
            save(fig, f"{label}_marginals", f"{label}_ode_marginals.png")
        if student_path is not None:
            fig, axes = plt.subplots(
                2,
                count,
                figsize=(4 * count, 8),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            for row, (label, path) in enumerate(models):
                for index, axis in enumerate(axes[row]):
                    panel(axis, index, path, label)
                axes[row, 0].set_ylabel(label)
            fig.tight_layout()
            save(fig, "comparison_marginals", "comparison_marginals.png")
    return files
