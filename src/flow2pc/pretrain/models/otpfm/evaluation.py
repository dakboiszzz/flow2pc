"""Native finite-interval sampling/W2 results and final images, separate from ODE EMD."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from flow2pc.data.preprocessing import scaler_from_statistics
from flow2pc.utils.checkpoint import atomic_json
from flow2pc.utils.tracking import LocalLogger

from .profiles import get_profile


def _plot_marginals(dataset, trajectories, times, output, label, epoch):
    import matplotlib.pyplot as plt

    scaler = scaler_from_statistics(dataset.preprocessing)
    shape = trajectories.shape
    predictions = scaler.inverse_transform(trajectories.reshape(-1, shape[-1])).reshape(shape)
    count = len(dataset.clouds)
    columns = 4
    fig, axes = plt.subplots((count + columns - 1) // columns, columns,
                             figsize=(16, 3.5 * ((count + columns - 1) // columns)), squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= count:
            axis.axis("off")
            continue
        target = scaler.inverse_transform(dataset.clouds[index].numpy())
        time = dataset.metadata.model_times[index]
        step = int(np.argmin(np.abs(times - time)))
        predicted = predictions[step]
        if dataset.dim == 1:
            bins = np.histogram_bin_edges(np.r_[target[:, 0], predicted[:, 0]], bins=40)
            axis.hist(target[:, 0], bins=bins, density=True, histtype="step", label="Observed")
            axis.hist(predicted[:, 0], bins=bins, density=True, histtype="step", label="OTP-FM EMA")
            axis.set(xlabel="Value (raw units)", ylabel="Density")
        else:
            axis.scatter(target[:, 0], target[:, 1], s=8, alpha=0.4, label="Observed")
            axis.scatter(predicted[:, 0], predicted[:, 1], s=8, alpha=0.4, label="OTP-FM EMA")
            axis.set(xlabel="Coordinate 1 (raw units)", ylabel="Coordinate 2 (raw units)", aspect="equal")
        holdout = "held out" if index in dataset.heldout_indices else "observed time"
        axis.set_title(f"{dataset.labels[index]} | {holdout}\nnearest stored t={times[step]:.5f}")
    axes[0, 0].legend()
    fig.suptitle(f"{dataset.metadata.name} | {label}, epoch {epoch} | native finite-interval EMA sampling")
    fig.tight_layout()
    path = output / f"{label}_marginals.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_results(trainer, dataset, native, run_dir, best, logger=None):
    """Publish cached native outcomes; do not introduce an extra model rollout."""
    from ._source.reproduction import log_trajectory_previews
    from .tracking import RunSink

    run_dir = Path(run_dir)
    output = run_dir / "eval"
    output.mkdir(parents=True, exist_ok=True)
    scores = []
    for index in range(1, len(dataset.clouds)):
        key = f"eval/w2_t{index}"
        scores.append({"index": index, "model_time": dataset.metadata.model_times[index],
                       "physical_time": dataset.metadata.physical_times[index],
                       "held_out": index in dataset.heldout_indices, "teacher_w2": best["metrics"][key]})
    result = {
        "schema_version": 1, "method": "otpfm", "model": "teacher", "selected_epoch": best["epoch"],
        "teacher_heldout_w2": best["score"], "teacher_train_w2": best["metrics"]["eval/w2_rest"],
        "selection_rule": native.get("checkpoint_selection", "mean_w2"),
        "selection_score": best.get("selection_score", best["score"]), "scores": scores,
        "protocol": {"name": "native_otpfm", "sampler": "EMA finite-interval consistency sampler",
                     "instantaneous_teacher": "EMA v_func_ema(x,t,t); not the native finite-step metric",
                     "eval_n_steps": native["eval_n_steps"], "eval_num_samples": native["eval_num_samples"],
                     "holdout_indices": dataset.heldout_indices, "potential_times": native["tks"],
                     "formula": "sqrt(uniform-mass OT squared-Euclidean cost); native unequal-size truncation",
                     "time_rule": get_profile(dataset.metadata.name).metric_time_rule,
                     "units": "raw coordinates", "figure_time_rule": "nearest stored time at each observed time knot"},
    }
    atomic_json(output / "metrics.json", result)
    media = {}
    for label, epoch in (("best", best["epoch"]), ("final", native["epochs"])):
        path = run_dir / "trajectories" / f"trajectories_epoch{epoch:04d}.npz"
        with np.load(path, allow_pickle=False) as payload:
            trajectories, times = payload["trajectories"], payload["t_eval"]
        media[f"{label}_marginals"] = str(_plot_marginals(dataset, trajectories, times, output, label, epoch).resolve())
    atomic_json(output / "media.json", media)
    atomic_json(output / "status.json", {"status": "complete", "protocol": "native_otpfm"})
    if logger is not None:
        logger.log({"native/heldout_w2": result["teacher_heldout_w2"],
                    "native/train_w2": result["teacher_train_w2"], "native/selected_epoch": best["epoch"]})
        log_trajectory_previews(RunSink(logger), trainer, run_dir, native, best["epoch"])
        if logger.run is not None:
            import wandb

            with logger.remote("OTP-FM final figures"):
                logger.run.log({f"plots/{name}": wandb.Image(path) for name, path in media.items()})
                artifact = wandb.Artifact(f"{logger.run.id}-native-evaluation", type="evaluation")
                for name in ("metrics.json", "media.json", *[Path(p).name for p in media.values()]):
                    artifact.add_file(str(output / name))
                logger.run.log_artifact(artifact)
    return result


def evaluate_checkpoint(config):
    """Explicit CLI evaluation using original native sampling/metric functions.

    This function is never invoked by the metadata-only launch preview.
    """
    import importlib

    from flow2pc.data.restore import restore_dataset
    from .adapter import load

    run_dir = Path(config["output"]["dir"])
    teacher = load(run_dir / "checkpoints/best.pt", config["training"]["device"])
    dataset = restore_dataset(teacher.manifest["dataset"], teacher.preprocessing)
    native = teacher.manifest["model_config"]["native_config"]
    epoch = teacher.manifest["selected_checkpoint"]["epoch"]
    source = dataset.clouds[0][:native["eval_num_samples"]].to(config["training"]["device"])
    trajectories, times = teacher.sample(source, native["eval_n_steps"])
    profile = get_profile(dataset.metadata.name)
    module = importlib.import_module(profile.trainer_module, __package__ + "._source")
    cls = getattr(module, profile.trainer_class)
    scope = SimpleNamespace(marginals={i: x for i, x in enumerate(dataset.clouds)},
                            train_times=dataset.observed, holdout_times=dataset.heldout_indices,
                            scaler=scaler_from_statistics(dataset.preprocessing), epoch_trajectories=[trajectories.numpy()],
                            trajectory_t_eval=times.numpy(), losses={**{f"w2_t{i}": [] for i in range(1, len(dataset.clouds))},
                                                                  "w2_rest": [], "metric_epochs": []},
                            logger=logging.getLogger(__name__))
    cls._compute_metrics(scope, epoch)
    values = {f"eval/{key}": float(value[-1]) for key, value in scope.losses.items() if key != "metric_epochs"}
    best = {"epoch": epoch, "score": float(np.mean([values[f"eval/w2_t{i}"] for i in dataset.heldout_indices])), "metrics": values}
    if native.get("checkpoint_selection") == "mean_squared_w2":
        best["selection_score"] = float(np.mean([values[f"eval/w2_t{i}"] ** 2 for i in dataset.heldout_indices]))
    # Use a separate evaluation output: never overwrite the native training
    # trajectories, best/final images or source metrics during an explicit check.
    output = run_dir / "native_checkpoint_eval"
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "selected_trajectories.npz", trajectories=trajectories.numpy(), t_eval=times.numpy())
    image = _plot_marginals(dataset, trajectories.numpy(), times.numpy(), output, "selected", epoch)
    result = {"selected_epoch": epoch, "teacher_heldout_w2": best["score"], "metrics": values, "protocol": "native_otpfm"}
    atomic_json(output / "metrics.json", result)
    atomic_json(output / "media.json", {"selected_marginals": str(image.resolve())})
    logger = LocalLogger(run_dir, config, append=True)
    try:
        logger.log({"native_checkpoint_eval/heldout_w2": best["score"]})
        if logger.run is not None:
            import wandb

            with logger.remote("OTP-FM explicit evaluation image"):
                logger.run.log({"native_checkpoint_eval/marginals": wandb.Image(str(image))})
    finally:
        logger.close()
    return {"evaluation_dir": str(output), "media_manifest": str(output / "media.json"), **result}
