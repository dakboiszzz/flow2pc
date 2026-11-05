"""Exact population distances for direct-PC and velocity-driven ODE predictions."""

import json
import random
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from flow2pc.evaluation.ode import emd, fixed_subset, raw_coordinates
from flow2pc.utils.checkpoint import atomic_json
from flow2pc.utils.experiment_protocols import benchmark, dataset_tag


@contextmanager
def evaluation_seed(seed, device):
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    numpy_state, python_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        for index in devices:
            torch.cuda.default_generators[index].manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


@torch.no_grad()
def integrate(velocity, initial, times, substeps=8):
    states, values = [initial.clone()], initial.clone()
    for left, right in pairwise(times):
        h = (right - left) / substeps
        for index in range(substeps):
            t = left + index * h
            k1 = velocity(values, t)
            k2 = velocity(values + h * k1 / 2, t + h / 2)
            k3 = velocity(values + h * k2 / 2, t + h / 2)
            k4 = velocity(values + h * k3, t + h)
            values = values + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if not torch.isfinite(values).all():
                raise FloatingPointError("Nonfinite ODE trajectory; reduce the step size")
        states.append(values.clone())
    return torch.stack(states)


def _rollout_batch(velocity, initial, times, settings):
    evaluations = 0

    def checked_velocity(x, t):
        nonlocal evaluations
        evaluations += 1
        if evaluations > int(settings.get("max_nfe", 20000)):
            raise RuntimeError("ODE function-evaluation budget exceeded")
        result = velocity(x, t)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite ODE velocity")
        return result

    if settings.get("solver", "rk4") == "rk4":
        return integrate(checked_velocity, initial, times, int(settings.get("ode_substeps", 16)))
    from torchdiffeq import odeint
    return odeint(lambda t, x: checked_velocity(x, t), initial, times,
                  method=settings["solver"], rtol=settings.get("tolerance", 1e-5),
                  atol=settings.get("tolerance", 1e-5))


def _rollout(velocity, initial, times, settings):
    # Release each batch's GPU path before starting the next one. In particular,
    # evaluating a large population must not require one equally large PC batch.
    count = int(settings.get("batch_size", 128))
    paths = [_rollout_batch(velocity, batch, times, settings).detach().cpu()
             for batch in initial.split(count)]
    return torch.cat(paths, dim=1)


def _direct_samples(student, count, time, settings):
    batch_size = int(settings.get("batch_size", 128))
    return torch.cat([
        student.sample(min(batch_size, count - start), time).flatten(1).detach().cpu()
        for start in range(0, count, batch_size)
    ])


def _population(data, index, split):
    heldout = index in data.heldout_indices
    if split == "all":
        return getattr(data, "marginals", data.validation_marginals)[index]
    if split == "train":
        return data.train_marginals[index]
    if split == "test":
        if not heldout:
            return data.validation_marginals[index][:0]
        return getattr(data, "test_marginals", data.validation_marginals)[index]
    if split != "validation":
        raise ValueError(f"Unknown population split {split!r}")
    return (getattr(data, "test_marginals", data.validation_marginals)[index]
            if heldout else data.validation_marginals[index])


def _raw(values, data):
    if hasattr(data, "preprocessing"):
        return raw_coordinates(values.detach().cpu().flatten(1), data.preprocessing).double()
    norm = getattr(data, "normalization", {})
    values = values.detach().cpu().double().flatten(1)
    if "mean" in norm:
        return values * torch.tensor(norm["scale"]) + torch.tensor(norm["mean"])
    return values


def from_ali_dataset(dataset):
    """Read-only view of the existing ALI data contract for shared diagnostics."""
    from flow2pc.data.base import heldout_indices
    from flow2pc.evaluation.runner import population

    count = len(dataset.metadata.model_times)
    return SimpleNamespace(
        metadata=dataset.metadata, dim=dataset.dim, sample_shape=(dataset.dim,),
        source_info={"profile": dataset.metadata.name}, normalization={},
        preprocessing=dataset.preprocessing,
        times=torch.tensor(dataset.metadata.model_times),
        observed_indices=list(dataset.observed), heldout_indices=heldout_indices(dataset),
        marginals=[population(dataset, i, "all") for i in range(count)],
        train_marginals=[population(dataset, i, "train") for i in range(count)],
        validation_marginals=[population(dataset, i, "val") for i in range(count)],
        test_marginals=[population(dataset, i, "test") for i in range(count)],
    )


def _metric_transform(data, settings):
    tag = dataset_tag(data)
    space = settings.get("metric_space", "dataset")
    standardized = space == "teacher" or (space == "dataset" and
        (tag.startswith(("cite", "multi", "eb")) and data.dim == 5))
    if space not in ("dataset", "raw", "teacher"):
        raise ValueError("metric_space must be dataset, raw or teacher")
    if space == "teacher":
        return lambda x: x.detach().cpu().double().flatten(1), "teacher_input_coordinates"
    if standardized:
        # Paper 5D metrics use the benchmark's whitening, independently of
        # method-specific input normalization. This does not refit the teacher.
        values = torch.cat([_raw(x, data) for x in data.marginals])
        mean, scale = values.mean(0), values.std(0, unbiased=False).clamp_min(1e-12)
        return lambda x: (_raw(x, data) - mean) / scale, "benchmark_standardized_5D"
    return lambda x: _raw(x, data), "raw_feature_coordinates"


def _distances(real, predicted, transport):
    result = {"emd": emd(real, predicted)}
    if transport:
        count = min(len(real), len(predicted))
        # Match the released transport benchmark's equal-size W2 helper.
        result["w2"] = emd(real[:count], predicted[:count], order=2)
    return result


def _numerical_failure(error):
    return isinstance(error, FloatingPointError) or (
        isinstance(error, (RuntimeError, AssertionError)) and any(word in str(error).lower() for word in
        ("nonfinite", "non-finite", "probability tensor", "nan", "out of memory",
         "underflow in dt", "max_num_steps", "function-evaluation budget")))


@torch.no_grad()
def evaluate(student, teacher, data, settings, device, output_dir=None):
    """Both prediction modes are diagnostics; neither selects the checkpoint.

    Direct samples get separate RNG scopes so enabling them cannot change ODE
    particles, references, or subsequent training draws. Numerical failures are
    reported per prediction mode, while programming/contract errors propagate.
    """
    device = torch.device(device)
    samples, seed = int(settings.get("samples", 512)), int(settings.get("seed", 2026))
    if samples < 1 or int(settings.get("batch_size", 128)) < 1:
        raise ValueError("Evaluation samples and batch_size must be positive")
    if int(settings.get("ode_substeps", 16)) < 1:
        raise ValueError("Evaluation ode_substeps must be positive")
    protocol = benchmark(data)
    transport = protocol["dataset"] in ("gom", "beijing")
    transform, units = _metric_transform(data, settings)
    indices = list(settings.get("time_indices", range(len(data.times))))
    if (not indices or any(type(i) is not int for i in indices)
            or indices != sorted(set(indices))
            or any(i < 0 or i >= len(data.times) for i in indices)):
        raise ValueError("Invalid evaluation time_indices")
    was_training = student.training if student is not None else None
    if student is not None:
        student.eval()
    arrays, report = {}, {"schema_version": 2, "space": units, "by_time": {},
        "times": data.times.tolist(), "heldout_indices": data.heldout_indices,
        "protocol": {**protocol, "settings": settings,
                     "emd_formula": "uniform-mass W1, Euclidean costs, full dimension",
                     "w2_formula": "sqrt(uniform-mass OT with squared Euclidean costs)",
                     "transport_w2_population_policy": "truncate both sets to equal length, as in OTP-FM helper",
                     "student_selection": "validation_velocity_MSE_only"}}
    try:
        with evaluation_seed(seed, device):
            source = _population(data, 0, settings.get("source_split", "all"))
            initial = fixed_subset(source, samples, seed).flatten(1).to(device)
            if not len(initial):
                raise ValueError("Evaluation source is empty")
            times = data.times.to(initial)
            arrays["times"] = times.cpu().numpy()
            paths = {}
            modes = [("teacher_ode", teacher.velocity)]
            if student is not None:
                modes.append(("student_ode", student.velocity))
            for name, velocity in modes:
                try:
                    path = _rollout(velocity, initial, times, settings)
                    if not torch.isfinite(path).all():
                        raise FloatingPointError("Nonfinite ODE samples")
                    paths[name] = path
                    arrays[name] = np.stack([_raw(x, data).numpy() for x in path])
                except (FloatingPointError, RuntimeError, AssertionError) as error:
                    if not _numerical_failure(error):
                        raise
                    report[name + "_error"] = str(error)
            direct_records = []
            direct_final = None
            for index in indices:
                pool = _population(data, index, settings.get("reference_split", "all"))
                if not len(pool):
                    raise ValueError(f"Evaluation reference {index} is empty")
                reference = fixed_subset(pool, samples, seed + index + 1)
                real = transform(reference)
                arrays[f"reference_{index}"] = _raw(reference, data).numpy()
                row = {"heldout": index in data.heldout_indices, "reference_count": len(real)}
                predictions = {}
                for name, velocity in modes:
                    # ALI-CFM's single-cell/ST benchmark pushes the preceding
                    # observed population to each left-out time, rather than
                    # treating a full earliest-time rollout as the same score.
                    if (index in data.heldout_indices and
                            protocol["rollout_origin"] == "preceding_observed"):
                        left = max(i for i in data.observed_indices if i < index)
                        local_source = fixed_subset(_population(data, left, settings.get("source_split", "all")), samples, seed + left)
                        try:
                            predicted = _rollout(velocity, local_source.flatten(1).to(device), times[[left, index]], settings)[-1]
                            if not torch.isfinite(predicted).all():
                                raise FloatingPointError("Nonfinite local ODE samples")
                        except (FloatingPointError, RuntimeError, AssertionError) as error:
                            if not _numerical_failure(error):
                                raise
                            row[name + "_error"] = str(error)
                            continue
                        row["source_index"] = left
                        arrays[f"benchmark_{name}_{index}"] = _raw(predicted, data).numpy()
                    else:
                        if name not in paths:
                            continue
                        predicted = paths[name][index]
                        row["source_index"] = 0
                    predictions[name] = transform(predicted)
                    row.update({name + "_" + key: value for key, value in
                                _distances(real, predictions[name], transport).items()})
                if "teacher_ode" in predictions and "student_ode" in predictions:
                    row["paired_ode_mse"] = float((predictions["student_ode"] - predictions["teacher_ode"]).square().mean())
                if student is not None:
                    try:
                        with evaluation_seed(seed + 100000 + index, device):
                            direct = _direct_samples(student, len(initial), times[index], settings)
                        if not torch.isfinite(direct).all():
                            raise FloatingPointError("Nonfinite direct PC samples")
                        row.update({"direct_pc_" + key: value for key, value in
                                    _distances(real, transform(direct), transport).items()})
                        if "teacher_ode" in predictions:
                            row.update({"direct_pc_teacher_" + key: value for key, value in
                                        _distances(predictions["teacher_ode"], transform(direct), transport).items()})
                        direct_records.append(_raw(direct, data).numpy())
                        arrays[f"direct_pc_{index}"] = _raw(direct, data).numpy()
                        if index == len(data.times) - 1:
                            direct_final = direct
                    except (FloatingPointError, RuntimeError) as error:
                        if not _numerical_failure(error):
                            raise
                        row["direct_pc_error"] = str(error)
                report["by_time"][str(index)] = row
            for mode in ("teacher_ode", "student_ode", "direct_pc"):
                for metric in ("emd", "w2") if transport else ("emd",):
                    key = mode + "_" + metric
                    for label, chosen in (("mean", [i for i in indices if i != 0]),
                                          ("heldout", [i for i in indices if i in data.heldout_indices])):
                        if chosen and all(key in report["by_time"][str(i)] for i in chosen):
                            report[key + "_" + label] = float(np.mean([report["by_time"][str(i)][key] for i in chosen]))
            if len(direct_records) == len(data.times):
                arrays["direct_pc"] = np.stack(direct_records)
            if output_dir is not None:
                output = Path(output_dir)
                output.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(output / "trajectories.npz", **arrays)
                atomic_json(output / "evaluation.json", report)
                save_figures(output)
                if settings.get("decode_images", False):
                    _decode_images(paths, direct_final, data, settings, device, output)
            return report
    finally:
        if student is not None:
            student.train(was_training)


def _decode_images(paths, direct_final, data, settings, device, output):
    """Preserve the existing optional AFHQ decoder, in teacher coordinates."""
    if tuple(data.sample_shape) != (4, 16, 16):
        raise ValueError("decode_images is only supported for AFHQ")
    from torchvision.utils import save_image
    from flow2pc.evaluation.afhq import AFHQDecoder

    decoder = AFHQDecoder(path=settings.get("vae_path", "checkpoints/metric_flow/vae"),
                          device=device)
    values = {name: path[-1] for name, path in paths.items()}
    if direct_final is not None:
        values["direct_pc"] = direct_final
    media_path = output / "media.json"
    media = json.loads(media_path.read_text()) if media_path.is_file() else {}
    for name, latent in values.items():
        path = output / f"{name}_decoded.png"
        save_image(decoder(latent[:16].to(device)), path, nrow=4)
        media[name + "_decoded"] = str(path.resolve())
    atomic_json(media_path, media)


def evaluate_checkpoint(checkpoint_path, *, output_dir=None, settings=None, device="cpu", role="student"):
    if role == "teacher":
        return evaluate_teacher_checkpoint(checkpoint_path, output_dir=output_dir, device=device, settings=settings)
    if role != "student":
        raise ValueError(f"Unknown evaluation role: {role!r}")
    from flow2pc.distill.runner import _identity, load_resources
    from flow2pc.distill.student import build_student_from_config
    from flow2pc.utils.experiment import ExperimentConfig
    from flow2pc.utils.mfm_config import checkpoint_training_config, validate_config
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") not in {"metric_pc_distill", "mmfm_pc_distill"}:
        raise ValueError("Expected an MFM- or MMFM-distilled student checkpoint")
    config = ExperimentConfig(**checkpoint_training_config(checkpoint))
    validate_config(config)
    device = torch.device(device)
    teacher, data = load_resources(config, device)
    if checkpoint["identity"] != _identity(teacher, data):
        raise ValueError("Teacher/data identity differs from the student checkpoint")
    student = build_student_from_config(checkpoint["model_config"]).to(device)
    student.load_state_dict(checkpoint["state_dict"], strict=True)
    destination = Path(output_dir) if output_dir is not None else checkpoint_path.resolve().parents[1] / "reevaluation"
    report = evaluate(student, teacher, data, {**config.evaluation, **(settings or {})}, device, destination)
    report.update(checkpoint_step=checkpoint["step"], checkpoint=str(checkpoint_path.resolve()))
    atomic_json(destination / "evaluation.json", report)
    return report


def evaluate_teacher_checkpoint(checkpoint, *, output_dir=None, device="cpu", settings=None):
    from flow2pc.pretrain.models.mmfm.adapter import load_teacher, load_teacher_data
    teacher = load_teacher({"checkpoint_path": str(checkpoint)}, device)
    data = load_teacher_data(teacher)
    destination = Path(output_dir) if output_dir else Path(checkpoint).resolve().parents[1] / "reevaluation"
    return evaluate(None, teacher, data, settings or {}, device, destination)


def save_figures(directory):
    """Observed populations and both prediction modes in raw feature coordinates."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    figures = {}
    with np.load(directory / "trajectories.npz") as arrays:
        times = arrays["times"]
        available = [i for i in range(len(times)) if f"reference_{i}" in arrays]
        metadata_path = directory / "evaluation.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        if metadata.get("protocol", {}).get("dataset") == "cell_tracking" and len(times) == 115:
            shown = [i for i in (0, 24, 49, 74, 99, 114) if i in available]
        else:
            shown = available if len(available) <= 8 else [available[i] for i in np.linspace(0, len(available) - 1, 6).astype(int)]
        modes = [name for name in ("teacher_ode", "student_ode", "direct_pc")
                 if name in arrays or any(f"{name}_{i}" in arrays or
                    f"benchmark_{name}_{i}" in arrays for i in shown)]
        if not modes or not shown:
            atomic_json(directory / "media.json", figures)
            return figures
        fig, axes = plt.subplots(len(modes), len(shown), figsize=(3.3 * len(shown), 3 * len(modes)), squeeze=False)
        for row, name in enumerate(modes):
            for col, index in enumerate(shown):
                ax = axes[row, col]
                observed = arrays[f"reference_{index}"]
                key = f"direct_pc_{index}" if name == "direct_pc" else name
                local_key = f"benchmark_{name}_{index}"
                if key not in arrays and local_key not in arrays:
                    ax.set_title(f"{name}: unavailable")
                    continue
                predicted = (arrays[local_key] if local_key in arrays else
                             arrays[key] if name == "direct_pc" else arrays[name][index])
                if observed.shape[-1] == 1:
                    ax.hist(observed[:, 0], bins=30, density=True, alpha=.45, label="Observed")
                    ax.hist(predicted[:, 0], bins=30, density=True, alpha=.45, label=name)
                else:
                    ax.scatter(observed[:, 0], observed[:, 1], s=3, alpha=.35, label="Observed")
                    ax.scatter(predicted[:, 0], predicted[:, 1], s=3, alpha=.35, label=name)
                ax.set_title(f"{name} | t={times[index]:.3g}")
                ax.legend(fontsize=7)
        fig.suptitle("Observed / teacher ODE / student ODE / direct PC | raw coordinates")
        fig.tight_layout()
        path = directory / "comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figures["comparison"] = str(path.resolve())
    atomic_json(directory / "media.json", figures)
    return figures


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate an MFM/MMFM-distilled PC; optional shared AFHQ VAE decode"
    )
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--ode-substeps", type=int)
    parser.add_argument("--decode-images", action="store_true")
    parser.add_argument("--vae-path", type=Path)
    args = parser.parse_args()
    settings = {}
    for key in ("samples", "ode_substeps", "vae_path"):
        value = getattr(args, key)
        if value is not None:
            settings[key] = str(value) if key == "vae_path" else value
    if args.decode_images:
        settings["decode_images"] = True
    if any(
        int(settings[key]) < 1 for key in ("samples", "ode_substeps") if key in settings
    ):
        parser.error("samples and ode-substeps must be positive")
    evaluate_checkpoint(
        args.student_checkpoint,
        output_dir=args.output_dir,
        settings=settings,
        device=args.device,
    )


if __name__ == "__main__":
    main()
