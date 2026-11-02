"""Separate direct-PC marginal metrics and paired teacher/student ODE metrics."""

from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch


@contextmanager
def evaluation_seed(seed, device):
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    numpy_state = np.random.get_state()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(numpy_state)


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
                raise FloatingPointError(
                    "Nonfinite MFM ODE trajectory; reduce the ODE step size"
                )
        states.append(values.clone())
    return torch.stack(states)


@torch.no_grad()
def sliced_wasserstein(x, y, projections=32):
    x, y = x.flatten(1), y.flatten(1)
    directions = torch.randn(x.shape[1], projections, device=x.device, dtype=x.dtype)
    directions /= directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    q = torch.linspace(0, 1, min(256, max(len(x), len(y))), device=x.device)
    left = torch.quantile(x @ directions, q, dim=0)
    right = torch.quantile(y @ directions, q, dim=0)
    return (left - right).square().mean().sqrt().item()


@torch.no_grad()
def evaluate(student, teacher, data, settings, device, output_dir=None):
    samples = int(settings.get("samples", 128))
    steps = int(settings.get("ode_substeps", 8))
    projections = int(settings.get("projections", 32))
    seed = int(settings.get("seed", 1234))
    was_training = student.training
    student.eval()
    try:
        with evaluation_seed(seed, device):
            source = data.validation_marginals[0]
            initial = (
                source[torch.randint(len(source), (samples,))].flatten(1).to(device)
            )
            times = data.times.to(device)
            teacher_path = integrate(
                lambda x, t: teacher.velocity(x, t), initial, times, steps
            )
            student_path = integrate(
                lambda x, t: student.velocity(x, t), initial, times, steps
            )
            report = {
                "space": "teacher_input_coordinates",
                "times": times.cpu().tolist(),
                "heldout_indices": data.heldout_indices,
                "by_time": {},
            }
            direct_samples = []
            for index, t in enumerate(times):
                values = data.validation_marginals[index]
                reference = (
                    values[torch.randint(len(values), (samples,))].flatten(1).to(device)
                )
                direct = student.sample(samples, t).flatten(1)
                if not torch.isfinite(direct).all():
                    raise FloatingPointError("Nonfinite direct PC samples")
                direct_samples.append(direct)
                metrics = {
                    "direct_pc_swd": sliced_wasserstein(direct, reference, projections),
                    "student_ode_swd": sliced_wasserstein(
                        student_path[index], reference, projections
                    ),
                    "teacher_ode_swd": sliced_wasserstein(
                        teacher_path[index], reference, projections
                    ),
                    "direct_pc_teacher_swd": sliced_wasserstein(
                        direct, teacher_path[index], projections
                    ),
                    "paired_ode_mse": (student_path[index] - teacher_path[index])
                    .square()
                    .mean()
                    .item(),
                    "nll_per_dimension": -student.log_prob(reference, t).mean().item()
                    / data.dim,
                }
                if "mean" in data.normalization:
                    mean = torch.tensor(data.normalization["mean"], device=device)
                    scale = torch.tensor(data.normalization["scale"], device=device)
                    metrics["direct_pc_swd_before_whitening"] = sliced_wasserstein(
                        direct * scale + mean, reference * scale + mean, projections
                    )
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise FloatingPointError("Nonfinite MFM evaluation metrics")
                report["by_time"][str(index)] = metrics
            if output_dir is not None:
                output_dir = Path(output_dir)
                np.savez_compressed(
                    output_dir / "trajectories.npz",
                    times=times.cpu().numpy(),
                    teacher_ode=teacher_path.cpu().numpy(),
                    student_ode=student_path.cpu().numpy(),
                    direct_pc=torch.stack(direct_samples).cpu().numpy(),
                )
                if settings.get("decode_images", False):
                    if data.sample_shape != (4, 16, 16):
                        raise ValueError("decode_images is only supported for AFHQ")
                    from torchvision.utils import save_image

                    from flow2pc.evaluation.afhq import AFHQDecoder

                    decoder = AFHQDecoder(
                        path=settings.get("vae_path", "checkpoints/metric_flow/vae"),
                        device=device,
                    )
                    for name, latent in (
                        ("teacher", teacher_path[-1]),
                        ("student_ode", student_path[-1]),
                        ("student_direct", direct_samples[-1]),
                    ):
                        save_image(
                            decoder(latent[:16]), output_dir / f"{name}.png", nrow=4
                        )
            return report
    finally:
        student.train(was_training)


def evaluate_checkpoint(
    checkpoint_path, *, output_dir=None, settings=None, device="cpu"
):
    """Evaluate or decode an existing student without another training step."""
    import json

    from flow2pc.utils.experiment import ExperimentConfig
    from flow2pc.distill.mfm import (
        _identity,
        build_student,
        load_resources,
        validate_config,
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "metric_pc_distill":
        raise ValueError("Expected a metric_pc_distill student checkpoint")
    config = ExperimentConfig(**checkpoint["training_config"])
    validate_config(config)
    device = torch.device(device)
    teacher, data = load_resources(config, device)
    if checkpoint["identity"] != _identity(teacher, data):
        raise ValueError("Teacher/data identity differs from the distilled checkpoint")
    student = build_student(checkpoint["model_config"]).to(device)
    student.load_state_dict(checkpoint["state_dict"])
    destination = (
        Path(output_dir)
        if output_dir is not None
        else checkpoint_path.resolve().parents[1] / "reevaluation"
    )
    destination.mkdir(parents=True, exist_ok=True)
    report = evaluate(
        student,
        teacher,
        data,
        {**config.evaluation, **(settings or {})},
        device,
        destination,
    )
    (destination / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate an MFM-distilled PC; optional shared AFHQ VAE decode"
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
