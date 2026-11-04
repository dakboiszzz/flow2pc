"""Separate direct-PC marginal metrics and paired teacher/student ODE metrics."""

from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from flow2pc.utils.checkpoint import atomic_json


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
            times = data.times.to(initial)
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
                values = (
                    data.test_marginals[index]
                    if hasattr(data, "test_marginals") and index in data.heldout_indices
                    else data.validation_marginals[index]
                )
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
    checkpoint_path, *, output_dir=None, settings=None, device="cpu", role="student"
):
    """Evaluate an MFM/MMFM student or an unconditional MMFM teacher."""
    if role == "teacher":
        return evaluate_teacher_checkpoint(
            checkpoint_path, output_dir=output_dir, device=device, settings=settings
        )
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
        raise ValueError("Teacher/data identity differs from the distilled checkpoint")
    student = build_student_from_config(checkpoint["model_config"]).to(device)
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
    if checkpoint["model_type"] == "mmfm_pc_distill":
        save_figures(destination)
    atomic_json(destination / "evaluation.json", report)
    return report


def evaluate_teacher_checkpoint(checkpoint, *, output_dir=None, device='cpu', settings=None):
    """Evaluate an unconditional MMFM teacher using shared ODE metrics."""
    from flow2pc.pretrain.models.mmfm.adapter import load_teacher, load_teacher_data
    options = {'samples': 128, 'projections': 32, 'ode_substeps': 16, 'seed': 1234, **(settings or {})}
    device = torch.device(device)
    teacher = load_teacher({'checkpoint_path': str(checkpoint)}, device)
    data = load_teacher_data(teacher)
    destination = Path(output_dir or Path(checkpoint).resolve().parents[1] / 'reevaluation')
    destination.mkdir(parents=True, exist_ok=True)
    with evaluation_seed(options['seed'], device), torch.no_grad():
        source = data.validation_marginals[0]
        initial = source[torch.randint(len(source), (options['samples'],))].to(device)
        trajectory = integrate(teacher.velocity, initial, data.times.to(device), options['ode_substeps'])
        scores = {}
        for i in range(len(data.times)):
            pool = data.test_marginals[i] if hasattr(data, 'test_marginals') and i in data.heldout_indices else data.validation_marginals[i]
            reference = pool[torch.randint(len(pool), (options['samples'],))].to(device)
            scores[str(i)] = {'teacher_ode_swd': sliced_wasserstein(trajectory[i], reference, options['projections']), 'heldout': i in data.heldout_indices}
        np.savez_compressed(destination / 'trajectories.npz', teacher_ode=trajectory.cpu().numpy(), times=data.times.numpy())
    report = {'method': 'mmfm', 'by_time': scores, 'times': data.times.tolist()}
    save_figures(destination)
    atomic_json(destination / 'evaluation.json', report)
    return report


def save_figures(directory):
    """Render saved ODE paths; high-dimensional datasets show coordinates 1 and 2."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    with np.load(directory / 'trajectories.npz') as arrays:
        times = arrays['times']
        for name in ('teacher_ode', 'student_ode'):
            if name not in arrays:
                continue
            paths = arrays[name]
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection='3d' if paths.shape[-1] == 3 else None)
            for i in range(min(64, paths.shape[1])):
                if paths.shape[-1] == 1:
                    ax.plot(times, paths[:, i, 0], alpha=0.3)
                elif paths.shape[-1] == 3:
                    ax.plot(*paths[:, i, :3].T, alpha=0.3)
                else:
                    ax.plot(*paths[:, i, :2].T, alpha=0.3)
            ax.set_title(name + ' (teacher input coordinates)')
            fig.savefig(directory / f'{name}.png', dpi=160, bbox_inches='tight')
            plt.close(fig)


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
