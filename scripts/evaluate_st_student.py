#!/usr/bin/env python3
"""Evaluate one distilled ST PC and log metrics/media to its W&B run."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if not sys.path or sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

# The training launcher may run on nodes where ~/.config is read-only. Set a
# writable cache before importing matplotlib, otherwise the first evaluation
# can block while repeatedly rebuilding its font cache.
_mpl_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "flow2pc-matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from PIL import Image
from scipy.optimize import linear_sum_assignment

from repc.data import load_st_clouds
from train import build_student, resolve_device


def emd_assignment(pred: np.ndarray, target: np.ndarray, max_points: int, seed: int) -> float:
    """Approximate balanced 2-D EMD using a Hungarian assignment."""
    rng = np.random.default_rng(seed)
    n = min(len(pred), len(target), max_points)
    pred = pred[rng.choice(len(pred), n, replace=False)]
    target = target[rng.choice(len(target), n, replace=False)]
    cost = np.sqrt(((pred[:, None, :] - target[None, :, :]) ** 2).sum(axis=-1))
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--held-out-slide", choices=("U2", "U3", "U4", "U5"), required=True)
    p.add_argument("--reference-dir", type=Path, required=True)
    p.add_argument(
        "--he-image-dir",
        type=Path,
        default=Path("outputs/ST/he_visualization_package/images/he_stains"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=1024)
    p.add_argument(
        "--ode-steps",
        type=int,
        default=100,
        help="RK4 steps used to integrate the learned velocity field from U2 to each ST time",
    )
    p.add_argument("--ode-batch-size", type=int, default=64)
    p.add_argument("--velocity-batch-size", type=int, default=64)
    p.add_argument("--max-emd-points", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--wandb-project", default="flow2pc")
    p.add_argument("--wandb-entity")
    p.add_argument("--wandb-run-name", required=True)
    p.add_argument("--wandb-run-id")
    p.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return p.parse_args()


@torch.no_grad()
def integrate_rk4_student(
    model: torch.nn.Module,
    x0: torch.Tensor,
    num_steps: int,
) -> torch.Tensor:
    """Integrate the learned PC velocity field and retain every ODE step."""
    if num_steps < 1:
        raise ValueError("--ode-steps must be positive")
    device = x0.device
    dt = 1.0 / num_steps
    time_steps = torch.linspace(
        0.0, 1.0, num_steps + 1, device=device, dtype=x0.dtype
    )
    trajectory = [x0.clone()]
    current_x = x0.clone()
    for step in range(num_steps):
        t_curr = time_steps[step].expand(current_x.shape[0])
        t_mid = (time_steps[step] + 0.5 * dt).expand(current_x.shape[0])
        t_next = time_steps[step + 1].expand(current_x.shape[0])
        k1 = model.velocity(current_x, t_curr)
        k2 = model.velocity(current_x + 0.5 * dt * k1, t_mid)
        k3 = model.velocity(current_x + 0.5 * dt * k2, t_mid)
        k4 = model.velocity(current_x + dt * k3, t_next)
        current_x = current_x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory.append(current_x.clone())
    return torch.stack(trajectory, dim=1)


def integrate_rk4_chunked(model, x0_cpu, num_steps, batch_size, device):
    """Integrate particles in chunks so HCLT activations stay bounded."""
    if batch_size < 1:
        raise ValueError("--ode-batch-size must be positive")
    chunks = []
    for start in range(0, len(x0_cpu), batch_size):
        x0 = x0_cpu[start : start + batch_size].to(device)
        chunks.append(integrate_rk4_student(model, x0, num_steps).cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def batched_log_prob(model, values_cpu, time, batch_size, device):
    parts = []
    for start in range(0, len(values_cpu), batch_size):
        values = values_cpu[start : start + batch_size].to(device)
        parts.append(model.log_prob(values, time.expand(len(values))).detach().cpu())
    return torch.cat(parts)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_student(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    slides = ("U2", "U3", "U4", "U5")
    clouds = load_st_clouds(args.data_dir, normalized=True, slides=slides)
    held_out_index = slides.index(args.held_out_slide)
    times = torch.tensor([0.0, 1 / 3, 2 / 3, 1.0], device=device)
    # ST distillation is trained on the U2 -> {U3,U4,U5} flow. Evaluation
    # must follow the learned velocity field from U2, as in
    # plot_knot_student.py, rather than drawing independent PC marginals.
    source_cpu = clouds[0]
    source_indices = torch.arange(args.num_samples) % len(source_cpu)
    x0 = source_cpu[source_indices]
    ode_path = integrate_rk4_chunked(
        model, x0, args.ode_steps, args.ode_batch_size, device
    )
    step_indices = [
        min(args.ode_steps, int(round(float(time) * args.ode_steps)))
        for time in times.cpu()
    ]
    # Keep the evaluator's existing [time, sample, coordinate] convention.
    path = ode_path[:, step_indices].permute(1, 0, 2).cpu().numpy()

    target = clouds[held_out_index].numpy()
    prediction = path[held_out_index]
    emd = emd_assignment(prediction, target, args.max_emd_points, args.seed)
    mean_mse = float(((prediction.mean(0) - target.mean(0)) ** 2).mean())
    var_mse = float(((prediction.var(0) - target.var(0)) ** 2).mean())
    endpoint_log_prob = batched_log_prob(
        model,
        torch.as_tensor(target),
        times[held_out_index],
        args.velocity_batch_size,
        device,
    )
    endpoint_nll = float(-endpoint_log_prob.mean().item())

    # Compare the velocity field on the same generator path used by training.
    with torch.no_grad():
        velocity_batch_size = min(
            args.velocity_batch_size, len(clouds[0]), len(clouds[-1])
        )
        source = clouds[0][:velocity_batch_size]
        destination = clouds[-1][:velocity_batch_size]
        query_times = torch.tensor([0.1, 1 / 3, 0.5, 2 / 3, 0.9], device=device)
        velocity_errors = []
        teacher_energy = []
        student_energy = []
        from repc.flow import ALICFMGeneratorWrapper, ALICFMTeacherWrapper
        teacher = ALICFMTeacherWrapper(
            args.reference_dir / "teacher.pth", map_to_interval=False
        ).to(device).eval()
        generator_model = ALICFMGeneratorWrapper(
            args.reference_dir / "generator.pth"
        ).to(device).eval()
        for query_time in query_times:
            batch_time = query_time.expand(len(source))
            generated = generator_model(source.to(device), destination.to(device), batch_time)
            teacher_velocity = teacher(batch_time, generated)
            student_velocity = model.velocity(generated, batch_time)
            velocity_errors.append((student_velocity - teacher_velocity).square().mean())
            teacher_energy.append(teacher_velocity.square().mean())
            student_energy.append(student_velocity.square().mean())
        velocity_mse = float(torch.stack(velocity_errors).mean().item())
        teacher_rms = float(torch.stack(teacher_energy).mean().sqrt().item())
        student_rms = float(torch.stack(student_energy).mean().sqrt().item())
    relative_velocity_mse = velocity_mse / max(teacher_rms**2, 1e-12)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    teacher_coordinates = np.load(args.reference_dir / f"teacher_{args.held_out_slide}_coordinates.npy")
    generator_coordinates = np.load(args.reference_dir / f"generator_{args.held_out_slide}_coordinates.npy")
    # The published ALI-CFM prediction arrays are in raw image coordinates;
    # the Flow2PC student and CSV clouds use the shared [0, 1] normalization.
    metadata = json.loads((args.data_dir / "metadata.json").read_text())
    coordinate_min = np.asarray(metadata["min"], dtype=np.float32)
    coordinate_span = np.asarray(metadata["span"], dtype=np.float32)
    teacher_coordinates = (teacher_coordinates - coordinate_min) / coordinate_span
    generator_coordinates = (generator_coordinates - coordinate_min) / coordinate_span
    target_raw = target * coordinate_span + coordinate_min
    prediction_raw = prediction * coordinate_span + coordinate_min
    teacher_raw = teacher_coordinates * coordinate_span + coordinate_min
    generator_raw = generator_coordinates * coordinate_span + coordinate_min
    student_emd_raw = emd_assignment(
        prediction_raw, target_raw, args.max_emd_points, args.seed
    )
    teacher_emd_raw = emd_assignment(
        teacher_raw, target_raw, args.max_emd_points, args.seed
    )
    generator_emd_raw = emd_assignment(
        generator_raw, target_raw, args.max_emd_points, args.seed
    )
    # Render on the clean TIFF supplied by the visualization package. Its
    # Plotter convention is raw coordinate column 1 -> horizontal pixel axis
    # and column 0 -> vertical pixel axis.
    he_path = args.he_image_dir / f"{args.held_out_slide}_aligned_to_U5.tif"
    if not he_path.is_file():
        raise FileNotFoundError(
            f"clean HE image not found: {he_path}; refusing to use an annotated PNG as a background"
        )
    he_image = np.asarray(Image.open(he_path).convert("RGB"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    panels = (
        (target_raw, "ground truth", "red"),
        (teacher_raw, "teacher", "red"),
        (prediction_raw, "student ODE (RK4)", "red"),
    )
    for axis, (samples, title, color) in zip(axes, panels):
        axis.imshow(he_image, origin="upper")
        axis.scatter(
            samples[:, 1], samples[:, 0], s=10, alpha=0.7,
            facecolors="none", edgecolors=color, linewidths=0.8,
        )
        axis.set_title(title)
        axis.set_xlim(0, he_image.shape[1])
        axis.set_ylim(he_image.shape[0], 0)
        axis.axis("off")
    fig.suptitle(
        f"ST H&E overlay: {args.held_out_slide} at t={held_out_index / 3:.2f}"
    )
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    metrics = {
        "eval/st_held_out_emd_normalized": emd,
        "eval/st_held_out_emd_raw": student_emd_raw,
        "eval/st_teacher_emd_raw": teacher_emd_raw,
        "eval/st_generator_emd_raw": generator_emd_raw,
        "eval/st_held_out_mean_mse": mean_mse,
        "eval/st_held_out_var_mse": var_mse,
        "eval/st_held_out_nll_nats": endpoint_nll,
        "eval/st_velocity_mse": velocity_mse,
        "eval/st_relative_velocity_mse": relative_velocity_mse,
        "eval/st_teacher_velocity_rms": teacher_rms,
        "eval/st_student_velocity_rms": student_rms,
        "eval/st_num_samples": args.num_samples,
        "eval/st_ode_steps": args.ode_steps,
        "eval/st_held_out_index": held_out_index,
    }
    metrics_path = args.output.with_name("st_eval_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    with wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        id=args.wandb_run_id,
        name=args.wandb_run_name,
        resume="must" if args.wandb_run_id else "allow",
        mode=args.wandb_mode,
        dir=str(args.output.parent.parent),
    ) as run:
        run.log({
            **metrics,
            "eval/st_he_overlay": wandb.Image(
                str(args.output),
                caption=f"Clean H&E overlay: {args.held_out_slide}; ground truth vs teacher vs student",
            ),
        })
        # Keep the PNG as a directly downloadable W&B run file as well as a
        # media panel entry. This makes the artifact discoverable even when a
        # UI view hides media keys behind the run's history table.
        run.save(str(args.output), policy="now")
        for key, value in metrics.items():
            run.summary[key] = value
    print(" ".join(f"{key}={value:.6f}" for key, value in metrics.items() if isinstance(value, float)))


if __name__ == "__main__":
    main()
