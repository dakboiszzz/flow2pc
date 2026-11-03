"""Export Arch/Sphere/LiDAR student ODE figures with MFM pretrain plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flow2pc.data.metric_common import seeded
from flow2pc.data.metric_registry import load_dataset
from flow2pc.distill.student import build_student_from_config
from flow2pc.utils.mfm_source import fingerprint, source_module


@torch.no_grad()
def export(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_type") != "metric_pc_distill":
        raise ValueError(f"Not an MFM student checkpoint: {checkpoint_path}")
    config = checkpoint["training_config"]
    name = config["dataset"]["name"]
    if name not in ("arch", "sphere", "lidar"):
        raise ValueError("Only Arch, Sphere and LiDAR are supported")
    settings = {
        **config["dataset"].get("options", {}),
        "name": name,
        "batch_size": int(config["training"].get("batch_size", 128)),
        "seed": int(config["training"].get("seed", 42)),
    }
    if config["dataset"].get("data_dir") is not None:
        settings["data_dir"] = config["dataset"]["data_dir"]
    data = load_dataset(settings, {"teacher_metadata": checkpoint["teacher_metadata"]})
    identity = checkpoint["identity"]
    if (
        identity["data"] != data.source_info["snapshots_sha256"]
        or identity["split"] != data.split_indices
        or identity["normalization"] != data.normalization
        or identity["heldout"] != data.heldout_indices
    ):
        raise ValueError(f"Dataset identity differs from checkpoint: {checkpoint_path}")
    if name == "lidar":
        metadata = checkpoint["teacher_metadata"]
        native_args = dict(metadata["args"])
        native_args["data_path"] = data.source_info["files"][0]["path"]
        with seeded(int(native_args.get("seed_current", 42))):
            native = source_module("dataloaders.lidar_data", metadata.get("repo_path"))
            scene = native.LidarDataModule(SimpleNamespace(**native_args))
        left, right = 0, 1
        # Upstream plot_lidar displays the first 512 validation trajectories.
        source = scene.val_x0[:512]
        if not torch.equal(source, data.marginals[0][-len(scene.val_x0):][:512]):
            raise ValueError("LiDAR plot source differs from the checkpoint data")
        endpoints = [source]
        count = len(source)
    else:
        if len(data.heldout_indices) != 1:
            raise ValueError("Expected one interior held-out time, as in MFM pretraining")
        middle = data.heldout_indices[0]
        left, right = middle - 1, middle + 1
        source, target = data.marginals[left], data.marginals[right]
        count = min(400 if name == "arch" else 100, len(source), len(target))
        rng = np.random.default_rng(1234)
        source_indices = rng.choice(len(source), count, replace=False)
        target_indices = rng.choice(len(target), count, replace=False)
        endpoints = [source[source_indices], target[target_indices]]
    student = build_student_from_config(checkpoint["model_config"]).to(device)
    student.load_state_dict(checkpoint["state_dict"])
    student.eval()
    x = endpoints[0].to(device)
    times = torch.linspace(float(data.times[left]), float(data.times[right]), 101, device=device)
    path = [x.cpu().clone()]
    for index in range(100):
        x = x + (times[index + 1] - times[index]) * student.velocity(x, times[index])
        if not torch.isfinite(x).all():
            raise FloatingPointError(f"Nonfinite student ODE at step {index + 1}")
        path.append(x.cpu().clone())
    trajectory = torch.stack(path)
    plotting = source_module("utils", checkpoint["teacher_metadata"].get("repo_path"))
    destination = checkpoint_path.parent.parent
    with seeded(1234):
        if name == "arch":
            figures = [plotting.plot_arch(
                endpoints, trajectory, [0, 1], str(destination / "arch_trajs.png"),
                n_samples=count,
            )]
            names = ["arch_trajs.png"]
        elif name == "sphere":
            figures = plotting.plot_sphere(
                endpoints, trajectory, [0, 1], str(destination / "sphere_trajs.png"),
                n_samples=count,
            )
            names = ["sphere_trajs.png"] + [f"sphere_trajs_view_{i}.png" for i in (2, 3, 4)]
        else:
            cloud = scene.dataset
            if native_args.get("whiten", False):
                shape = trajectory.shape
                trajectory = torch.as_tensor(scene.scaler.inverse_transform(
                    trajectory.reshape(-1, 3).numpy()
                ).reshape(shape))
                cloud = torch.as_tensor(scene.scaler.inverse_transform(cloud.numpy()))
            figure = plt.figure(figsize=(5, 4))
            ax = figure.add_subplot(111, projection="3d", computed_zorder=False)
            ax.view_init(elev=30, azim=-115, roll=0)
            plotting.plot_lidar(ax, cloud, xs=trajectory.transpose(0, 1))
            figure.savefig(destination / "lidar_interpolant.png", dpi=300)
            figures = [figure]
            names = ["lidar_interpolant.png"]
    for figure in figures:
        plt.close(figure)
    provenance = {
        "dataset": name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": fingerprint(checkpoint_path),
        "checkpoint_step": checkpoint["step"],
        "best_validation_loss": checkpoint["best_validation_loss"],
        "trajectory": "student velocity ODE; Euler 100 steps, matching pretrain plotting protocol",
        "time_interval": [float(times[0]), float(times[-1])],
        "samples": count,
        "seed": 1234,
        "endpoint_points": "independently sampled real source and target marginals, as in pretrain",
        "lines": "student ODE trajectories started at the selected real source points",
        "plot_source": plotting.__file__,
        "plot_source_sha256": fingerprint(plotting.__file__),
        "figures": names,
    }
    if name == "lidar":
        provenance.update(
            endpoint_points="native LiDAR validation source, as in pretrain",
            lines="student ODE states shown at 25 times over the complete ground scene",
            coordinate_space="original pretrain scene coordinates (whitening inverted)",
            source_seed=int(native_args.get("seed_current", 42)),
            scene_points=len(cloud),
        )
    (destination / "pretrain_plots.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    print(f"{name}/{destination.name}: {len(names)} images from {checkpoint_path.name} "
          f"(step {checkpoint['step']})", flush=True)
    return len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/tuning/mfm")
    parser.add_argument("--datasets", nargs="+", choices=("arch", "sphere", "lidar"), default=["arch", "sphere", "lidar"])
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoints = [
        path for name in args.datasets
        for path in sorted((args.root / name).glob(f"*/runs/*/checkpoints/{args.checkpoint}.pt"))
    ]
    if not checkpoints:
        parser.error(f"No matching checkpoints under {args.root}")
    print(f"Rendering {len(checkpoints)} runs on {args.device}", flush=True)
    total = 0
    for checkpoint in checkpoints:
        total += export(checkpoint, torch.device(args.device))
    print(f"Done: {total} images across {len(checkpoints)} runs", flush=True)


if __name__ == "__main__":
    main()
