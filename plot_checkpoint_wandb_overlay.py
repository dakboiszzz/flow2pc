#!/usr/bin/env python3
"""Save the eight-panel cyan-data/red-CFM microscopy overlay for a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper

from ali_cfm.nets import MLP


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mask-stack", type=Path, required=True,
                        help="Cell4 mask stack with 115 frames")
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="Directory containing t000.tif through t114.tif")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda")
                          and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    min_xy, max_xy = (torch.as_tensor(v, dtype=torch.float32) for v in checkpoint["min_max"])
    scale = max_xy - min_xy
    if min_xy.shape != (2,) or max_xy.shape != (2,) or (scale <= 0).any():
        raise ValueError("Checkpoint must contain valid 2D global min-max bounds")

    masks = np.load(args.mask_stack, mmap_mode="r")
    if masks.ndim != 3 or masks.shape[0] != 115:
        raise ValueError(f"Expected mask stack (115,H,W), got {masks.shape}")
    data_pixels = [np.argwhere(mask)[:, [1, 0]].astype(np.float32) for mask in masks]
    if any(len(frame) == 0 for frame in data_pixels):
        raise ValueError("Every frame must contain Cell4 pixels")
    data_min = np.concatenate(data_pixels).min(axis=0)
    data_max = np.concatenate(data_pixels).max(axis=0)
    if not (np.allclose(data_min, min_xy.numpy())
            and np.allclose(data_max, max_xy.numpy())):
        raise ValueError(
            "Mask stack bounds do not match checkpoint min_max: "
            f"masks=({data_min},{data_max}), checkpoint=({min_xy.numpy()},{max_xy.numpy()})"
        )

    model = MLP(dim=int(config["dim"]), time_varying=True,
                w=int(config["cfm_dim"])).to(device)
    model.load_state_dict(checkpoint["cfm_model"], strict=True)
    model.eval().requires_grad_(False)

    x0_pixels = torch.from_numpy(data_pixels[0])
    x0_normalized = ((x0_pixels - min_xy) / scale).to(device)
    times = torch.linspace(0.0, 1.0, len(masks), device=device)
    node = NeuralODE(torch_wrapper(model), solver=str(config.get("ode_solver", "dopri5")),
                     sensitivity="adjoint")
    with torch.no_grad():
        trajectory = node.trajectory(x0_normalized, t_span=times)
    trajectory_pixels = (trajectory.cpu() * scale + min_xy).numpy()
    if not np.isfinite(trajectory_pixels).all():
        raise FloatingPointError("CFM rollout contains nonfinite pixel coordinates")

    frames = np.linspace(0, len(masks) - 1, 8, dtype=int)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, frame in zip(axes.flat, frames):
        image_path = args.image_dir / f"t{frame:03d}.tif"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            ax.imshow(np.asarray(image), cmap="gray")
        ax.scatter(*data_pixels[frame].T, s=1, alpha=0.12, color="cyan")
        ax.scatter(*trajectory_pixels[frame].T, s=2, alpha=0.35, color="red")
        ax.set_title(f"Frame {frame}: cyan=data, red=CFM")
        ax.axis("off")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlay: {args.output}")
    print(f"Frames: {frames.tolist()}; particles per frame: {len(trajectory_pixels[0])}")


if __name__ == "__main__":
    main()
