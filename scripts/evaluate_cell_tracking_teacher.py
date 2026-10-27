#!/usr/bin/env python3
"""Evaluate the selected Cell4 teacher using the existing eight-panel overlay.

Preparation only: this file runs nothing unless invoked explicitly. Evaluation
delegates to plot_checkpoint_wandb_overlay.py, preserving its native MLP,
global-time NeuralODE, all-pixel source, normalization and cyan/red rendering.
No teacher/student training or PC sampling is performed. This is a qualitative
full-mask evaluation, not a numerical EMD benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import runpy
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "d2c071a2b2fea1931c5c515d4dbc35e8f436043f58322fa0c8083b3503049129"
FRAME_INDICES = [0, 16, 32, 48, 65, 81, 97, 114]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ali-root", type=Path, default=ROOT.parent / "ali-cfm")
    parser.add_argument("--checkpoint", type=Path,
                        help="default: ../ali-cfm/changed_files/ali_cfm_model.pth, never latest-run")
    parser.add_argument("--expected-sha256", default=EXPECTED_SHA256,
                        help="pin the selected checkpoint to the handoff identity")
    parser.add_argument("--mask-stack", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path(
        "/data/lthung/flow2pc/cell_tracking/teacher_reference/teacher_d2c071a2b2fe_overlay.png"))
    parser.add_argument("--device", default="cpu", help="cpu or an explicitly selected cuda:N")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    args = parser.parse_args()
    if args.threads < 1 or not re.fullmatch(r"cpu|cuda:\d+", args.device):
        parser.error("threads must be positive; device must be cpu or cuda:N")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", args.expected_sha256):
        parser.error("expected-sha256 must contain exactly 64 hexadecimal characters")
    args.ali_root = args.ali_root.resolve()
    args.checkpoint = (args.checkpoint or args.ali_root / "changed_files/ali_cfm_model.pth").resolve()
    args.mask_stack = (args.mask_stack or args.ali_root /
                       "data/cell_tracking/exports/Cell4_masks/mask_cell4_stack.npy").resolve()
    args.image_dir = (args.image_dir or args.ali_root /
                      "data/cell_tracking/PhC-C2DH-U373/01").resolve()
    args.output = args.output.resolve()
    if args.output.suffix.lower() != ".png":
        parser.error("output must be a PNG path")
    return args


def main() -> None:
    args = parse_args()
    overlay_script = ROOT / "plot_checkpoint_wandb_overlay.py"
    native_nets = args.ali_root / "ali_cfm/nets.py"
    for path in (args.checkpoint, args.mask_stack, overlay_script, native_nets):
        if not path.is_file():
            raise FileNotFoundError(path)
    for frame in FRAME_INDICES:
        path = args.image_dir / f"t{frame:03d}.tif"
        if not path.is_file():
            raise FileNotFoundError(path)
    digest = sha256(args.checkpoint)
    if digest != args.expected_sha256.lower():
        raise ValueError(
            f"Checkpoint differs from the selected teacher in CELL_TRACKING_TEACHER_README.md. "
            f"Expected {args.expected_sha256.lower()}, found {digest}. "
            "Do not substitute the archived latest-run checkpoint.")

    # changed_files is a partial patch tree; the complete native package is at
    # ali-root/ali_cfm. Its MLP supplies the saved cfm_model architecture.
    sys.path.insert(0, str(args.ali_root))
    nets = importlib.import_module("ali_cfm.nets")
    if Path(nets.__file__).resolve() != native_nets.resolve():
        raise ImportError(f"ali_cfm.nets came from the wrong checkout: {nets.__file__}")
    import torch

    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda:"):
        index = int(args.device.split(":")[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"Requested {args.device} is unavailable; select an available GPU or --device cpu")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(args.output.parent / ".mplconfig"))
    original_argv = sys.argv
    try:
        sys.argv = [str(overlay_script), "--checkpoint", str(args.checkpoint),
                    "--mask-stack", str(args.mask_stack), "--image-dir", str(args.image_dir),
                    "--output", str(args.output), "--device", args.device]
        runpy.run_path(str(overlay_script), run_name="__main__")
    finally:
        sys.argv = original_argv
    if not args.output.is_file():
        raise RuntimeError("Overlay evaluation did not produce the expected PNG")

    metadata = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest,
        "mask_stack": str(args.mask_stack), "mask_stack_sha256": sha256(args.mask_stack),
        "image_dir": str(args.image_dir), "output": str(args.output), "device": args.device,
        "native_mlp_source": str(native_nets), "native_mlp_sha256": sha256(native_nets),
        "overlay_source_sha256": sha256(overlay_script),
        "teacher_weights": "cfm_model", "time_convention": "global t=i/114",
        "normalization": "checkpoint min_max, checked against all mask bounds by overlay script",
        "coordinate_order": "x,y pixels; raw mask indices y,x are swapped",
        "source": "every nonzero Cell4 pixel in frame 0, not the ten training points",
        "solver": "native torchdyn NeuralODE with checkpoint ode_solver and native defaults",
        "frames_zero_based": FRAME_INDICES, "frames_one_based": [f + 1 for f in FRAME_INDICES],
        "colors": {"cyan": "actual Cell4 mask pixels", "red": "CFM ODE prediction"},
        "emd_computed": False,
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Evaluation metadata: {metadata_path}")

    if args.wandb_mode != "disabled":
        # Upload after writing local results so a tracking outage cannot discard
        # the requested overlay. No training or pretrained-model modification.
        run = None
        try:
            import wandb
            run = wandb.init(project="flow2pc", entity=args.wandb_entity, mode=args.wandb_mode,
                             job_type="teacher-evaluation", group="cell-tracking-teacher",
                             name=f"cell-tracking-teacher-{digest[:12]}-full-mask-overlay",
                             dir=str(args.output.parent), config=metadata)
            run.log({"teacher/full_mask_overlay": wandb.Image(str(args.output))})
            run.save(str(args.output), base_path=str(args.output.parent))
            run.save(str(metadata_path), base_path=str(args.output.parent))
            print(f"W&B: {run.url}")
        except Exception as error:
            print(f"Overlay and metadata are saved locally; W&B upload failed: {error}", file=sys.stderr)
        finally:
            if run is not None:
                run.finish()


if __name__ == "__main__":
    main()
