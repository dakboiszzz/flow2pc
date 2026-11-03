#!/usr/bin/env python3
"""Launch existing recipes against the local bundle; no model imports in preview."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ALI_DATASETS = (
    "cite5", "cite50", "cite100", "multi5", "multi50", "multi100",
    "eb", "st", "cell_tracking", "knot",
)
TRANSPORT_DATASETS = ("gom", "beijing")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("pretrain", "distill", "both"))
    result.add_argument("dataset", choices=(*ALI_DATASETS, *TRANSPORT_DATASETS))
    result.add_argument("--method", choices=("alicfm", "otpfm"), default="alicfm")
    result.add_argument("--recipe", choices=("w2", "w2inf", "kl", "mmd"), default="w2inf",
                        help="OTP-FM transport recipe; other profiles only use w2inf")
    result.add_argument("--data-root", type=Path, default=REPO_ROOT / "data",
                        help="Unified dataset directory (default: repository data/)")
    result.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/local")
    result.add_argument("--run-name", default="default", help="Separate output namespace")
    result.add_argument("--teacher-checkpoint", type=Path,
                        help="Existing framework teacher checkpoint, for distill only")
    result.add_argument("--device", help="Override training/distillation device, e.g. cuda:0")
    result.add_argument("--logging", choices=("disabled", "offline", "online"),
                        default="disabled")
    result.add_argument("--python", default=os.environ.get("FLOW2PC_PYTHON", sys.executable))
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="Print commands only; do not load data or write files")
    mode.add_argument("--validate-only", action="store_true",
                      help="Validate one stage with main.py; distill needs a teacher")
    return result


def required_files(dataset, root):
    """Only inspect paths here; loaders validate contents when explicitly invoked."""
    if dataset in TRANSPORT_DATASETS:
        return [root / dataset / "marginals.npz"]
    if dataset.startswith("cite"):
        return [root / "single_cell/op_cite_inputs_0.h5ad"]
    if dataset.startswith("multi"):
        return [root / "single_cell/op_train_multi_targets_0.h5ad"]
    if dataset == "eb":
        return [root / "single_cell/ebdata_v3.h5ad"]
    if dataset == "st":
        directory = root / "spatial_transcriptomics/aligned/aligned_spots"
        return [directory / f"U{i}_tumor_coordinates.csv" for i in range(2, 6)]
    if dataset == "cell_tracking":
        return [root / "cell_tracking/prepared/mask_cell4_stack.npy"]
    return []  # Knot uses the existing analytic generator, not stored trajectories.


def build_commands(args):
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        raise ValueError("--run-name must be a single directory name")
    if args.recipe != "w2inf" and (
        args.method != "otpfm" or args.dataset not in TRANSPORT_DATASETS
    ):
        raise ValueError("--recipe applies only to OTP-FM GOM/Beijing")
    if args.teacher_checkpoint and args.stage != "distill":
        raise ValueError("--teacher-checkpoint applies only to distill")
    if args.validate_only and args.stage == "both":
        raise ValueError("Validate stages separately: distill requires a trained teacher")
    if args.stage != "pretrain":
        if args.method == "otpfm":
            raise ValueError(
                "OTP-FM distillation is not integrated: its conditional tuple sampler "
                "cannot be replaced by the ALI-CFM endpoint interpolant."
            )
        if args.dataset in TRANSPORT_DATASETS:
            raise ValueError(
                "GOM/Beijing pretraining is available, but distillation needs an explicit "
                "validation design: the transport protocol has no row-validation split. "
                "Temporal test holdouts must not be reused for student selection."
            )

    data_root = args.data_root.expanduser().resolve()
    run = args.output_root.expanduser().resolve() / args.method / args.dataset
    if args.method == "otpfm":
        run /= args.recipe
    run /= args.run_name
    checkpoint = (
        args.teacher_checkpoint.expanduser().resolve() if args.teacher_checkpoint
        else run / "pretrain/checkpoints/last.pt"
    )
    stages = ("pretrain", "distill") if args.stage == "both" else (args.stage,)
    commands = []
    for stage in stages:
        if args.method == "alicfm":
            directory = REPO_ROOT / "configs" / stage
            if args.dataset in ALI_DATASETS:
                directory /= "ali_cfm"
            config = directory / f"alicfm_{args.dataset}.yaml"
        else:
            suffix = f"_{args.recipe}" if args.dataset in TRANSPORT_DATASETS else ""
            config = REPO_ROOT / "configs/pretrain" / f"otpfm_{args.dataset}{suffix}.yaml"
        if not config.is_file():
            raise ValueError(f"Missing recipe: {config}")
        command = [args.python, str(REPO_ROOT / "main.py"), "--config", str(config)]
        overrides = {
            "output.dir": str(run / stage),
            "logging.mode": args.logging,
        }
        if stage == "pretrain":
            overrides["data.root"] = str(data_root)
            if args.method == "otpfm" and args.dataset in TRANSPORT_DATASETS:
                overrides["model.params.native_config.data_dir"] = str(data_root / args.dataset)
        else:
            # Data, time mapping and preprocessing are restored from this teacher.
            overrides["teacher.checkpoint"] = str(checkpoint)
        if args.device:
            key = "training.device" if stage == "pretrain" else "distillation.device"
            overrides[key] = args.device
        for key, value in overrides.items():
            command.extend(("--set", f"{key}={json.dumps(value)}"))
        if args.validate_only:
            command.append("--validate-only")
        commands.append(command)
    return commands, data_root, checkpoint


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        commands, data_root, checkpoint = build_commands(args)
        if not args.dry_run:
            # For existing teachers their manifest, not --data-root, owns the data path.
            if args.stage != "distill":
                missing = [path for path in required_files(args.dataset, data_root)
                           if not path.is_file()]
                if missing:
                    raise ValueError("Missing local data: " + ", ".join(map(str, missing)))
            if args.stage == "distill" and not checkpoint.is_file():
                raise ValueError(f"Teacher checkpoint does not exist: {checkpoint}")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(REPO_ROOT / "src"), env.get("PYTHONPATH"),
        )))
        for command in commands:
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    except ValueError as error:
        cli.error(str(error))
    except subprocess.CalledProcessError as error:
        return error.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
