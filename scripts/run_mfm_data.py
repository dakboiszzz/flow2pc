#!/usr/bin/env python3
"""Prepare and run the non-AFHQ MFM catalog. --dry-run never writes or trains."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flow2pc.data.mfm_catalog import (  # noqa: E402
    DATA_ROOT,
    PROFILES,
    link_sources,
    prepared_path,
)
from flow2pc.utils.paths import MFM_PRETRAIN_ROOT  # noqa: E402


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("action", choices=("list", "prepare", "pretrain", "distill", "both"))
    result.add_argument("profile", nargs="?", default="all", choices=(*PROFILES, "all"))
    result.add_argument("--student", choices=("hclt", "gaussian", "both"), default="hclt")
    result.add_argument("--data-root", type=Path, default=DATA_ROOT)
    result.add_argument("--bundle", type=Path, default=ROOT / "data")
    result.add_argument("--source-root", type=Path, default=MFM_PRETRAIN_ROOT)
    result.add_argument("--native-data", type=Path, default=MFM_PRETRAIN_ROOT / "data")
    result.add_argument("--output-root", type=Path, default=ROOT / "outputs/mfm")
    result.add_argument("--run-name", default="default")
    result.add_argument("--teacher-run", type=Path, help="Existing teacher run, for distill of one profile")
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--epochs", type=int, help="Override teacher epochs per phase")
    result.add_argument("--steps", type=int, help="Override student update count")
    result.add_argument("--batch-size", type=int, help="Override requested batch size for both stages")
    result.add_argument("--logging", choices=("disabled", "offline", "online"), default="online")
    result.add_argument("--python", default=os.environ.get("FLOW2PC_PYTHON", sys.executable))
    result.add_argument("--links-only", action="store_true", help="For prepare: only create source symlinks/catalog")
    result.add_argument("--dry-run", action="store_true")
    return result


def commands(args, name):
    output = args.output_root.expanduser().resolve() / name / args.run_name
    teacher = args.teacher_run.expanduser().resolve() if args.teacher_run else output / "pretrain"
    prepared = prepared_path(args.data_root, name, args.seed)
    if args.teacher_run and not args.dry_run:
        pair = json.loads((teacher / "teacher_pair.json").read_text())
        if pair["binding"]["profile"] != name:
            raise ValueError("--teacher-run belongs to a different data profile")
        prepared = Path(pair["binding"]["prepared_path"])
    common = {"seed": args.seed, "wandb.mode": args.logging}
    stages = []
    if args.action in ("pretrain", "both"):
        values = {**common, "data.root": str(prepared), "output.dir": str(output / "pretrain"),
                  "model.params.source_root": str(args.source_root.expanduser().resolve()), "training.device": args.device}
        if args.epochs is not None:
            values["training.epochs"] = args.epochs
        if args.batch_size is not None:
            values["training.batch_size"] = args.batch_size
        stages.append((ROOT / f"configs/pretrain/mfm/mfm_{name}.yaml", values))
    if args.action in ("distill", "both"):
        students = ("hclt", "gaussian") if args.student == "both" else (args.student,)
        for student in students:
            values = {**common, "data.root": str(prepared), "output.dir": str(output / "distill" / student),
                      "teacher.checkpoint": str(teacher / "checkpoints/flow.pt"),
                      "teacher.params.geopath_checkpoint": str(teacher / "checkpoints/geopath.pt"),
                      "teacher.params.repo_path": str(args.source_root.expanduser().resolve()), "distillation.device": args.device}
            if args.steps is not None:
                values["distillation.steps"] = args.steps
            if args.batch_size is not None:
                values["distillation.batch_size"] = args.batch_size
            stages.append((ROOT / f"configs/distill/mfm/mfm_{name}_{student}.yaml", values))
    result = []
    for config, overrides in stages:
        if not config.is_file():
            raise FileNotFoundError(f"Recipe missing: {config}")
        command = [args.python, str(ROOT / "main.py"), "--config", str(config)]
        for key, value in overrides.items():
            command.extend(("--set", f"{key}={json.dumps(value)}"))
        result.append(command)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.action == "list":
        for name, spec in PROFILES.items():
            print(f"{name:25s} {spec['dim']:3d}D  holdouts={spec['holdouts']}")
        return 0
    if Path(args.run_name).name != args.run_name or args.run_name in ("", ".", ".."):
        cli.error("--run-name must be a single directory name")
    if args.teacher_run and (args.action != "distill" or args.profile == "all"):
        cli.error("--teacher-run requires distill and one profile")
    if args.links_only and args.action != "prepare":
        cli.error("--links-only applies only to prepare")
    if not 0 <= args.seed < 2**32 or any(x is not None and x < 1 for x in (args.epochs, args.steps, args.batch_size)):
        cli.error("Invalid seed or training count")
    names = list(PROFILES) if args.profile == "all" else [args.profile]
    try:
        plans = [(name, commands(args, name)) for name in names]
        if args.action in ("prepare", "both"):
            print(f"Prepare source links under {args.data_root / 'sources'}", flush=True)
            if not args.dry_run:
                link_sources(args.data_root, bundle=args.bundle, native_data=args.native_data)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(ROOT / "src"), env.get("PYTHONPATH"))))
        for name, jobs in plans:
            if args.action in ("prepare", "both") and not args.links_only:
                print(f"Prepare {name}: {prepared_path(args.data_root, name, args.seed)}", flush=True)
                if not args.dry_run:
                    # Use the requested training interpreter for optional data dependencies too.
                    subprocess.run([
                        args.python, "-m", "flow2pc.data.prepare_mfm", name,
                        "--data-root", str(args.data_root.expanduser().resolve()), "--seed", str(args.seed),
                        "--source-root", str(args.source_root.expanduser().resolve()),
                    ], env=env, cwd=ROOT, check=True)
            for command in jobs:
                print(shlex.join(command), flush=True)
                if not args.dry_run:
                    subprocess.run(command, cwd=ROOT, env=env, check=True)
    except subprocess.CalledProcessError as error:
        return error.returncode
    except (ValueError, OSError) as error:
        cli.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
