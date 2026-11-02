#!/usr/bin/env python3
"""Run a supplied ../otp-fm reproduction control; imports no Flow2PC model code."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-root", type=Path, default=Path(__file__).resolve().parents[2] / "otp-fm")
    parser.add_argument("--dataset", choices=("gom", "beijing"), required=True)
    parser.add_argument("--config", type=Path, required=True, help="Complete effective native JSON config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root, output = args.native_root.resolve(), args.output.resolve()
    config = json.loads(args.config.read_text())
    config["seed"] = args.seed
    data_root = Path(config["data_dir"])
    required = (data_root / "GoMvortex_data.npy" if args.dataset == "gom" else data_root / "dingling_pm25_13marginals.npy")
    if not required.is_file():
        raise FileNotFoundError(f"Native preprocessed data missing: {required}; automatic download is not allowed")
    if not (root / "experiments/reproduce.py").is_file():
        raise FileNotFoundError("Native OTP-FM reproduction runner is missing")
    if output.exists():
        raise FileExistsError(f"Preserving existing native control: {output}")
    print(json.dumps({"dataset": args.dataset, "recipe": args.recipe, "seed": args.seed,
                      "native_root": str(root), "output": str(output), "dry_run": args.dry_run}, indent=2), flush=True)
    if args.dry_run:
        return 0
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root), str(root / "src"), str(Path(__file__).resolve().parents[1] / "src")]
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    os.chdir(root)
    import torch
    import wandb
    from experiments import reproduce
    from flow2pc.utils.wandb_io import native_wandb_io

    # Keep the original trainer/loop and isolate only its shared CSV output,
    # exactly as in the migrated pipeline. Native source files are not edited.
    setup_name = "setup_gulfofmexico" if args.dataset == "gom" else "setup_beijingair"
    setup_original = getattr(reproduce.original, setup_name)
    def setup_with_local_csv(*setup_args, **setup_kwargs):
        setup = setup_original(*setup_args, **setup_kwargs)
        cls = setup["trainer_class"]
        append_original = cls.append_to_master_csv
        cls.append_to_master_csv = lambda self, master_csv_path=None: append_original(self, self.save_dir / "native_master_results.csv")
        return setup
    setattr(reproduce.original, setup_name, setup_with_local_csv)

    torch.set_num_threads(args.threads)
    namespace = argparse.Namespace(skip_completed=False, restart_incomplete=False,
                                   device=args.device, threads=args.threads, project="flow2pc",
                                   entity=None, group="flow2pc-otpfm-native-parity", wandb_mode=args.mode,
                                   epochs=None, smoke=False, extra_source_files=())
    provenance = {"source": reproduce.source_metadata(), "data_file": str(required),
                  "data_sha256": hashlib.sha256(required.read_bytes()).hexdigest(),
                  "control": "Original native reproduce.run_one; logging-only W&B guards and per-run CSV"}
    def tracking_mode_changed(mode):
        # Native run_one checks args.wandb_mode after SDK startup. Report the
        # actual fallback mode so its strict online check does not abort training.
        namespace.wandb_mode = mode
        path = output / "execution.json"
        execution = json.loads(path.read_text())
        execution.update(requested_wandb_mode=args.mode, wandb_mode=mode)
        path.write_text(json.dumps(execution, indent=2) + "\n")

    with native_wandb_io(wandb, output, args.mode, mode_changed=tracking_mode_changed):
        reproduce.run_one(namespace, args.dataset, None, args.recipe, args.seed, config, output, provenance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
