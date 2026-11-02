#!/usr/bin/env python3
"""Recover W&B uploads from saved transport results; never train/evaluate models."""

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from run_alicfm_transport_sweep import ROOT, Run, inspect_run, read_json
from run_alicfm_checks import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/data/lthung/flow2pc/alicfm-transport/sweep-512"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sync", action="store_true", help="Also upload recovered offline W&B histories, serially")
    args = parser.parse_args()
    root = args.output_root.resolve()
    selected = []
    for dataset in ("gom", "beijing"):
        for manifest in sorted((root / dataset).glob("*/manifest.json")):
            output = manifest.parent
            config = yaml.safe_load((output / "resolved_config.yaml").read_text())
            run = Run(dataset, output.name, config, output, {})
            inspect_run(run)
            if run.status == "complete" and run.tracking_status == "pending_sync":
                selected.append(run)
    print(json.dumps({"dry_run": args.dry_run, "sync": args.sync,
                      "recoverable_runs": len(selected), "runs": [r.name for r in selected]}, indent=2), flush=True)
    if args.dry_run or not selected:
        return 0
    with (root / ".sweep.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Stop the transport controller before recovering uploads")
        sys.path.insert(0, str(ROOT / "src"))
        import wandb

        from flow2pc.evaluation.runner import publish_evaluation
        from flow2pc.pretrain.models.alicfm.trainer import LocalLogger

        failed = 0
        for item in selected:
            try:
                previous_tracking = read_json(item.output / "tracking_status.json")
                config = copy.deepcopy(item.config)
                config["logging"]["mode"] = "offline"
                logger = LocalLogger(item.output, config, append=True)
                try:
                    if logger.run is None:
                        raise RuntimeError("Cannot prepare offline W&B history; originals are preserved")
                    offline_dir = Path(logger.run.dir).parent
                    # Explicit steps reproduce the original transport logger's
                    # scalar sequence; do not append duplicate local JSON rows.
                    with (item.output / "metrics.jsonl").open() as handle:
                        for step, line in enumerate(handle):
                            logger.run.log(json.loads(line), step=step)
                    result = read_json(item.output / "eval/metrics.json")
                    scalars = {k: v for k, v in result.items() if k.startswith(("teacher_", "student_"))}
                    media = {k: Path(p) for k, p in read_json(item.output / "eval/media.json").items()}
                    publish_evaluation(logger.run, result["scores"], scalars, media, item.output / "eval")
                    artifact = wandb.Artifact(f"{logger.run.id}-teacher", type="model")
                    for name in ("checkpoints/last.pt", "manifest.json", "preprocessing.json", "resolved_config.yaml"):
                        artifact.add_file(str(item.output / name))
                    logger.run.log_artifact(artifact)
                finally:
                    logger.close()
                # Preserve the previous error in tracking metadata; local results
                # were already checked and need neither fitting nor reevaluation.
                previous = {p: read_json(item.output / p) for p in ("status.json", "eval/status.json")}
                tracking = read_json(item.output / "tracking_status.json")
                tracking.update(status="pending_sync", previous_run_status=previous_tracking.get("previous_run_status", previous),
                                recovery_directory=str(offline_dir))
                atomic_json(item.output / "tracking_status.json", tracking)
                if args.sync:
                    environment = {**os.environ}
                    environment.pop("WANDB_MODE", None)
                    subprocess.run([sys.executable, "-m", "wandb", "sync", "--project", "flow2pc", str(offline_dir)],
                                   env=environment, check=True)
                    tracking["status"] = "synced"
                    atomic_json(item.output / "tracking_status.json", tracking)
                atomic_json(item.output / "status.json", {"status": "complete", "evaluation_status": "complete"})
                atomic_json(item.output / "eval/status.json", {"status": "complete", "binding_sha256": result["binding_sha256"]})
                print(f"Recovered tracking: {item.name}", flush=True)
            except (OSError, ValueError, RuntimeError, wandb.errors.Error, subprocess.SubprocessError) as error:
                failed += 1
                print(f"Tracking still pending for {item.name}: {error}", file=sys.stderr, flush=True)
        return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
