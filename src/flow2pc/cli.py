import argparse
from dataclasses import asdict
import json
import os
import sys

import yaml

from flow2pc.data.registry import datasets
from flow2pc.pretrain.registry import teachers
from flow2pc.utils.config import load_config
from flow2pc.utils.paths import REPO_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description="Flow2PC method-driven framework")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--config")
    action.add_argument("--list-models", action="store_true")
    action.add_argument("--list-datasets", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Resolve config and data metadata without training or writing outputs",
    )
    mode.add_argument(
        "--evaluate",
        choices=("teacher", "student"),
        help="Evaluate existing checkpoints and generate figures without training",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML setting, e.g. --set training.device=cuda:1",
    )
    args = parser.parse_args(argv)
    if args.list_models or args.list_datasets:
        registry = teachers if args.list_models else datasets
        listing = registry.describe()
        if args.list_datasets:
            from flow2pc.data.metric_registry import dataset_names

            for name in dataset_names():
                if name not in listing:
                    listing[name] = {
                        "status": "implemented",
                        "description": "MFM dataset; available with teacher.name: mfm",
                    }
                else:
                    listing[name]["description"] += " Also available for MFM."
        print(json.dumps(listing, indent=2))
        return 0
    try:
        config = load_config(args.config, args.set)
        if args.evaluate:
            if isinstance(config, tuple):
                raise ValueError(
                    "--evaluate uses ALI-CFM/OTP-FM checkpoint configs; "
                    "for MFM/MMFM use a stage: evaluate config instead"
                )
            from flow2pc.evaluation.entrypoint import run as evaluate_checkpoint

            print(json.dumps(evaluate_checkpoint(config, args.evaluate), indent=2))
            return 0
        if isinstance(config, tuple):
            stage, resolved = config
            if args.validate_only:
                print(json.dumps({
                    "stage": stage,
                    "resolved": asdict(resolved) if stage == "distill" else resolved,
                }, indent=2))
                return 0
            previous = os.getcwd()
            try:
                os.chdir(REPO_ROOT)
                if stage == "pretrain":
                    if resolved.get("method") == "mmfm":
                        from flow2pc.pretrain.models.mmfm.trainer import run
                    else:
                        from flow2pc.pretrain.models.mfm.trainer import run
                elif stage == "distill":
                    from flow2pc.distill.runner import run
                else:
                    from flow2pc.evaluation.metric_flow import evaluate_checkpoint

                    if resolved.get("backend") == "mmfm":
                        from flow2pc.evaluation.mmfm import evaluate_checkpoint as evaluate_mmfm

                        print(json.dumps(evaluate_mmfm(
                            resolved["checkpoint"], role=resolved.get("role", "student"),
                            output_dir=resolved.get("output_dir"),
                            device=resolved.get("device", "cpu"), settings=resolved.get("params"),
                        ), indent=2))
                        return 0

                    print(json.dumps(evaluate_checkpoint(
                        resolved["checkpoint"],
                        output_dir=resolved.get("output_dir"),
                        device=resolved.get("device", "cpu"),
                        settings=resolved.get("params"),
                    ), indent=2))
                    return 0
                print(json.dumps({"output_dir": str(run(resolved))}, indent=2))
            finally:
                os.chdir(previous)
            return 0
        if config["stage"] == "distill":
            from flow2pc.distill.runner import prepare, run

            if args.validate_only:
                inspected = dict(
                    config, distillation={**config["distillation"], "device": "cpu"}
                )
                teacher, dataset, sampler = prepare(inspected)
                print(
                    json.dumps(
                        {
                            "config": config,
                            "dataset": dataset.contract(),
                            "sampler": sampler,
                            "teacher_frozen": not any(
                                p.requires_grad for p in teacher.parameters()
                            ),
                        },
                        indent=2,
                    )
                )
                return 0
            print(json.dumps({"checkpoint": str(run(config))}, indent=2))
            return 0
        from flow2pc.pretrain.runner import prepare, run

        if args.validate_only:
            _, dataset = prepare(config)
            print(
                json.dumps({"config": config, "dataset": dataset.contract()}, indent=2)
            )
        else:
            print(json.dumps({"checkpoint": str(run(config))}, indent=2))
        return 0
    except (
        ValueError,
        NotImplementedError,
        OSError,
        ImportError,
        yaml.YAMLError,
    ) as error:
        print(f"flow2pc: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
