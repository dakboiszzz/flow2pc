import argparse
import json
import sys

import yaml

from flow2pc.data.registry import datasets
from flow2pc.pretrain.registry import teachers
from flow2pc.utils.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Flow2PC method-driven framework")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--config")
    action.add_argument("--list-models", action="store_true")
    action.add_argument("--list-datasets", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Resolve config and data metadata without training or writing outputs",
    )
    args = parser.parse_args(argv)
    if args.list_models or args.list_datasets:
        registry = teachers if args.list_models else datasets
        print(json.dumps(registry.describe(), indent=2))
        return 0
    try:
        config = load_config(args.config)
        if config["stage"] == "distill":
            if args.validate_only:
                print(json.dumps(config, indent=2))
                return 0
            raise NotImplementedError(
                "Framework distillation execution remains a separate migration phase. "
                "The HCLT and frozen ALI-CFM interfaces remain available. "
                "This command currently validates the distillation config contract only."
            )
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
