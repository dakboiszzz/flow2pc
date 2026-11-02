#!/usr/bin/env python3
"""Run the supplied ALI-CFM driver, seed 42 / first fold, without Flow2PC imports.

Single-cell uses the actual upstream working checkout. ST uses its matching
historical implementation in an isolated source snapshot. Nothing executes on
import; --dry-run inspects paths/source only and creates no output directories.
"""

import argparse
import ast
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import types


ST_REVISION = "501e9db6a3bcb1b96eb893882b62a56f78fe1823"
CONFIGS = {
    f"{dataset}{dim}": f"configs/single_cell/{directory}/{dataset}-{dim}D.yaml"
    for dataset in ("cite", "multi")
    for dim, directory in ((5, "5D"), (50, "50D"), (100, "100d"))
}
CONFIGS["st"] = "configs/st/st_cfg.yaml"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


class FirstFold(ast.NodeTransformer):
    """Select t=1 in the driver and its result tables; preserve all time maps."""

    def __init__(self):
        self.replacements = 0
        self.selector = ast.dump(ast.parse("timesteps_list[1:-1]", mode="eval").body)

    def visit_Subscript(self, node):
        if ast.dump(node) == self.selector:
            self.replacements += 1
            return ast.copy_location(ast.List(elts=[ast.Constant(1)], ctx=ast.Load()), node)
        return self.generic_visit(node)


def snapshot_st(root, target):
    archive = git(root, "archive", ST_REVISION, "ali_cfm", CONFIGS["st"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle.getmembers():
            path = target / member.name
            if not path.resolve().is_relative_to(target.resolve()):
                raise ValueError(f"Unexpected archive path: {member.name}")
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source:
                    path.write_bytes(source.read())
            else:
                raise ValueError(f"Unsupported archive member: {member.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=CONFIGS, required=True)
    parser.add_argument("--native-root", type=Path, default=Path(__file__).resolve().parents[2] / "ali-cfm")
    parser.add_argument("--data-root", type=Path, default=Path("/data/lthung/ali-cfm-data"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/lthung/flow2pc/alicfm-native-controls"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    native_root, data_root = args.native_root.resolve(), args.data_root.resolve()
    output = args.output_root.resolve() / args.dataset / "seed42-t1"
    is_st = args.dataset == "st"
    driver = f"ali_cfm/scripts/train_ali_{'st' if is_st else 'single_cell'}.py"
    source = (git(native_root, "show", f"{ST_REVISION}:{driver}").decode()
              if is_st else (native_root / driver).read_text())
    tree = ast.parse(source, filename=driver)
    selector = FirstFold()
    tree = ast.fix_missing_locations(selector.visit(tree))
    expected = 2 if is_st else 3
    if selector.replacements != expected:
        raise ValueError(f"Native driver changed: expected {expected} fold selectors, found {selector.replacements}")
    config_path = CONFIGS[args.dataset]
    if is_st:
        git(native_root, "show", f"{ST_REVISION}:{config_path}")
        required = [data_root / "spatial_transcriptomics/aligned/aligned_spots" / f"U{i}_tumor_coordinates.csv" for i in range(2, 6)]
        required += [data_root / "spatial_transcriptomics/aligned/ref_U5_warped_images" / f"U{i}_aligned_to_U5.tif" for i in range(2, 6)]
    else:
        if not (native_root / config_path).is_file():
            raise FileNotFoundError(native_root / config_path)
        filename = "op_cite_inputs_0.h5ad" if args.dataset.startswith("cite") else "op_train_multi_targets_0.h5ad"
        required = [data_root / "single_cell" / filename]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing data: {missing}")
    if output.exists():
        raise FileExistsError(f"Preserving existing output: {output}; choose a new --output-root")
    plan = {
        "dataset": args.dataset, "native_root": str(native_root),
        "reference": ST_REVISION if is_st else "supplied_working_checkout",
        "head": git(native_root, "rev-parse", "HEAD").decode().strip(),
        "driver": driver, "driver_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "config": config_path, "seed": 42, "heldout_index": 1,
        "data_root": str(data_root), "output": str(output), "wandb_project": "flow2pc",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "adaptations": ["Select first fold in driver and result tables", "seed_list=[42]", "data/output/W&B paths and names"],
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return

    output.mkdir(parents=True, exist_ok=False)
    root = native_root
    if is_st:
        root = output / "native_source"
        snapshot_st(native_root, root)
    plan["source_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "ali_cfm").rglob("*.py"))
    }
    plan["config_sha256"] = hashlib.sha256((root / config_path).read_bytes()).hexdigest()
    os.environ["WANDB_DIR"] = str(output)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["WANDB_PROJECT"] = "flow2pc"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    os.chdir(root)
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(root / config_path)
    overrides = {
        "seed_list": [42], "device": "cuda:0", "wandb_project": "flow2pc",
        "wandb_name": f"native-control-{args.dataset}-seed42-t1", "wandb_mode": "online",
    }
    if is_st:
        overrides.update(
            st_data_dir=str(data_root / "spatial_transcriptomics/aligned/aligned_spots"),
            st_image_dir=str(data_root / "spatial_transcriptomics/aligned/ref_U5_warped_images"),
        )
    else:
        overrides.update(output_dir=str(output), wandb_group="native-migration-controls")
    cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    OmegaConf.save(cfg, output / "native_config.yaml")
    (output / "control_manifest.json").write_text(json.dumps(plan, indent=2) + "\n")
    adapted = ast.unparse(tree) + "\n"
    driver_path = output / "native_driver_first_fold.py"
    driver_path.write_text(adapted)
    data_utils = importlib.import_module("ali_cfm.data_utils")
    # Only redirect the native loader's path constant; keep its implementation.
    if not is_st:
        data_utils.DATA_PATH = data_root / "single_cell"
    module = types.ModuleType("native_control_driver")
    module.__file__, module.__package__ = str(driver_path), "ali_cfm.scripts"
    sys.modules[module.__name__] = module
    exec(compile(adapted, str(driver_path), "exec"), module.__dict__)
    module.train_ali(cfg)


if __name__ == "__main__":
    main()
