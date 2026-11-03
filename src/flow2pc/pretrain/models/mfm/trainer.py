"""Delegate MFM pretraining to its original geopath and flow trainer."""

import os
from pathlib import Path
import subprocess
import sys

import yaml

from flow2pc.utils.paths import MFM_PRETRAIN_ROOT, REPO_ROOT


DATA_TYPES = {
    "arch": ("arch", None),
    "sphere": ("sphere", None),
    "cite": ("scrna", "op_cite_inputs_0.h5ad"),
    "multi": ("scrna", "op_train_multi_targets_0.h5ad"),
    "eb": ("scrna", "eb_velocity_v5.npz"),
    "lidar": ("lidar", "rainier2-thin.las"),
    "afhq": ("image", "afhq"),
}


def _path(value, label, base=REPO_ROOT):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty path")
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _mapping(value, label, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    unknown, missing = set(value) - set(allowed), set(required) - set(value)
    if unknown or missing:
        raise ValueError(f"{label}: unknown {sorted(unknown)}, missing {sorted(missing)}")
    return value


def prepare(config):
    """Resolve a native MFM command without importing the trainer or writing data."""
    if config.get("model", {}).get("params", {}).get("backend") == "snapshots":
        from flow2pc.pretrain.models.mfm.snapshots import prepare as prepare_snapshots

        return prepare_snapshots(config)
    _mapping(config, "MFM config", {"stage", "model", "data", "training", "seed", "output"},
             {"stage", "model", "data", "training", "output"})
    model = _mapping(config["model"], "model", {"name", "params"}, {"name"})
    if model["name"] != "mfm":
        raise ValueError("Native MFM pretraining requires model.name: mfm")
    params = _mapping(model.get("params", {}), "model.params", {"source_root", "config_path"}, {"config_path"})
    data = _mapping(config["data"], "data", {"name", "root", "params"}, {"name", "root"})
    _mapping(data.get("params", {}), "data.params", set())
    _mapping(config["training"], "training", set())
    output = _mapping(config["output"], "output", {"dir"}, {"dir"})
    seed = config.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if data["name"] not in DATA_TYPES:
        raise ValueError(f"Unsupported MFM dataset: {data['name']}")
    source = _path(params.get("source_root", str(MFM_PRETRAIN_ROOT)), "model.params.source_root")
    if not (source / "mfm/train/main.py").is_file():
        raise FileNotFoundError(f"MFM trainer is missing under {source}")
    config_name = params["config_path"]
    native_config = _path(config_name, "model.params.config_path", source)
    if not native_config.is_file() and not Path(config_name).is_absolute():
        native_config = _path(config_name, "model.params.config_path")
    if not native_config.is_file():
        raise FileNotFoundError(f"MFM native config is missing: {native_config}")
    native = yaml.safe_load(native_config.read_text())
    if not isinstance(native, dict):
        raise ValueError("MFM native config must be a mapping")
    data_type, filename = DATA_TYPES[data["name"]]
    if native.get("mfm") is not True or native.get("data_type") != data_type:
        raise ValueError("MFM native config must enable mfm and match data_type")
    if data_type in ("scrna", "lidar", "image") and native.get("data_name") != data["name"]:
        raise ValueError("MFM native data_name does not match data.name")
    if "working_dir" in native or "data_path" in native:
        raise ValueError("Set MFM working_dir via output.dir and data via data.root")
    if native.get("seeds") != [seed]:
        raise ValueError("MFM native config must declare seeds equal to [seed]")
    data_root = _path(data["root"], "data.root")
    if not data_root.is_dir():
        raise FileNotFoundError(f"MFM data.root is missing: {data_root}")
    if filename and not (data_root / filename).exists():
        raise FileNotFoundError(f"MFM input is missing: {data_root / filename}")
    output_dir = _path(output["dir"], "output.dir")
    return {
        "method": "mfm", "source_root": str(source), "data_root": str(data_root),
        "native_config": str(native_config), "output_dir": str(output_dir),
        "input_name": filename,
        "command": [sys.executable, "-m", "mfm.train.main", "--config_path",
                    str(native_config), "--working_dir", str(output_dir)],
    }


def _prepare_output_data(resolved):
    """Expose only this run's input, preserving source data and existing files."""
    output = Path(resolved["output_dir"])
    data_root = Path(resolved["data_root"])
    filename = resolved["input_name"]
    source = data_root / filename if filename else None
    if source is not None and not source.exists():
        raise FileNotFoundError(f"MFM input is missing: {source}")
    data_dir = output / "data"
    output.mkdir(parents=True, exist_ok=True)
    if data_dir.is_symlink():
        if data_dir.resolve() != data_root:
            raise ValueError(f"{data_dir} differs from configured data.root")
        # Unlink the legacy directory alias, never any of its children.
        data_dir.unlink()
    elif data_dir.exists() and not data_dir.is_dir():
        raise ValueError(f"{data_dir} must be a directory")
    if source is None:
        return output
    if data_dir.resolve() == data_root:
        raise ValueError(f"{data_dir} must be separate from configured data.root")
    data_dir.mkdir(exist_ok=True)
    data_link = data_dir / filename
    if data_link.is_symlink():
        if data_link.resolve() != source.resolve():
            raise ValueError(f"{data_link} differs from configured input")
    elif data_link.exists():
        raise ValueError(f"{data_link} already exists; refusing to overwrite it")
    else:
        data_link.symlink_to(source, target_is_directory=source.is_dir())
    return output


def run(resolved):
    if resolved.get("backend") == "snapshots":
        from flow2pc.pretrain.models.mfm.snapshots import run as run_snapshots

        return run_snapshots(resolved)
    output = _prepare_output_data(resolved)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        resolved["source_root"], env.get("PYTHONPATH"))))
    subprocess.run(resolved["command"], cwd=resolved["source_root"], env=env, check=True)
    return output


class MFM:
    def validate(self, config):
        return prepare(config)

    def fit(self, dataset, config, run_dir):
        return run(dataset if isinstance(dataset, dict) else prepare(config))

    def load(self, checkpoint, device):
        raise ValueError(
            "MFM requires both flow and geopath checkpoints; use the MFM "
            "distillation config with teacher.params.geopath_checkpoint"
        )
