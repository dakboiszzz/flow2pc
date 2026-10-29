"""Shared execution and discovery; each method owns its optimization stages."""

import json
from pathlib import Path

from flow2pc.data.registry import datasets
from flow2pc.pretrain.registry import teachers


def prepare(config):
    data = config["data"]
    method = teachers.get(config["model"]["name"])()
    method.validate(config)
    dataset = datasets.get(data["name"])(
        name=data["name"],
        root=Path(data["root"]),
        seed=config["seed"],
        params=data["params"],
    )
    return method, dataset


def run(config):
    method, dataset = prepare(config)
    return method.fit(dataset, config, Path(config["output"]["dir"]))


def load_teacher(checkpoint, device="cpu"):
    manifest_path = Path(checkpoint).resolve().parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return teachers.get(manifest["method"])().load(checkpoint, device)
