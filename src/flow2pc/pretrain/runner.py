"""Shared execution and discovery; each method owns its optimization stages."""

import json
from pathlib import Path

from flow2pc.data.registry import datasets
from flow2pc.pretrain.registry import teachers


def prepare(config):
    if config["model"]["name"] == "mfm":
        from flow2pc.pretrain.models.mfm.trainer import prepare as prepare_mfm

        return teachers.get("mfm")(), prepare_mfm(config)
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


def load_teacher(
    checkpoint, device="cpu", *, legacy_metadata=None, generator_checkpoint=None
):
    if legacy_metadata is not None:
        from flow2pc.pretrain.models.alicfm.legacy import load

        return load(checkpoint, legacy_metadata, device, generator_checkpoint)
    manifest_path = Path(checkpoint).resolve().parent.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return teachers.get(manifest["method"])().load(checkpoint, device)
