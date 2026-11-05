"""Native MMFM model construction and frozen checkpoint velocity interface."""

from dataclasses import dataclass
from pathlib import Path

import torch

from flow2pc.data.metric_profiles import dataset_identity
from flow2pc.data.metric_registry import load_dataset
from flow2pc.utils.mfm_source import mmfm_provenance as provenance
from flow2pc.utils.mfm_source import source_module
from flow2pc.utils.checkpoint import sha256


def build_model(config, source_root):
    if config.get("conditional_model") is not False:
        raise ValueError("Only explicitly unconditional MMFM models are supported")
    return source_module("models", source_root, method="mmfm").VectorFieldModel(**config)


def model_velocity(model, values, times):
    if values.ndim != 2:
        raise ValueError("MMFM velocity expects [batch, dim] values")
    times = torch.as_tensor(times, dtype=values.dtype, device=values.device)
    if times.numel() == 1:
        times = times.reshape(1).expand(len(values))
    if times.numel() != len(values):
        raise ValueError("MMFM needs one time per sample or a scalar time")
    return model(torch.cat([values, times.reshape(-1, 1)], dim=1))


@dataclass
class LoadedMMFM:
    model: object
    metadata: dict
    source_root: str

    def velocity(self, values, time, context=None):
        if context is not None:
            raise ValueError("Unconditional MMFM does not accept context")
        return model_velocity(self.model, values, time)


def load_teacher(options, device="cpu"):
    path = Path(options["checkpoint_path"])
    if not path.is_file():
        raise FileNotFoundError(f"MMFM teacher checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "mmfm_teacher_v1":
        raise ValueError("Expected a Flow2PC unconditional MMFM teacher checkpoint")
    metadata = dict(checkpoint["metadata"])
    if options.get("dataset", metadata["dataset_config"]["name"]) != metadata["dataset_config"]["name"]:
        raise ValueError("MMFM checkpoint dataset differs from requested dataset")
    source_root = str(options.get("source_root", metadata["source_root"]))
    if provenance(source_root)["sha256"] != metadata["source_provenance"]["sha256"]:
        raise ValueError("MMFM source files differ from checkpoint provenance")
    model = build_model(metadata["model_config"], source_root).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval().requires_grad_(False)
    metadata["flow_sha256"] = sha256(path)
    return LoadedMMFM(model, metadata, source_root)


def load_teacher_data(teacher, dataset_config=None):
    saved = teacher.metadata["dataset_config"]
    requested = dataset_config or {}
    if requested.get("name", saved["name"]) != saved["name"]:
        raise ValueError("MMFM dataset name differs from checkpoint")
    options = dict(saved)
    overrides = requested.get("options", {})
    if overrides:
        raise ValueError("MMFM distill restores data options from the teacher checkpoint")
    if requested.get("data_dir") is not None:
        options["data_dir"] = requested["data_dir"]
    data = load_dataset(options, {"dataset_profile": teacher.metadata["dataset_profile"]})
    actual = dataset_identity(data)
    expected = teacher.metadata["dataset_identity"]
    # File locations may change; hashes and all scientific metadata must match.
    from copy import deepcopy
    actual, expected = deepcopy(actual), deepcopy(expected)
    for value in (actual, expected):
        for entry in value["source_info"].get("files", []):
            entry.pop("path", None)
    if actual != expected:
        raise ValueError("MMFM data, preprocessing, time grid or split differs from checkpoint")
    return data
