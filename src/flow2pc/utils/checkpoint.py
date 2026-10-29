"""Versioned, hashed, weights-only teacher artifacts independent of distillation."""

import hashlib
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch

from flow2pc.data.base import TemporalMetadata


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_manifest(manifest):
    required = {
        "schema_version",
        "artifact_type",
        "method",
        "method_version",
        "model_config",
        "training_config",
        "dataset",
        "seed",
        "checkpoint_relpath",
        "preprocessing_file",
        "preprocessing_sha256",
        "checkpoint_sha256",
        "source_provenance",
    }
    if not isinstance(manifest, dict) or required - manifest.keys():
        raise ValueError("Teacher manifest is missing required fields")
    if manifest["schema_version"] != 1 or manifest["artifact_type"] != "teacher":
        raise ValueError("Unsupported manifest schema or artifact type")
    data = manifest["dataset"]
    try:
        metadata = TemporalMetadata(
            **{
                key: tuple(data[key])
                if key in ("physical_times", "model_times")
                else data[key]
                for key in TemporalMetadata.__dataclass_fields__
            }
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete temporal dataset metadata") from error
    if data.get("time_transform") != metadata.to_dict()["time_transform"]:
        raise ValueError("Inconsistent dataset time transformation")
    if manifest["model_config"].get("dim") != metadata.dim:
        raise ValueError("Teacher and dataset feature dimensions differ")
    for key in ("checkpoint_relpath", "preprocessing_file"):
        path = Path(manifest[key])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{key} must remain inside the run directory")
    return metadata


def save_teacher(run_dir, payload, manifest, preprocessing):
    run_dir = Path(run_dir)
    atomic_json(run_dir / "preprocessing.json", preprocessing)
    manifest = dict(manifest)
    manifest.update(
        schema_version=1,
        artifact_type="teacher",
        checkpoint_relpath="checkpoints/last.pt",
        preprocessing_file="preprocessing.json",
        preprocessing_sha256=sha256(run_dir / "preprocessing.json"),
    )
    # Bind metadata to weights as well as binding weights to the manifest.
    payload = dict(payload, manifest_identity=identity(manifest))
    checkpoint = run_dir / manifest["checkpoint_relpath"]
    with atomic_path(checkpoint) as temporary:
        torch.save(payload, temporary)
    manifest["checkpoint_sha256"] = sha256(checkpoint)
    validate_manifest(manifest)
    atomic_json(run_dir / "manifest.json", manifest)
    return checkpoint


def load_teacher_artifact(checkpoint, device="cpu"):
    checkpoint = Path(checkpoint).resolve()
    run_dir = checkpoint.parent.parent
    manifest = json.loads((run_dir / "manifest.json").read_text())
    validate_manifest(manifest)
    if checkpoint != (run_dir / manifest["checkpoint_relpath"]).resolve():
        raise ValueError("Checkpoint path differs from its manifest")
    if sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    preprocessing_path = run_dir / manifest["preprocessing_file"]
    if sha256(preprocessing_path) != manifest["preprocessing_sha256"]:
        raise ValueError("Preprocessing checksum mismatch")
    preprocessing = json.loads(preprocessing_path.read_text())
    if identity(preprocessing) != manifest["dataset"]["preprocessing_id"]:
        raise ValueError("Dataset preprocessing identity mismatch")
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    binding = {k: v for k, v in manifest.items() if k != "checkpoint_sha256"}
    if payload.get("manifest_identity") != identity(binding):
        raise ValueError("Checkpoint metadata binding mismatch")
    return payload, manifest
