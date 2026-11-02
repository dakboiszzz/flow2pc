"""MFM marginals, preprocessing, and deterministic distillation splits."""

from __future__ import annotations

import random
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from flow2pc.data.metric_base import DatasetBundle
from flow2pc.utils.mfm_source import (
    fingerprint,
    sample_shape,
    source_module,
    validate_dataset,
)

from flow2pc.utils.paths import MFM_ROOT

DEFAULT_DATA = MFM_ROOT / "data"
FILES = {
    "cite": "op_cite_inputs_0.h5ad",
    "multi": "op_train_multi_targets_0.h5ad",
    "eb": "eb_velocity_v5.npz",
    "lidar": "rainier2-thin.las",
}


@dataclass
class MetricDatasetBundle(DatasetBundle):
    marginals: list = field(default_factory=list)
    train_marginals: list = field(default_factory=list)
    validation_marginals: list = field(default_factory=list)
    heldout_indices: list = field(default_factory=list)
    normalization: dict = field(default_factory=dict)
    split_indices: dict = field(default_factory=dict)

    @property
    def observed_indices(self):
        return [i for i in range(len(self.marginals)) if i not in self.heldout_indices]


@contextmanager
def seeded(seed):
    """Replay source generators without altering the caller's RNG streams."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    with torch.random.fork_rng(devices=[]):
        np.random.seed(seed)
        random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def _groups(values, labels, order=None):
    order = np.unique(labels) if order is None else order
    return [
        torch.as_tensor(np.asarray(values[labels == label]), dtype=torch.float32)
        for label in order
    ], np.asarray(order).tolist()


def _path(config, name):
    root = Path(config.get("data_dir") or DEFAULT_DATA)
    if root.is_file():
        return root
    return root / FILES[name]


def load_temporal(name, config, metadata):
    args = metadata["args"]
    normalization = {}
    if config.get("snapshots_path"):
        # Portable user-provided snapshots; values are in the teacher's input coordinates.
        path = Path(config["snapshots_path"])
        with np.load(path, allow_pickle=False) as archive:
            groups, labels = _groups(archive["values"], archive["labels"])
        return groups, labels, normalization, [path]
    if name in ("arch", "sphere"):
        with seeded(int(args.get("seed_current", 42))):
            module = source_module(
                "dataloaders.trajectory_data", metadata.get("repo_path")
            )
            values, labels, order = getattr(module, f"generate_{name}_data")()
        files = []
    else:
        path = _path(config, name)
        if not path.is_file():
            raise FileNotFoundError(f"MFM {name} data not found: {path}")
        files = [path]
        dim = int(args["dim"])
        if name == "eb":
            with np.load(path, allow_pickle=False) as archive:
                values, labels = archive["pcs"][:, :dim], archive["sample_labels"]
            order = np.unique(labels)
        else:
            import anndata

            archive = anndata.read_h5ad(path, backed="r")
            try:
                values = np.asarray(archive.obsm["X_pca"][:, :dim])
                category = archive.obs["day"].astype("category")
                labels, order = category.to_numpy(), category.cat.categories.to_numpy()
            finally:
                archive.file.close()
    if values.shape[1] != int(args["dim"]):
        raise ValueError(
            f"{name} has {values.shape[1]} PCA columns, teacher needs {args['dim']}"
        )
    if args.get("whiten", False):
        from sklearn.preprocessing import StandardScaler

        # Upstream fits this transform on all frames, including temporal holdouts.
        # Preserve its coordinates rather than fitting a different transform.
        scaler = StandardScaler()
        values = scaler.fit_transform(values)
        normalization = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "fit_scope": "all_source_frames_as_in_upstream",
        }
    groups, labels = _groups(values, labels, order)
    return groups, labels, normalization, files


def load_lidar(config, metadata):
    if config.get("snapshots_path"):
        return load_temporal("lidar", config, metadata)
    path = _path(config, "lidar")
    if not path.is_file():
        raise FileNotFoundError(f"MFM LiDAR scene not found: {path}")
    args = dict(metadata["args"])
    args["data_path"] = str(path)
    with seeded(int(args.get("seed_current", 42))):
        cls = source_module(
            "dataloaders.lidar_data", metadata.get("repo_path")
        ).LidarDataModule
        data = cls(SimpleNamespace(**args))
    groups = [
        torch.cat([train.dataset, valid.dataset]).float()
        for train, valid in zip(
            data.train_dataloaders, data.val_dataloaders, strict=True
        )
    ]
    norm = {}
    if args.get("whiten", False):
        norm = {
            "mean": data.scaler.mean_.tolist(),
            "scale": data.scaler.scale_.tolist(),
            "fit_scope": "full_ground_scene_as_in_upstream",
        }
    return groups, [0, 1], norm, [path]


def load_afhq(config, metadata):
    args = metadata["args"]
    root = Path(config.get("data_dir") or DEFAULT_DATA / "afhq")
    if (
        not (root / "afhq_train_latent_dataset_128.pt").is_file()
        and (root / "afhq").is_dir()
    ):
        root = root / "afhq"
    files = [root / f"afhq_{split}_latent_dataset_128.pt" for split in ("train", "val")]
    labels = {"cat": 0, "dog": 1, "wild": 2}
    selected = [args.get("x0_label", "dog"), args.get("x1_label", "cat")]
    if len(set(selected)) != 2 or any(label not in labels for label in selected):
        raise ValueError(
            "AFHQ source and target labels must be two distinct supported classes"
        )
    groups = []
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"MFM AFHQ latent cache not found: {path}")
        archive = torch.load(path, map_location="cpu", weights_only=True)
        values, classes = archive["mean"], archive["label"]
        if tuple(values.shape[1:]) != (4, 16, 16) or len(classes) != len(values):
            raise ValueError(
                "AFHQ requires latent means [N,4,16,16] with one class per row"
            )
        groups.append([values[classes == labels[label]].float() for label in selected])
    return groups[0], groups[1], selected, files


def load(name, config, context):
    metadata = context["teacher_metadata"]
    args = metadata["args"]
    validate_dataset(args, name)
    shape = sample_shape(args)
    seed = int(config.get("seed", 42))
    generator = torch.Generator().manual_seed(seed)
    heldout = args.get("t_exclude_current")
    heldout = [] if heldout is None or heldout == -1 else [int(heldout)]
    if name == "afhq":
        training, validation, labels, files = load_afhq(config, metadata)
        marginals = [
            torch.cat([a, b]) for a, b in zip(training, validation, strict=True)
        ]
        normalization = {
            "latent_statistic": "mean",
            "latent_scale": 1.0,
            "vae_id": "CompVis/stable-diffusion-v1-4",
        }
        splits = {
            str(i): {
                "train": list(range(len(a))),
                "validation": list(range(len(a), len(a) + len(b))),
            }
            for i, (a, b) in enumerate(zip(training, validation, strict=True))
        }
    else:
        marginals, labels, normalization, files = (
            load_lidar(config, metadata)
            if name == "lidar"
            else load_temporal(name, config, metadata)
        )
        training, validation, splits = [], [], {}
        ratio = float(config.get("train_fraction", 0.9))
        if not 0 < ratio < 1:
            raise ValueError("train_fraction must be between zero and one")
        for i, values in enumerate(marginals):
            if len(values) < 2:
                raise ValueError("Each MFM marginal needs at least two samples")
            indices = torch.randperm(len(values), generator=generator)
            cut = max(1, min(len(values) - 1, int(len(values) * ratio)))
            train_idx = indices[:cut] if i not in heldout else indices[:0]
            val_idx = indices[cut:] if i not in heldout else indices
            training.append(values[train_idx])
            validation.append(values[val_idx])
            splits[str(i)] = {
                "train": train_idx.tolist(),
                "validation": val_idx.tolist(),
            }
    if len(marginals) < 2 or any(i <= 0 or i >= len(marginals) - 1 for i in heldout):
        raise ValueError(
            "Temporal holdouts must be interior indices of a grid with at least two frames"
        )
    for values in marginals:
        if tuple(values.shape[1:]) != shape or not torch.isfinite(values).all():
            raise ValueError(f"MFM {name} requires finite samples with shape {shape}")
    observed = [i for i in range(len(marginals)) if i not in heldout]
    flat = torch.cat([training[i].flatten(1) for i in observed])
    indices = torch.cat([torch.full((len(training[i]),), i) for i in observed])
    loader = DataLoader(
        TensorDataset(flat, indices),
        batch_size=config.get("batch_size", 128),
        shuffle=True,
        generator=generator,
    )
    info = {
        "dataset": name,
        "labels": labels,
        "physical_times": (
            [float(value) for value in labels]
            if all(isinstance(value, Real) and math.isfinite(value) for value in labels)
            else None
        ),
        "model_times": torch.linspace(0, 1, len(marginals)).tolist(),
        "time_transform": "ordinal_snapshot_index_to_unit_interval",
        "shape": list(shape),
        "marginal_sizes": [len(x) for x in marginals],
        "split_seed": seed,
        "split_protocol": "new_deterministic_distillation_split",
        "heldout_indices": heldout,
        "files": [
            {"path": str(path.resolve()), "sha256": fingerprint(path)} for path in files
        ],
        "snapshots_sha256": _tensor_fingerprint(marginals),
    }
    return MetricDatasetBundle(
        loader,
        shape,
        "metric",
        training=training,
        validation=validation,
        times=torch.linspace(0, 1, len(marginals)),
        source_info=info,
        marginals=marginals,
        train_marginals=training,
        validation_marginals=validation,
        heldout_indices=heldout,
        normalization=normalization,
        split_indices=splits,
    )


def _tensor_fingerprint(values):
    import hashlib

    digest = hashlib.sha256()
    for tensor in values:
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()
