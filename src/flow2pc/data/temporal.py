"""Shared snapshot datasets, with no teacher dependencies or trajectory pairing."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from flow2pc.data.base import TemporalMetadata
from flow2pc.utils.checkpoint import identity, sha256


class TensorMarginal(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def materialize(self):
        return self.values


class SnapshotDataset:
    """Deterministic marginal splits and train-fitted affine preprocessing.

    Saved preprocessing can be supplied when reloading a trained teacher. No
    source values contribute to a new fit in that case. Rows are unpaired.
    """

    def __init__(self, *, name, root, seed, params, preprocessing=None):
        self.root, self.name, self.seed = Path(root), name, seed
        self.params = dict(params)
        clouds, physical, labels, provenance = self.read_source()
        self.raw = [torch.as_tensor(x, dtype=torch.float32) for x in clouds]
        if not self.raw or self.raw[0].ndim != 2:
            raise ValueError("Expected nonempty [N,D] marginal arrays")
        self.dim = self.raw[0].shape[1]
        if any(
            x.ndim != 2
            or x.shape[1] != self.dim
            or len(x) < 3
            or not torch.isfinite(x).all()
            for x in self.raw
        ):
            raise ValueError(
                "Every marginal must contain finite [N,D] values and at least three rows"
            )
        self.heldout = params.get("heldout_index")
        if self.heldout is not None and (
            type(self.heldout) is not int or not 0 < self.heldout < len(self.raw) - 1
        ):
            raise ValueError("heldout_index must identify an interior observation")
        self.observed = [i for i in range(len(self.raw)) if i != self.heldout]
        val, test = (
            params.get("validation_fraction", 0.05),
            params.get("test_fraction", 0.05),
        )
        if (
            any(
                type(v) not in (int, float) or not np.isfinite(v) or not 0 < v < 1
                for v in (val, test)
            )
            or val + test >= 1
        ):
            raise ValueError(
                "Validation/test fractions must be positive and sum below one"
            )
        rng = np.random.default_rng(seed)
        self.rows = []
        for i, cloud in enumerate(self.raw):
            order = rng.permutation(len(cloud))
            if i == self.heldout:
                row = {"train": order[:0], "val": order[:0], "test": order}
            else:
                nv, nt = (
                    max(1, round(len(cloud) * val)),
                    max(1, round(len(cloud) * test)),
                )
                if nv + nt >= len(cloud):
                    raise ValueError("Split fractions leave no training rows")
                row = {
                    "val": order[:nv],
                    "test": order[nv : nv + nt],
                    "train": order[nv + nt :],
                }
            self.rows.append(row)
        normalise, whiten = params.get("normalise", True), params.get("whiten", False)
        if type(normalise) is not bool or type(whiten) is not bool:
            raise ValueError("normalise and whiten must be booleans")
        if preprocessing is None:
            train = torch.cat(
                [self.raw[i][self.rows[i]["train"]] for i in self.observed]
            ).double()
            shift, scale = (
                torch.zeros(self.dim, dtype=torch.float64),
                torch.ones(self.dim, dtype=torch.float64),
            )
            if whiten:
                shift, scale = (
                    train.mean(0),
                    train.std(0, unbiased=False).clamp_min(1e-12),
                )
            if normalise:
                standardized = (train - shift) / scale
                lo, hi = standardized.amin(0), standardized.amax(0)
                scale_range = (hi - lo).clamp_min(1e-12)
                shift, scale = shift + scale * lo, scale * scale_range
            preprocessing = {
                "kind": "affine",
                "fit_split": "train",
                "fit_time_indices": self.observed,
                "normalise": normalise,
                "whiten": whiten,
                "shift": shift.float().tolist(),
                "scale": scale.float().tolist(),
                "feature_key": provenance["feature_key"],
                "pca_recomputed": False,
            }
        self.preprocessing = dict(preprocessing)
        if (
            preprocessing.get("feature_key", provenance["feature_key"])
            != provenance["feature_key"]
        ):
            raise ValueError(
                "Saved preprocessing belongs to a different coordinate representation"
            )
        self.shift = torch.tensor(preprocessing["shift"], dtype=torch.float32)
        self.scale = torch.tensor(preprocessing["scale"], dtype=torch.float32)
        if (
            self.shift.shape != (self.dim,)
            or self.scale.shape != (self.dim,)
            or not torch.isfinite(self.shift).all()
            or not torch.isfinite(self.scale).all()
            or (self.scale <= 0).any()
        ):
            raise ValueError("Invalid saved affine preprocessing")
        self.clouds = [(x - self.shift) / self.scale for x in self.raw]
        self.source = provenance
        self.labels = labels
        self.metadata = TemporalMetadata(
            name=name,
            dim=self.dim,
            physical_times=tuple(float(x) for x in physical),
            model_times=tuple(i / (len(self.raw) - 1) for i in range(len(self.raw))),
            preprocessing_id=identity(self.preprocessing),
            source_version=identity(provenance),
            split_seed=seed,
        )

    def read_source(self):
        raise NotImplementedError

    def get_marginal(self, time_index, split):
        if type(time_index) is not int or time_index not in range(len(self.clouds)):
            raise ValueError("Marginal index outside the recorded time grid")
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown split")
        return TensorMarginal(self.clouds[time_index][self.rows[time_index][split]])

    def contract(self):
        return {
            **self.metadata.to_dict(),
            "root": str(self.root.resolve()),
            "params": self.params,
            "source": self.source,
            "labels": self.labels,
            "heldout_index": self.heldout,
            "observed_indices": self.observed,
            "cells_per_time": [len(x) for x in self.raw],
            "split_scheme": "independent_rows_within_marginals; heldout_test_only",
            "split_counts": [{k: len(v) for k, v in r.items()} for r in self.rows],
            "split_indices_sha256": identity(
                [{k: v.tolist() for k, v in r.items()} for r in self.rows]
            ),
            "initial_law": {
                "kind": "empirical_earliest_marginal",
                "label": self.labels[0],
            },
        }


class SpatialTranscriptomics(SnapshotDataset):
    def read_source(self):
        allowed = {
            "heldout_index",
            "validation_fraction",
            "test_fraction",
            "normalise",
            "whiten",
        }
        if set(self.params) - allowed:
            raise ValueError(f"Unknown ST params: {set(self.params) - allowed}")
        directory = self.root / "spatial_transcriptomics/aligned/aligned_spots"
        files = [directory / f"U{i}_tumor_coordinates.csv" for i in range(2, 6)]
        clouds = [np.loadtxt(p, delimiter=",", ndmin=2) for p in files]
        if any(cloud.shape[1] != 2 for cloud in clouds):
            raise ValueError(
                "ST tumor coordinate files must have exactly two columns in (y,x) order"
            )
        return (
            clouds,
            [2, 3, 4, 5],
            [f"U{i}" for i in range(2, 6)],
            {
                "files": {str(p.relative_to(self.root)): sha256(p) for p in files},
                "feature_key": "aligned_tumor_y_x",
                "coordinate_order": ["y", "x"],
                "physical_time_units": "ordered_slide_indices; not measured elapsed time",
            },
        )


class CellTracking(SnapshotDataset):
    def read_source(self):
        allowed = {
            "heldout_index",
            "validation_fraction",
            "test_fraction",
            "normalise",
            "whiten",
        }
        if set(self.params) - allowed:
            raise ValueError(
                f"Unknown cell-tracking params: {set(self.params) - allowed}"
            )
        path = self.root / "cell_tracking/prepared/mask_cell4_stack.npy"
        masks = np.load(path, mmap_mode="r", allow_pickle=False)
        if masks.ndim != 3 or len(masks) != 115:
            raise ValueError("Expected the 115-frame cell-4 binary mask export")
        clouds = [np.argwhere(mask)[:, [1, 0]].copy() for mask in masks]
        return (
            clouds,
            list(range(115)),
            [str(i + 1) for i in range(115)],
            {
                "files": {str(path.relative_to(self.root)): sha256(path)},
                "feature_key": "cell4_mask_xy",
                "coordinate_order": ["x", "y"],
                "physical_time_units": "frame_index",
            },
        )


class EmbryoidBody(SnapshotDataset):
    def read_source(self):
        import h5py

        allowed = {
            "heldout_index",
            "validation_fraction",
            "test_fraction",
            "normalise",
            "whiten",
        }
        if set(self.params) - allowed:
            raise ValueError(
                "EB currently preserves the upstream eb-2d PHATE representation; PCA/NPZ is a different protocol"
            )
        path = self.root / "single_cell/ebdata_v3.h5ad"
        with h5py.File(path, "r") as handle:
            values = handle["obsm/X_phate"][:]
            node = handle["obs/sample_labels"]
            codes = node["codes"][:]
            labels = [
                x.decode() if isinstance(x, bytes) else str(x)
                for x in node["categories"][:]
            ]
            if (codes < 0).any() or values.shape[1] != 2:
                raise ValueError(
                    "Invalid EB PHATE representation or missing time annotations"
                )
        # Match the original np.unique label ordering rather than assuming the
        # H5AD category-code order is chronological.
        order = sorted(range(len(labels)), key=lambda i: labels[i])
        clouds = [values[codes == i] for i in order]
        labels = [labels[i] for i in order]
        return (
            clouds,
            list(range(len(labels))),
            labels,
            {
                "files": {str(path.relative_to(self.root)): sha256(path)},
                "feature_key": "obsm/X_phate",
                "physical_time_units": "ordinal_category_index",
                "time_note": "Original labels are day ranges; no exact day/midpoint or trajectory IDs inferred",
            },
        )
