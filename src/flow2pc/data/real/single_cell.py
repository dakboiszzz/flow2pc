"""Shared CITE/Multi loader; original H5AD PCA is never recomputed or modified."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from flow2pc.data.base import TemporalMetadata
from flow2pc.utils.checkpoint import identity, sha256

FILES = {"cite": "op_cite_inputs_0.h5ad", "multi": "op_train_multi_targets_0.h5ad"}


def read_days(handle):
    node = handle["obs/day"]
    if isinstance(node, h5py.Group):
        codes = node["codes"][:]
        if (codes < 0).any():
            raise ValueError("Missing day labels")
        return node["categories"][:][codes]
    return node[:]


class H5Marginal(Dataset):
    def __init__(self, path, rows, dim, preprocessing):
        self.path, self.rows, self.dim = path, rows, dim
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.rows)

    def transform(self, values):
        values = torch.as_tensor(values, dtype=torch.float32)
        if not torch.isfinite(values).all():
            raise ValueError("Non-finite PCA coordinates")
        p = self.preprocessing
        return (values - torch.tensor(p["shift"])) / torch.tensor(p["scale"])

    def __getitem__(self, index):
        with h5py.File(self.path, "r") as handle:
            values = handle["obsm/X_pca"][int(self.rows[index]), : self.dim]
        return self.transform(values)

    def materialize(self):
        with h5py.File(self.path, "r") as handle:
            values = handle["obsm/X_pca"][np.sort(self.rows), : self.dim]
        # Upstream training samples exchangeable marginals; sorted source rows
        # preserve the split while avoiding HDF5 random-access ordering limits.
        return self.transform(values)


class SingleCell:
    def __init__(self, *, name, root, seed, params, preprocessing=None):
        allowed = {
            "dim",
            "heldout_index",
            "validation_fraction",
            "test_fraction",
            "normalise",
            "whiten",
        }
        if set(params) - allowed:
            raise ValueError(f"Unknown single-cell params: {set(params) - allowed}")
        self.dim = params.get("dim", 50)
        if type(self.dim) is not int or not 1 <= self.dim <= 100:
            raise ValueError("PCA dim must be an integer in [1,100]")
        self.heldout = params.get("heldout_index", 1)
        if self.heldout not in (1, 2) or type(self.heldout) is not int:
            raise ValueError("heldout_index must be 1 or 2")
        val = params.get("validation_fraction", 0.05)
        test = params.get("test_fraction", 0.05)
        if (
            any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not np.isfinite(v)
                or not 0 < v < 1
                for v in (val, test)
            )
            or val + test >= 1
        ):
            raise ValueError(
                "Validation/test fractions must be positive and sum below 1"
            )
        normalise, whiten = params.get("normalise", True), params.get("whiten", False)
        if type(normalise) is not bool or type(whiten) is not bool:
            raise ValueError("normalise and whiten must be booleans")
        self.path = Path(root) / "single_cell" / FILES[name]
        with h5py.File(self.path, "r") as handle:
            features = handle["obsm/X_pca"]
            if features.ndim != 2 or features.shape[1] < self.dim:
                raise ValueError("Source PCA has insufficient dimensions")
            days = read_days(handle)
            if len(days) != features.shape[0] or set(np.unique(days)) != {2, 3, 4, 7}:
                raise ValueError("Expected CITE/Multi day labels [2,3,4,7]")
        self.rows = []
        self.counts = []
        rng = np.random.default_rng(seed)
        for i, day in enumerate((2, 3, 4, 7)):
            rows = np.flatnonzero(days == day)
            self.counts.append(len(rows))
            if i == self.heldout:
                splits = {"train": rows[:0], "val": rows[:0], "test": rows}
            else:
                if len(rows) < 3:
                    raise ValueError("Observed days require at least three cells")
                order = rng.permutation(rows)
                nv, nt = max(1, round(len(rows) * val)), max(1, round(len(rows) * test))
                if nv + nt >= len(rows):
                    raise ValueError("Split fractions leave no training cells")
                splits = {
                    "val": order[:nv],
                    "test": order[nv : nv + nt],
                    "train": order[nv + nt :],
                }
            self.rows.append(splits)
        self.observed = [i for i in range(4) if i != self.heldout]
        # Fit only observed training cells. No held-out day or val/test values
        # contribute to these statistics. Access is explicit, not import-time.
        train_rows = np.sort(
            np.concatenate([self.rows[i]["train"] for i in self.observed])
        )
        shift, scale = np.zeros(self.dim), np.ones(self.dim)
        if preprocessing is None and (normalise or whiten):
            with h5py.File(self.path, "r") as handle:
                values = handle["obsm/X_pca"][train_rows, : self.dim].astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Non-finite training PCA")
            if whiten:
                shift, scale = values.mean(0), values.std(0)
                scale = np.where(scale > 0, scale, 1)
            if normalise:
                standardized = (values - shift) / scale
                lo, hi = standardized.min(0), standardized.max(0)
                scale_range = np.where(hi > lo, hi - lo, 1)
                shift, scale = shift + scale * lo, scale * scale_range
        self.preprocessing = {
            "kind": "affine",
            "fit_split": "train",
            "fit_time_indices": self.observed,
            "normalise": normalise,
            "whiten": whiten,
            "shift": shift.astype(np.float32).tolist(),
            "scale": scale.astype(np.float32).tolist(),
            "feature_key": "obsm/X_pca",
            "pca_recomputed": False,
        }
        if preprocessing is not None:
            shift = np.asarray(preprocessing.get("shift", []), dtype=np.float32)
            scale = np.asarray(preprocessing.get("scale", []), dtype=np.float32)
            if (
                shift.shape != (self.dim,)
                or scale.shape != (self.dim,)
                or not np.isfinite(shift).all()
                or not np.isfinite(scale).all()
                or (scale <= 0).any()
            ):
                raise ValueError("Invalid saved teacher preprocessing")
            if preprocessing.get("feature_key") != "obsm/X_pca":
                raise ValueError(
                    "This loader requires the teacher's released PCA representation"
                )
            self.preprocessing = dict(preprocessing)
        self.metadata = TemporalMetadata(
            name=name,
            dim=self.dim,
            physical_times=(2.0, 3.0, 4.0, 7.0),
            model_times=(0.0, 1 / 3, 2 / 3, 1.0),
            preprocessing_id=identity(self.preprocessing),
            source_version=sha256(self.path),
            split_seed=seed,
        )
        self.params = dict(params)

    def get_marginal(self, time_index, split):
        if type(time_index) is not int or time_index not in range(4):
            raise ValueError("time_index must be 0, 1, 2 or 3")
        if split not in ("train", "val", "test"):
            raise ValueError("split must be train, val or test")
        return H5Marginal(
            self.path, self.rows[time_index][split], self.dim, self.preprocessing
        )

    def contract(self):
        return {
            **self.metadata.to_dict(),
            "source_file": str(self.path),
            "source_sha256": self.metadata.source_version,
            "params": self.params,
            "heldout_index": self.heldout,
            "observed_indices": self.observed,
            "cells_per_time": self.counts,
            "split_scheme": "independent_cells_within_observed_days; heldout_day_test_only",
            "split_counts": [
                {k: len(v) for k, v in rows.items()} for rows in self.rows
            ],
            "split_indices_sha256": identity(
                [{k: v.tolist() for k, v in row.items()} for row in self.rows]
            ),
            "initial_law": {"kind": "empirical_earliest_marginal", "physical_day": 2},
        }
