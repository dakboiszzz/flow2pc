"""Read existing MFM snapshots through the shared temporal dataset contract.

No generation, refitting, resplitting, or teacher imports occur here. The prepared
arrays are already in model coordinates; inverse transforms use their saved stats.
"""

from pathlib import Path

import numpy as np
from flow2pc.data.base import TemporalMetadata
from flow2pc.data.mfm_snapshots import load
from flow2pc.data.temporal import TensorMarginal
from flow2pc.utils.checkpoint import identity


class PreparedMFMMarginals:
    def __init__(self, *, name, root, seed, params, preprocessing=None):
        if name not in ("arch", "sphere", "lidar") or params:
            raise ValueError("Prepared MFM profiles are arch/sphere/lidar, with no split overrides")
        self.name, self.root, self.seed = name, Path(root).resolve(), seed
        self.params = {}
        data = load({"data_dir": str(self.root)}, {})
        manifest = data.source_info
        if manifest["profile"] != name:
            raise ValueError("Prepared MFM profile differs from data.name")
        # Training seeds may vary; saved observation splits never change.
        self.split_seed = manifest["split_seed"]
        self.dim = data.dim
        self.clouds = data.marginals
        self.observed = data.observed_indices
        self.heldout_indices = list(data.heldout_indices)
        self.heldout = self.heldout_indices[0] if len(self.heldout_indices) == 1 else None
        self.labels = [str(label) for label in manifest["labels"]]
        self.rows = []
        for index, cloud in enumerate(self.clouds):
            saved = data.split_indices[str(index)]
            self.rows.append({
                "train": np.asarray(saved["train"], dtype=np.int64),
                "val": np.asarray(saved["validation"], dtype=np.int64),
                "test": np.arange(len(cloud), dtype=np.int64) if index in self.heldout_indices
                        else np.empty(0, dtype=np.int64),
            })
        norm = manifest["normalization"]
        expected = {
            "kind": "affine", "normalise": False,
            "whiten": norm.get("fit_scope") != "identity",
            "shift": norm["mean"], "scale": norm["scale"],
            "metric_shift": norm["mean"], "metric_scale": norm["scale"],
            "fit_scope": norm["fit_scope"],
            "prepared_identity": manifest["identity"],
            "feature_key": name, "already_applied": True,
        }
        if preprocessing is not None and preprocessing != expected:
            raise ValueError("Saved teacher preprocessing differs from prepared MFM coordinates")
        self.preprocessing = expected
        self.source = {
            "prepared_identity": manifest["identity"],
            "arrays": manifest["arrays"], "source_files": manifest["source_files"],
            "split_protocol": manifest["split_protocol"],
            "physical_times_in_source": manifest["physical_times"],
            "physical_time_units": manifest["physical_time_units"],
            "time_semantics": "synthetic transport parameter; not measured physical time",
            "has_trajectory_ids": False,
        }
        self.metric_clouds = data.metric_marginals if name == "lidar" else []
        physical = manifest["physical_times"]
        if physical is None:
            physical = list(range(len(self.clouds)))
        self.metadata = TemporalMetadata(
            name=name, dim=self.dim, physical_times=tuple(float(t) for t in physical),
            model_times=tuple(manifest["model_times"]),
            preprocessing_id=identity(expected), source_version=manifest["identity"],
            split_seed=self.split_seed,
        )

    def get_marginal(self, time_index, split):
        if type(time_index) is not int or time_index not in range(len(self.clouds)):
            raise ValueError("Marginal index outside prepared grid")
        if split == "all":
            return TensorMarginal(self.clouds[time_index])
        if split not in ("train", "val", "test"):
            raise ValueError("Unknown marginal split")
        return TensorMarginal(self.clouds[time_index][self.rows[time_index][split]])

    def contract(self):
        return {
            **self.metadata.to_dict(), "root": str(self.root), "params": {},
            "source": self.source, "labels": self.labels,
            "observed_indices": self.observed, "heldout_indices": self.heldout_indices,
            "heldout_index": self.heldout, "cells_per_time": [len(x) for x in self.clouds],
            "split_scheme": "saved_MFM_rows_and_time_holdouts_unchanged",
            "split_counts": [{key: len(value) for key, value in row.items()} for row in self.rows],
            "split_indices_sha256": identity([{key: value.tolist() for key, value in row.items()} for row in self.rows]),
            "initial_law": {"kind": "empirical_earliest_marginal", "label": self.labels[0]},
        }
