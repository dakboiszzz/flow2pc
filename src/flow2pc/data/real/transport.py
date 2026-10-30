"""Unpaired GOM/Beijing populations from the prepared OTP-FM data bundles."""

import numpy as np

from flow2pc.data.temporal import SnapshotDataset
from flow2pc.utils.checkpoint import sha256


class TransportMarginals(SnapshotDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The source protocol has temporal holdouts, with no extra row split.
        self.heldout_indices = self.source["holdout_indices"]
        self.observed = self.source["train_indices"]
        self.rows = []
        for index, cloud in enumerate(self.clouds):
            all_rows = np.arange(len(cloud), dtype=np.int64)
            empty = all_rows[:0]
            self.rows.append({
                "train": all_rows if index in self.observed else empty,
                "val": empty,
                "test": all_rows if index in self.heldout_indices else empty,
            })

    def read_source(self):
        if self.name not in ("gom", "beijing"):
            raise ValueError("TransportMarginals supports gom and beijing")
        allowed = {"normalise", "whiten", "preprocessing_protocol"}
        if set(self.params) - allowed:
            raise ValueError(f"Unknown transport data params: {set(self.params) - allowed}")
        if (self.params.get("preprocessing_protocol") != "native_all"
                or self.params.get("normalise") is not False
                or self.params.get("whiten") is not True):
            raise ValueError("Transport protocol requires native_all, whiten=true, normalise=false")
        path = self.root / self.name / "marginals.npz"
        with np.load(path, allow_pickle=False) as data:
            times = data["times"].copy()
            train, heldout = data["train_indices"].tolist(), data["holdout_indices"].tolist()
            if self.name == "gom":
                samples = data["samples"]
                if samples.shape != (9, 111, 2):
                    raise ValueError("Expected the original GOM (9,111,2) populations")
                clouds = [np.array(x, dtype=np.float32) for x in samples]
                physical, labels = list(range(9)), [f"Observation {i}" for i in range(9)]
                expected_train, expected_heldout = [0, 2, 4, 6, 8], [1, 3, 5, 7]
                units, feature = "observation_index; source elapsed-time units unspecified", "gom_xy"
            else:
                values, offsets = data["values"], data["offsets"]
                counts = [744, 706, 710, 716, 720, 721, 743, 734, 728, 702, 717, 730, 738]
                if (values.shape != (9409, 1) or offsets.shape != (14,)
                        or offsets.dtype.kind not in "iu" or offsets[0] != 0
                        or offsets[-1] != len(values) or np.diff(offsets).tolist() != counts):
                    raise ValueError("Expected the prepared 13 Dingling PM2.5 populations")
                clouds = [np.array(values[a:b], dtype=np.float32) for a, b in zip(offsets[:-1], offsets[1:])]
                physical = list(range(0, 25, 2))
                if data["month_indices"].tolist() != physical:
                    raise ValueError("Beijing calendar-month indices changed")
                labels = [f"{2013 + (2 + i * 2) // 12}-{(2 + i * 2) % 12 + 1:02d}" for i in range(13)]
                expected_train, expected_heldout = [0, 1, 3, 4, 6, 7, 9, 10, 12], [2, 5, 8, 11]
                units, feature = "calendar_month_index_since_2013-03", "dingling_pm25_micrograms_per_m3"
        if train != expected_train or heldout != expected_heldout:
            raise ValueError("Prepared training/holdout time indices differ from the documented protocol")
        if times.shape != (len(clouds),) or not np.allclose(times, np.linspace(0, 1, len(clouds)), rtol=0, atol=1e-7):
            raise ValueError("Expected uniformly indexed model times i/(T-1)")
        return clouds, physical, labels, {
            "files": {str(path.relative_to(self.root)): sha256(path)},
            "feature_key": feature, "physical_time_units": units,
            "train_indices": train, "holdout_indices": heldout,
            "has_trajectory_ids": False,
            "preprocessing_scope": "all marginals including temporal holdouts, per supplied data README",
            "coordinate_units": "source_coordinates" if self.name == "gom" else "micrograms_per_m3",
        }

    def contract(self):
        result = super().contract()
        result.update(
            heldout_indices=self.heldout_indices,
            split_scheme="all_rows_at_observed_times; temporal_holdouts_test_only; no_row_validation_split",
        )
        return result
