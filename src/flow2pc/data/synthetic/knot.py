"""Noisy knot marginals from the supplied analytic experiment generator."""

from pathlib import Path

import numpy as np

from flow2pc.data.temporal import SnapshotDataset
from flow2pc.utils.checkpoint import sha256


def knot_means(size):
    if type(size) is not int or size < 3 or size % 3:
        raise ValueError("Knot size must be a positive multiple of three")
    t = np.linspace(0.0, 3.0, size)
    u = t - 1.5
    a, b, c = np.split(u, 3)
    x = np.concatenate([3 * (a + 0.5), np.cos(2 * np.pi * (b - 0.75)), 3 * (c - 0.5)])
    y = np.concatenate(
        [
            -0.5 * np.tanh(5 * (a + 1)) + 0.5,
            np.sin(2 * np.pi * (b - 0.75)) + 1,
            0.5 * np.tanh(5 * (c - 1)) + 0.5,
        ]
    )
    return np.stack([x, y], 1), t / 3.0


class Knot(SnapshotDataset):
    def read_source(self):
        allowed = {
            "size",
            "samples_per_time",
            "noise_std",
            "heldout_index",
            "validation_fraction",
            "test_fraction",
            "normalise",
            "whiten",
        }
        if set(self.params) - allowed:
            raise ValueError(f"Unknown knot params: {set(self.params) - allowed}")
        size, count, noise = (
            self.params.get("size", 60),
            self.params.get("samples_per_time", 10),
            self.params.get("noise_std", 0.01),
        )
        if (
            type(count) is not int
            or count < 3
            or type(noise) not in (int, float)
            or not np.isfinite(noise)
            or noise <= 0
        ):
            raise ValueError("Invalid knot sample count or noise")
        means, times = knot_means(size)
        rng = np.random.default_rng(self.seed)
        clouds = [mean + rng.standard_normal((count, 2)) * noise for mean in means]
        return (
            clouds,
            times.tolist(),
            [str(i) for i in range(size)],
            {
                "generator": "ALI knot_distribution.py analytic curve; independent coordinate noise",
                "feature_key": "generated_xy",
                "size": size,
                "samples_per_time": count,
                "generator_source_sha256": sha256(Path(__file__)),
                "noise_std": noise,
                "seed": self.seed,
                "physical_time_units": "normalized analytic curve parameter",
            },
        )
