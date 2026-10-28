import itertools
import math
from bisect import bisect_right
from dataclasses import asdict, dataclass
from typing import Protocol


def time_grid(values, label):
    if len(values) < 2 or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in values
    ):
        raise ValueError(f"{label} requires at least two finite numeric times")
    if any(a >= b for a, b in itertools.pairwise(values)):
        raise ValueError(f"{label} must be strictly increasing")


def interpolate_time(value, source, target):
    if not math.isfinite(value) or not source[0] <= value <= source[-1]:
        raise ValueError("Time is outside the recorded interval")
    i = min(bisect_right(source, value) - 1, len(source) - 2)
    weight = (value - source[i]) / (source[i + 1] - source[i])
    return target[i] + weight * (target[i + 1] - target[i])


@dataclass(frozen=True)
class TemporalMetadata:
    name: str
    dim: int
    physical_times: tuple[float, ...]
    model_times: tuple[float, ...]
    preprocessing_id: str
    source_version: str
    split_seed: int
    has_trajectory_ids: bool = False
    dtype: str = "float32"

    def __post_init__(self):
        if type(self.dim) is not int or self.dim < 1:
            raise ValueError("dim must be a positive integer")
        time_grid(self.physical_times, "physical_times")
        time_grid(self.model_times, "model_times")
        if len(self.physical_times) != len(self.model_times):
            raise ValueError("Physical and model time grids have different lengths")
        if not self.name or not self.preprocessing_id or not self.source_version:
            raise ValueError(
                "Dataset name, source version and preprocessing are required"
            )

    def to_model_time(self, physical_t):
        return interpolate_time(physical_t, self.physical_times, self.model_times)

    def to_physical_time(self, model_t):
        return interpolate_time(model_t, self.model_times, self.physical_times)

    def to_dict(self):
        result = asdict(self)
        result["physical_times"] = list(self.physical_times)
        result["model_times"] = list(self.model_times)
        result["time_transform"] = {
            "kind": "piecewise_linear_between_recorded_knots",
            "physical_knots": list(self.physical_times),
            "model_knots": list(self.model_times),
            "velocity_convention": "dx/dmodel_time",
        }
        return result


class TemporalDataset(Protocol):
    @property
    def metadata(self) -> TemporalMetadata: ...

    def get_marginal(self, time_index: int, split: str):
        """Return an indexable marginal, without inventing cross-time pairings."""
        ...
