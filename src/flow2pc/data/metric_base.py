"""Shared dataset metadata and bundle interfaces."""
from dataclasses import dataclass, field
from math import prod
from typing import Any

@dataclass(frozen=True)
class DatasetSpec:
    factory: object
    pipeline: str = "multimarginal"
    real_nll: bool = False
    endpoint_coupling: str = "independent"
    endpoint_factory: object = None
    plot_script: str | None = None
    plot_metrics: bool = False
    validate: object = None
    finite_guard: bool = False


@dataclass
class DatasetBundle:
    loader: Any
    sample_shape: tuple
    pipeline: str
    training: Any = None
    validation: Any = None
    times: Any = None
    conditions: Any = None
    source_info: dict = field(default_factory=dict)

    @property
    def dim(self):
        return prod(self.sample_shape)

    def endpoint_pools(self):
        snapshots = self.loader.dataset.data
        return snapshots[0], snapshots[-1]


