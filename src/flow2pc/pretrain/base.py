from pathlib import Path
from typing import Protocol

import torch


class FrozenTeacher(Protocol):
    def velocity(
        self, x: torch.Tensor, t: torch.Tensor, *, context: dict | None = None
    ) -> torch.Tensor: ...


class TeacherMethod(Protocol):
    def fit(self, dataset, config: dict, run_dir: Path) -> Path: ...

    def load(self, checkpoint: Path, device: str) -> FrozenTeacher: ...


def validate_query(x, t, dim, parameter, context, *, allow_time_extrapolation=False):
    if context is not None:
        raise ValueError("This ALI-CFM adapter is unconditional; context must be None")
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] != dim:
        raise ValueError(f"Expected nonempty [B, {dim}] states")
    if x.device != parameter.device or x.dtype != parameter.dtype:
        raise ValueError("Input device and dtype must match the loaded teacher")
    times = torch.as_tensor(t, device=x.device, dtype=x.dtype)
    if times.ndim > 1 or (times.ndim == 1 and len(times) != len(x)):
        raise ValueError("Model time must be scalar or have shape [B]")
    if not torch.isfinite(x).all() or not torch.isfinite(times).all():
        raise ValueError("States and model times must be finite")
    if not allow_time_extrapolation and ((times < 0) | (times > 1)).any():
        raise ValueError("ALI-CFM requires global model time in [0,1]")
    return times.expand(len(x)) if times.ndim == 0 else times
