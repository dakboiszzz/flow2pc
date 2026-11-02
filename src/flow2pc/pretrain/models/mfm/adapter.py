"""Frozen MFM velocity and native geopath sampling interfaces."""

from __future__ import annotations

import math

import torch
from torch import nn


class MetricVelocity(nn.Module):
    def __init__(self, network, shape):
        super().__init__()
        self.network = network
        self.sample_shape = tuple(shape)

    def forward(self, time, values):
        shape = values.shape
        if tuple(shape[1:]) not in (self.sample_shape, (math.prod(self.sample_shape),)):
            raise ValueError(
                f"Expected samples shaped {self.sample_shape}, got {tuple(shape[1:])}"
            )
        time = torch.as_tensor(time, device=values.device, dtype=values.dtype)
        if time.numel() == 1:
            time = time.reshape(1).expand(len(values))
        if time.numel() != len(values):
            raise ValueError("time must be scalar or have one value per sample")
        result = self.network(
            time.reshape(len(values)), values.reshape(len(values), *self.sample_shape)
        )
        if result.shape != (len(values), *self.sample_shape):
            raise ValueError("MFM teacher returned an unexpected velocity shape")
        return result.reshape(shape)

class MetricPathSampler:
    """Native geopath points, without unnecessary geopath time derivatives."""

    def __init__(self, network, *, sigma, ot_method="exact", alpha=1.0):
        self.network = network
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.ot_method = ot_method

    def couple(self, x0, x1):
        if self.ot_method in (None, "None", "none"):
            return x0, x1
        from torchcfm.optimal_transport import OTPlanSampler

        return OTPlanSampler(method=self.ot_method).sample_plan(x0, x1, replace=True)

    @torch.no_grad()
    def sample(self, x0, x1, left, right, *, time=None, noise=None):
        if x0.shape != x1.shape or not right > left:
            raise ValueError(
                "Geopath requires matching endpoint batches and an increasing interval"
            )
        time = (
            torch.rand(len(x0), device=x0.device, dtype=x0.dtype) * (right - left)
            + left
            if time is None
            else time.to(x0)
        )
        if time.shape != (len(x0),) or bool(((time < left) | (time > right)).any()):
            raise ValueError("Global sampling times must lie inside their interval")
        t = time.reshape(len(x0), *([1] * (x0.ndim - 1)))
        u = (t - left) / (right - left)
        correction = self.network(x0, x1, t) if self.alpha else 0.0
        # This is compute_mu_t in the upstream MetricFlowMatcher.
        mean = (1 - u) * x0 + u * x1 + (1 - u.square() - (1 - u).square()) * correction
        epsilon = torch.randn_like(x0) if noise is None else noise.to(x0)
        return time, mean + self.sigma * epsilon
