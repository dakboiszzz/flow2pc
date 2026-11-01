"""Evaluation configuration without importing model implementations."""

import math

DEFAULTS = dict(
    enabled=False,
    num_samples=512,
    metric_samples=512,
    seed=2026,
    batch_size=64,
    tolerance=1e-5,
    max_nfe=50000,
    solver="dopri5",
    ode_steps=100,
    metric_space="historical",
    source_split="all",
    reference_split="all",
)


def resolve_evaluation(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS) - {"image_dir", "protocol"}:
        raise ValueError("Unknown evaluation settings")
    result = {**DEFAULTS, **value}
    if "protocol" in value and value["protocol"] not in ("shared_ode", "native_otpfm"):
        raise ValueError("Unknown evaluation.protocol")
    if type(result["enabled"]) is not bool:
        raise ValueError("evaluation.enabled must be boolean")
    for key in (
        "num_samples",
        "metric_samples",
        "seed",
        "batch_size",
        "max_nfe",
        "ode_steps",
    ):
        if type(result[key]) is not int or result[key] < (
            0 if key in ("num_samples", "metric_samples", "seed") else 1
        ):
            raise ValueError(f"Invalid evaluation.{key}")
    if result["seed"] >= 2**32:
        raise ValueError("evaluation.seed must fit an unsigned 32-bit integer")
    if (
        type(result["tolerance"]) not in (int, float)
        or not math.isfinite(result["tolerance"])
        or result["tolerance"] <= 0
    ):
        raise ValueError("evaluation.tolerance must be finite and positive")
    if result["solver"] not in ("dopri5", "rk4") or result["metric_space"] not in (
        "historical",
        "raw",
    ):
        raise ValueError("Invalid evaluation solver or metric space")
    for key in ("source_split", "reference_split"):
        if result[key] not in ("all", "train", "val", "test"):
            raise ValueError(f"Invalid evaluation.{key}")
    return result
