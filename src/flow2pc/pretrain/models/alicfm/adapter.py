"""Frozen ALI-CFM global-time velocity interface; no interval remapping."""

import json
from pathlib import Path

import torch
from torch import nn

from flow2pc.pretrain.base import validate_query
from flow2pc.pretrain.models.alicfm.profiles import make_models
from flow2pc.utils.checkpoint import load_teacher_artifact, sha256


class FrozenALICFM(nn.Module):
    def __init__(self, field, interpolant, manifest, preprocessing):
        super().__init__()
        self.field, self.interpolant = field, interpolant
        self.manifest, self.preprocessing = manifest, preprocessing
        self.dim = manifest["model_config"]["dim"]
        self.register_buffer(
            "shift",
            torch.tensor(
                preprocessing["shift"], device=next(field.parameters()).device
            ),
            persistent=False,
        )
        self.register_buffer(
            "scale",
            torch.tensor(
                preprocessing["scale"], device=next(field.parameters()).device
            ),
            persistent=False,
        )
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def velocity(self, x, t, *, context=None):
        times = validate_query(x, t, self.dim, next(self.field.parameters()), context)
        return self._evaluate(x, times)

    @torch.no_grad()
    def ode_velocity(self, t, x):
        # Adaptive solvers may evaluate internal stages beyond an endpoint.
        # Evaluate the same trained field there; do not clip or remap time.
        times = validate_query(
            x,
            t,
            self.dim,
            next(self.field.parameters()),
            None,
            allow_time_extrapolation=True,
        )
        return self._evaluate(x, times)

    def _evaluate(self, x, times):
        raw = self.manifest["model_config"].get("cfm_coordinates", "model") == "raw"
        field_x = x * self.scale + self.shift if raw else x
        config = self.manifest["model_config"]
        layout = config.get("time_input", "global")
        columns = [field_x]
        if layout == "global":
            columns.append(times[:, None])
        elif layout in ("local", "local_interval", "local_interval_global"):
            count = config["num_marginals"] - 1
            interval = torch.floor(times * count).clamp(0, count - 1)
            local = times * count - interval
            columns.append(local[:, None])
            if layout != "local":
                columns.append((interval / count)[:, None])
            if layout == "local_interval_global":
                columns.append(times[:, None])
        else:
            raise ValueError("Unknown saved teacher time-input layout")
        result = self.field(torch.cat(columns, dim=-1))
        if raw:
            result = result / self.scale
        if result.shape != x.shape or not torch.isfinite(result).all():
            raise ValueError("Teacher produced an invalid velocity")
        return result

    @torch.no_grad()
    def interpolate(self, x0, x1, t):
        parameter = next(self.field.parameters())
        times = validate_query(x0, t, self.dim, parameter, None)
        validate_query(x1, times, self.dim, parameter, None)
        if x0.shape != x1.shape:
            raise ValueError("Endpoint batch shapes differ")
        return self.interpolant(x0, x1, times[:, None], training=False)


def load(checkpoint, device="cpu"):
    payload, manifest = load_teacher_artifact(checkpoint, device)
    if manifest["method"] != "alicfm":
        raise ValueError("Artifact is not an ALI-CFM teacher")
    for source in manifest["source_provenance"]["files"]:
        local = Path(__file__).parent / "_source" / source["migrated_file"]
        if local.parent != Path(__file__).parent / "_source":
            raise ValueError("Invalid migrated-source path in teacher provenance")
        if sha256(local) != source["migrated_sha256"]:
            raise ValueError(
                "Teacher source provenance differs from this implementation"
            )
    config = manifest["model_config"]
    interpolant, _, field = make_models(config, device)
    field.load_state_dict(payload["cfm_model"], strict=True)
    interpolant.load_state_dict(payload["interpolant"], strict=True)
    preprocessing = json.loads(
        (
            Path(checkpoint).resolve().parent.parent / manifest["preprocessing_file"]
        ).read_text()
    )
    return FrozenALICFM(field, interpolant, manifest, preprocessing)
