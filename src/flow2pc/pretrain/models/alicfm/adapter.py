"""Frozen ALI-CFM global-time velocity interface; no interval remapping."""

import json
from pathlib import Path

import torch
from torch import nn

from flow2pc.pretrain.base import validate_query
from flow2pc.pretrain.models.alicfm.model import MLP, TrainableInterpolant
from flow2pc.utils.checkpoint import load_teacher_artifact, sha256


class FrozenALICFM(nn.Module):
    def __init__(self, field, interpolant, manifest, preprocessing):
        super().__init__()
        self.field, self.interpolant = field, interpolant
        self.manifest, self.preprocessing = manifest, preprocessing
        self.dim = manifest["model_config"]["dim"]
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def velocity(self, x, t, *, context=None):
        times = validate_query(x, t, self.dim, next(self.field.parameters()), context)
        result = self.field(torch.cat([x, times[:, None]], dim=-1))
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
    field = MLP(dim=config["dim"], w=config["cfm_dim"], time_varying=True).to(device)
    interpolant = TrainableInterpolant(
        dim=config["dim"],
        h_dim=config["interpolant_dim"],
        t_smooth=config["t_smooth"],
    ).to(device)
    field.load_state_dict(payload["cfm_model"], strict=True)
    interpolant.load_state_dict(payload["interpolant"], strict=True)
    preprocessing = json.loads(
        (
            Path(checkpoint).resolve().parent.parent / manifest["preprocessing_file"]
        ).read_text()
    )
    return FrozenALICFM(field, interpolant, manifest, preprocessing)
