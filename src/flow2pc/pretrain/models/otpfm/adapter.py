"""Frozen native EMA diagonal velocity and finite-interval sampling contracts."""

from pathlib import Path

import torch
from torch import nn

from flow2pc.pretrain.base import validate_query
from flow2pc.utils.checkpoint import load_teacher_artifact, sha256

from ._source.factory import create_model, create_potentials


class FrozenOTPFM(nn.Module):
    def __init__(self, model, manifest, preprocessing):
        super().__init__()
        self.model, self.manifest, self.preprocessing = model, manifest, preprocessing
        self.dim = model.d
        if model.flownet_ema is None or model.x_pred:
            raise ValueError("The instantaneous OTP-FM teacher adapter requires EMA and native v-prediction")
        self.eval().requires_grad_(False)

    def _times(self, x, t, context=None, extrapolate=False):
        try:
            return validate_query(x, t, self.dim, next(self.model.parameters()), context,
                                  allow_time_extrapolation=extrapolate)
        except ValueError as error:
            raise ValueError(str(error).replace("ALI-CFM", "OTP-FM")) from None

    @staticmethod
    def _checked(result, x):
        if result.shape != x.shape or not torch.isfinite(result).all():
            raise ValueError("OTP-FM teacher returned a nonfinite or incorrectly shaped velocity")
        return result

    @torch.no_grad()
    def velocity(self, x, t, *, context=None):
        times = self._times(x, t, context)
        return self._checked(self.model.v_func_ema(x, times, times), x)

    @torch.no_grad()
    def ode_velocity(self, t, x):
        times = self._times(x, t, extrapolate=True)
        return self._checked(self.model.v_func_ema(x, times, times), x)

    @torch.no_grad()
    def mean_velocity(self, x, t1, t2, *, context=None):
        start, end = self._times(x, t1, context), self._times(x, t2, context)
        if (end < start).any():
            raise ValueError("OTP-FM interval end must not precede its start")
        return self._checked(self.model.v_func_ema(x, start, end), x)

    @torch.no_grad()
    def sample(self, x0, n_steps):
        self._times(x0, 0.0)
        if type(n_steps) is not int or n_steps < 1:
            raise ValueError("Native sampling steps must be a positive integer")
        return self.model.sample(x0, n_steps=n_steps, ema=True)

    def interpolate(self, *args, **kwargs):
        raise NotImplementedError("OTP-FM conditional interpolation requires all observed tuple members; the PC sampler is a separate integration, not an ALI endpoint generator")


def load(checkpoint, device="cpu"):
    payload, manifest = load_teacher_artifact(checkpoint, device="cpu")
    if manifest["method"] != "otpfm":
        raise ValueError("Expected an OTP-FM teacher checkpoint")
    source = Path(__file__).with_name("_source")
    for record in manifest["source_provenance"]["files"]:
        if sha256(source / record["migrated_file"]) != record["migrated_sha256"]:
            raise ValueError("OTP-FM source differs from the checkpoint's recorded implementation")
    config = manifest["model_config"]["native_config"]
    model = create_model(config, manifest["model_config"]["dim"], create_potentials(config), device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    import json

    preprocessing = json.loads((Path(checkpoint).resolve().parent.parent / "preprocessing.json").read_text())
    return FrozenOTPFM(model, manifest, preprocessing)
