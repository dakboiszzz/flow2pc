"""Strict OTP-FM EMA adapter and cached conditional W2 interpolant.

Imports the trusted neighboring checkout explicitly. No ALI reconstruction,
Gaussian source fallback, downloads, or ODE sampling occur in this module.
"""

import importlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

from repc.single_cell import sha256_file

DEFAULT_OTP_ROOT = Path(__file__).resolve().parents[2] / "otp-fm"


def select_teacher(root, dataset):
    manifest_path = Path(root) / "reproduction/distillation_teachers.json"
    manifest = json.loads(manifest_path.read_text())
    matches = [row for row in manifest["teachers"] if row["dataset"] == dataset]
    if len(matches) != 1:
        raise ValueError(f"expected one selected teacher for {dataset}")
    entry = matches[0]
    for key in ("checkpoint", "config"):
        if sha256_file(Path(root) / entry[key]) != entry[key + "_sha256"]:
            raise ValueError(f"selected {dataset} {key} checksum mismatch")
    sources = {}
    for name, digest in manifest["reviewed_source_sha256"].items():
        if name.startswith("otp-fm/"):
            path = Path(root) / name.removeprefix("otp-fm/")
            if sha256_file(path) != digest:
                raise ValueError(f"reviewed OTP-FM source changed: {path}")
            sources[name] = digest
    # The kernel implementation is also part of the cache's executable contract.
    sources["otp-fm/src/otpfm/lambdas.py"] = sha256_file(Path(root) / "src/otpfm/lambdas.py")
    return entry, {"teacher_manifest_sha256": sha256_file(manifest_path),
                   "native_source_sha256": sources}


def native_factories(root):
    root = Path(root).resolve()
    for path in (root, root / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    module = importlib.import_module("experiments.train")
    if Path(module.__file__).resolve() != root / "experiments/train.py":
        raise ImportError("experiments.train was imported from a different checkout")
    return module.create_model, module.create_potentials


class OTPVelocity(nn.Module):
    def __init__(self, model):
        super().__init__()
        if model.flownet_ema is None or model.x_pred:
            raise ValueError("this adapter requires an EMA velocity-prediction teacher")
        self.model = model.eval().requires_grad_(False)

    def forward(self, time, values):
        if values.ndim != 2 or values.shape[1] != self.model.d:
            raise ValueError("expected normalized [batch, dimension] values")
        times = torch.as_tensor(time, device=values.device, dtype=values.dtype).reshape(-1)
        if times.numel() == 1:
            times = times.expand(len(values))
        if times.numel() != len(values):
            raise ValueError("expected scalar time or one time per value")
        return self.model.v_func_ema(values, times, times)


def load_teacher(root, dataset, device):
    entry, provenance = select_teacher(root, dataset)
    config = json.loads((Path(root) / entry["config"]).read_text())
    if config["potential"] != "w2inf" or not config["ot_coupling"] or not config["normalize"]:
        raise ValueError("fast cache requires the selected normalized, OT-coupled W2 preset")
    factory, potentials = native_factories(root)
    model = factory(config, entry["dim"], potentials(config), "cpu")
    saved = torch.load(Path(root) / entry["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    model.to(device)
    # These are ordinary attributes in the native model, not registered buffers.
    for name in ("tks", "dtks", "_x_time_deps_base", "_I_minus_A_inv", "_A_matrix"):
        value = getattr(model, name, None)
        if isinstance(value, torch.Tensor):
            setattr(model, name, value.to(device))
    teacher = OTPVelocity(model)
    return teacher, config, entry, provenance


@torch.no_grad()
def build_cache(model, tuples, alpha):
    from otpfm.potentials import W2InfPotential

    potentials = list(model.potentials.values())
    if not potentials or not all(isinstance(p, W2InfPotential) for p in potentials):
        raise ValueError("linear cache only supports W2InfPotential with precomputed OT tuples")
    k, n, dim = len(potentials), len(tuples), tuples.shape[-1]
    if tuples.ndim != 3 or tuples.shape[1] != k + 2 or not torch.isfinite(tuples).all():
        raise ValueError("invalid observed tuple shape or values")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0,1]")
    knots = tuples.new_tensor(list(model.potentials))
    strengths = tuples.new_tensor([p.strength for p in potentials])
    h = torch.stack([model.x_time_dep(p, knots[:, None]).reshape(k) for p in potentials], 1)
    a = h * strengths[None]
    system = torch.eye(k, device=tuples.device, dtype=tuples.dtype) - a
    lu, pivots = torch.linalg.lu_factor(system)
    x0, base = tuples[:, 0], tuples[:, -1] - tuples[:, 0]
    middle = tuples[:, 1:-1].transpose(0, 1)
    bases = x0[None] + knots[:, None, None] * base[None]
    rhs = bases - torch.einsum("ij,jnd->ind", a, middle)
    z = torch.linalg.lu_solve(lu, pivots, rhs.reshape(k, -1)).reshape(k, n, dim)
    residual = ((torch.einsum("ij,jnd->ind", system, z) - rhs).norm()
                / rhs.norm().clamp_min(1e-12)).item()
    if residual > 1e-5 or not torch.isfinite(z).all():
        raise ValueError(f"conditional linear solve failed: relative residual {residual}")
    cache = {"x0": x0, "base": base, "forces": strengths[:, None, None] * (z - middle),
             "alpha": float(alpha), "relative_linear_residual": residual,
             "condition_number": float(torch.linalg.cond(system)), "potential_times": knots}
    rows = torch.arange(n, device=tuples.device)
    for endpoint in (0., 1.):
        values, _ = sample_cache(model, cache, rows, tuples.new_full((n,), endpoint))
        expected = tuples[:, 0 if endpoint == 0 else -1]
        torch.testing.assert_close(values, expected, rtol=1e-5, atol=2e-5)
    return cache


def move_cache(cache, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in cache.items()}


def sample_cache(model, cache, rows, times):
    """Keep differentiable time queries for full-vector derivative validation."""
    t = times.reshape(-1, 1).to(cache["x0"])
    rows = rows.to(cache["x0"].device)
    if len(rows) != len(t):
        raise ValueError("one time is required per cached tuple row")
    potentials = list(model.potentials.values())
    h = torch.stack([model.x_time_dep(p, t) for p in potentials])
    j = torch.stack([model.v_time_dep(p, t) for p in potentials])
    forces = cache["forces"][:, rows]
    base = cache["base"][rows]
    values = cache["x0"][rows] + t * base + cache["alpha"] * (forces * h).sum(0)
    conditional_velocity = base + cache["alpha"] * (forces * j).sum(0)
    return values, conditional_velocity
