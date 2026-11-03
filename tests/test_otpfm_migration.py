"""OTP-FM contracts and source-parity regressions; run only when authorized."""

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from flow2pc.data.base import TemporalMetadata
from flow2pc.pretrain.models.otpfm.adapter import FrozenOTPFM
from flow2pc.pretrain.models.otpfm.provenance import verify_source
from flow2pc.pretrain.models.otpfm.trainer import OTPFMMethod
from flow2pc.utils.checkpoint import identity, load_teacher_artifact, save_teacher
from flow2pc.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/flow2pc/pretrain/models/otpfm"


def definitions(text):
    return {n.name: ast.dump(n) for n in ast.parse(text).body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}


def test_pinned_scientific_definitions_match_supplied_checkout():
    provenance = verify_source()
    native = ROOT.parent / "otp-fm"
    if not native.exists():
        pytest.skip("Native checkout unavailable")
    for record in provenance["files"]:
        upstream = native / record["upstream_file"]
        assert hashlib.sha256(upstream.read_bytes()).hexdigest() == record["upstream_sha256"]
        original = definitions(upstream.read_text())
        migrated = definitions((SOURCE / "_source" / record["migrated_file"]).read_text())
        for name, tree in migrated.items():
            assert tree == original[name], (record["migrated_file"], name)


@pytest.mark.parametrize("dataset", ["gom", "beijing"])
@pytest.mark.parametrize("recipe", ["w2", "w2inf", "mmd", "kl"])
def test_complete_native_presets_remain_explicit(dataset, recipe):
    config = load_config(ROOT / f"configs/pretrain/otpfm_{dataset}_{recipe}.yaml")
    OTPFMMethod().validate(config)
    assert config["evaluation"]["protocol"] == "native_otpfm"
    assert config["logging"]["project"] == "flow2pc"
    native = config["model"]["params"]["native_config"]
    if recipe in ("w2", "w2inf"):
        assert native["potential"] == "w2inf"
        assert native["ot_coupling"] is (recipe == "w2")
    assert native["normalize"] is True
    assert config["data"]["params"]["normalise"] is False


class InstantaneousFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.d, self.x_pred = 2, False
        self.flownet_ema = nn.Linear(2, 2)
        self.calls = []

    def v_func_ema(self, x, t1, t2):
        self.calls.append((t1.clone(), t2.clone()))
        return x + (t2 - t1)[:, None]


def test_frozen_teacher_queries_diagonal_ema_and_preserves_interval_api():
    native = InstantaneousFixture()
    teacher = FrozenOTPFM(native, {"model_config": {"dim": 2}}, {})
    values, times = torch.ones(3, 2), torch.tensor([0.1, 0.2, 0.3])
    torch.testing.assert_close(teacher.velocity(values, times), values)
    torch.testing.assert_close(native.calls[-1][0], native.calls[-1][1])
    torch.testing.assert_close(teacher.mean_velocity(values, times, times + 0.1), values + 0.1)
    assert not any(p.requires_grad for p in teacher.parameters())
    with pytest.raises(ValueError, match="context"):
        teacher.velocity(values, times, context={"unsupported": True})
    with pytest.raises(NotImplementedError, match="tuple"):
        teacher.interpolate(values, values, times)


def test_native_protocol_cannot_silently_become_shared_ode():
    config = load_config(ROOT / "configs/pretrain/otpfm_gom_w2.yaml")
    config["evaluation"]["protocol"] = "shared_ode"
    with pytest.raises(ValueError, match="native_otpfm"):
        OTPFMMethod().validate(config)


def test_best_checkpoint_contract_preserves_named_checkpoint(tmp_path):
    preprocessing = {"kind": "fixture", "shift": [0.0], "scale": [1.0]}
    metadata = TemporalMetadata("fixture", 1, (0.0, 1.0), (0.0, 1.0), identity(preprocessing), "fixture", 42)
    manifest = {"method": "fixture", "method_version": "fixture", "model_config": {"dim": 1},
                "training_config": {}, "dataset": metadata.to_dict(), "seed": 42, "source_provenance": {}}
    path = save_teacher(tmp_path, {"model_state_dict": {"weight": torch.ones(1)}}, manifest, preprocessing, checkpoint_name="best")
    assert path.name == "best.pt"
    payload, restored = load_teacher_artifact(path)
    assert restored["checkpoint_relpath"] == "checkpoints/best.pt"
    torch.testing.assert_close(payload["model_state_dict"]["weight"], torch.ones(1))


def test_teacher_package_has_no_student_imports():
    for path in SOURCE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("flow2pc.distill", "flow2pc.pc")), path


def test_shared_logger_alias_preserves_existing_import():
    from flow2pc.utils.tracking import LocalLogger
    from flow2pc.pretrain.models.alicfm.trainer import LocalLogger as PreviousPath

    assert PreviousPath is LocalLogger
