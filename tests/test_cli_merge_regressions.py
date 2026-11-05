"""Startup and evaluation routing contracts; no real model execution.

Run only when the user authorizes tests.
"""

import sys
from types import SimpleNamespace

import pytest

from flow2pc import cli
from flow2pc.pretrain.registry import teachers


def test_teacher_registry_preserves_all_implemented_methods():
    expected = {
        "alicfm": "flow2pc.pretrain.models.alicfm.trainer:ALICFM",
        "otpfm": "flow2pc.pretrain.models.otpfm.trainer:OTPFMMethod",
        "mfm": "flow2pc.pretrain.models.mfm.trainer:MFM",
        "mmfm": "flow2pc.pretrain.models.mmfm.trainer:MMFM",
    }
    assert set(teachers.describe()) == set(expected)
    for name, target in expected.items():
        assert teachers.require(name).target == target


@pytest.mark.parametrize("role", ["teacher", "student"])
def test_checkpoint_evaluation_returns_without_training(monkeypatch, role):
    config = {"stage": "pretrain" if role == "teacher" else "distill"}
    monkeypatch.setattr(cli, "load_config", lambda *args: config)
    calls = []

    def evaluate(received, received_role):
        calls.append((received, received_role))
        return {"model": received_role}

    monkeypatch.setitem(
        sys.modules, "flow2pc.evaluation.entrypoint",
        SimpleNamespace(run=evaluate),
    )
    assert cli.main(["--config", "unused.yaml", "--evaluate", role]) == 0
    assert calls == [(config, role)]


@pytest.mark.parametrize("stage", ["pretrain", "distill"])
def test_mfm_evaluate_flag_fails_instead_of_starting_training(monkeypatch, capsys, stage):
    monkeypatch.setattr(cli, "load_config", lambda *args: (stage, {}))
    assert cli.main(["--config", "unused.yaml", "--evaluate", "teacher"]) == 2
    assert "stage: evaluate" in capsys.readouterr().err
