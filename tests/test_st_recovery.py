"""Recovered ST teacher contracts. Prepared for explicit execution by the user."""

import ast
import subprocess
from pathlib import Path

import pytest
import torch
import yaml

from flow2pc.pretrain.models.alicfm._source.st_interpolant import STInterpolant
from flow2pc.pretrain.models.alicfm.profiles import make_models
from flow2pc.pretrain.models.alicfm.trainer import ALICFM

ROOT = Path(__file__).resolve().parents[1]
REVISION = "501e9db6a3bcb1b96eb893882b62a56f78fe1823"


def st_config():
    return yaml.safe_load((ROOT / "configs/pretrain/alicfm_st.yaml").read_text())


def test_st_configuration_keeps_historical_method_and_batches():
    config = st_config()
    ALICFM().validate(config)
    assert config["model"]["params"]["reg_term_type"] == "piecewise"
    assert config["training"]["ali_batch_size"] == 256
    assert config["training"]["batch_size"] == 128
    config["model"]["params"]["reg_term_type"] = "picewise_oskar_version"
    with pytest.raises(ValueError, match="recovered ST"):
        ALICFM().validate(config)


def test_st_factory_restores_original_class_and_state_keys():
    params = dict(st_config()["model"]["params"], dim=2)
    generator, _, field = make_models(params, "cpu")
    assert isinstance(generator, STInterpolant)
    assert generator.interpolant_net[0].weight.shape == (64, 5)
    assert field.net[0].weight.shape == (128, 3)
    assert list(generator.state_dict()) == [
        f"interpolant_net.{index}.{kind}"
        for index in (0, 2, 4)
        for kind in ("weight", "bias")
    ]


def test_st_piecewise_call_accepts_coupled_populations_and_trains_generator():
    torch.manual_seed(12)
    generator = STInterpolant(2, 8, t_smooth=0.0)
    path = {
        0.0: torch.zeros(4, 2),
        2 / 3: torch.ones(4, 2),
        1.0: torch.full((4, 2), 2.0),
    }
    loss = generator.compute_piecewise_reg_term(path, metric="l2")
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in generator.parameters()
    )


def test_st_methods_match_recovered_source_in_git():
    upstream = ROOT.parent / "ali-cfm"
    if not (upstream / ".git").exists():
        pytest.skip("Original ALI-CFM Git history is not available")
    result = subprocess.run(
        ["git", "-C", str(upstream), "show", f"{REVISION}:ali_cfm/nets.py"],
        capture_output=True,
        text=True,
        check=True,
    )
    original = next(
        node
        for node in ast.parse(result.stdout).body
        if isinstance(node, ast.ClassDef) and node.name == "TrainableInterpolant"
    )
    copied = ast.parse(
        (
            ROOT / "src/flow2pc/pretrain/models/alicfm/_source/st_interpolant.py"
        ).read_text()
    )
    copied = next(node for node in copied.body if isinstance(node, ast.ClassDef))
    for name in ("compute_piecewise_reg_term", "dI_dt"):
        left = next(
            n
            for n in original.body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
        right = next(
            n for n in copied.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        assert ast.dump(left, include_attributes=False) == ast.dump(
            right, include_attributes=False
        )
