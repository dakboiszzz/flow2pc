"""Source-only tests for the local MFM launcher; not run automatically."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mfm_data_launcher", ROOT / "scripts/run_mfm_data.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def overrides(command):
    return dict((command[i + 1].split("=", 1)[0], json.loads(command[i + 1].split("=", 1)[1]))
                for i, value in enumerate(command) if value == "--set")


@pytest.mark.parametrize("name", list(launcher.PROFILES))
def test_every_profile_has_three_configs_and_matching_pair(name):
    args = launcher.parser().parse_args(["both", name, "--student", "both", "--dry-run"])
    commands = launcher.commands(args, name)
    assert len(commands) == 3
    teacher, hclt, gaussian = [yaml.safe_load(Path(c[3]).read_text()) for c in commands]
    assert teacher["model"]["params"]["backend"] == "snapshots"
    assert teacher["model"]["params"]["native"]["velocity_metric"] == "land"
    assert teacher["model"]["params"]["native"]["rho"] > 0
    assert teacher["data"]["params"]["profile"] == name
    run = Path(overrides(commands[0])["output.dir"])
    for command, config in zip(commands[1:], (hclt, gaussian), strict=True):
        values = overrides(command)
        assert values["teacher.checkpoint"] == str(run / "checkpoints/flow.pt")
        assert values["teacher.params.geopath_checkpoint"] == str(run / "checkpoints/geopath.pt")
        assert config["evaluation"]["enabled"] is False
        assert "source_nll_weight" not in config["distillation"]


def test_preview_never_spawns_or_creates_data(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run started an operation")

    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    monkeypatch.setattr(launcher, "link_sources", forbidden)
    assert launcher.main(["both", "all", "--dry-run", "--data-root", str(tmp_path / "data"),
                          "--output-root", str(tmp_path / "outputs")]) == 0
    assert list(tmp_path.iterdir()) == []


def test_pretrain_failure_never_starts_distill(monkeypatch):
    calls = []
    monkeypatch.setattr(launcher, "link_sources", lambda *args, **kwargs: None)

    def execute(command, **kwargs):
        calls.append(command)
        if "--config" in command:
            raise subprocess.CalledProcessError(9, command)

    monkeypatch.setattr(launcher.subprocess, "run", execute)
    assert launcher.main(["both", "gom"]) == 9
    assert len(calls) == 2  # One preparation process, then one teacher process.
    assert all("configs/distill" not in " ".join(command) for command in calls)
