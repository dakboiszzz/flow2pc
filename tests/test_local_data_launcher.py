"""Launcher contracts without importing models or running training subprocesses."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "local_data_launcher", ROOT / "scripts/run_local_data.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def overrides(command):
    result = {}
    for index, token in enumerate(command):
        if token == "--set":
            key, value = command[index + 1].split("=", 1)
            result[key] = json.loads(value)
    return result


@pytest.mark.parametrize("dataset", launcher.ALI_DATASETS)
def test_both_stages_bind_the_same_local_teacher(tmp_path, dataset):
    args = launcher.parser().parse_args([
        "both", dataset, "--data-root", str(tmp_path / "data with spaces"),
        "--output-root", str(tmp_path / "results"), "--dry-run",
    ])
    commands, root, checkpoint = launcher.build_commands(args)
    pretrain, distill = map(overrides, commands)
    assert root == tmp_path / "data with spaces"
    assert pretrain["data.root"] == str(root)
    assert Path(pretrain["output.dir"]) / "checkpoints/last.pt" == checkpoint
    assert distill["teacher.checkpoint"] == str(checkpoint)
    assert "data.root" not in distill
    assert pretrain["logging.mode"] == distill["logging.mode"] == "disabled"


def test_dry_run_does_not_spawn_or_create_outputs(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preview must not start any subprocess")

    monkeypatch.setattr(launcher.subprocess, "run", forbidden)
    assert launcher.main([
        "both", "cite50", "--dry-run", "--output-root", str(tmp_path / "outputs"),
        "--data-root", str(tmp_path / "missing"),
    ]) == 0
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("dataset", launcher.TRANSPORT_DATASETS)
@pytest.mark.parametrize("recipe", ("w2", "w2inf", "kl", "mmd"))
def test_transport_paths_and_recipe(dataset, recipe):
    args = launcher.parser().parse_args([
        "pretrain", dataset, "--method", "otpfm", "--recipe", recipe,
    ])
    commands, root, _ = launcher.build_commands(args)
    values = overrides(commands[0])
    assert root == ROOT / "data"
    assert values["model.params.native_config.data_dir"] == str(root / dataset)
    assert commands[0][3].endswith(f"otpfm_{dataset}_{recipe}.yaml")


@pytest.mark.parametrize("arguments,reason", [
    (["distill", "eb", "--method", "otpfm"], "conditional tuple sampler"),
    (["both", "eb", "--validate-only"], "Validate stages separately"),
])
def test_unsupported_protocols_fail_before_execution(arguments, reason):
    with pytest.raises(ValueError, match=reason):
        launcher.build_commands(launcher.parser().parse_args(arguments))


@pytest.mark.parametrize("dataset", launcher.TRANSPORT_DATASETS)
def test_ali_transport_uses_the_temporal_validation_preset(dataset):
    import yaml

    args = launcher.parser().parse_args(["both", dataset, "--dry-run"])
    commands, _, checkpoint = launcher.build_commands(args)
    student = yaml.safe_load(Path(commands[1][3]).read_text())
    assert student["distillation"]["validation_scope"] == "heldout_times"
    assert overrides(commands[1])["teacher.checkpoint"] == str(checkpoint)
    assert Path(commands[0][3]).name == f"alicfm_{dataset}.yaml"
    assert Path(commands[1][3]).name == f"alicfm_{dataset}_hclt.yaml"


def test_failed_pretrain_does_not_start_student(tmp_path, monkeypatch):
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(launcher.subprocess, "run", fail)
    # Knot has an analytic source, so no local dataset fixture is needed.
    assert launcher.main([
        "both", "knot", "--output-root", str(tmp_path),
    ]) == 7
    assert len(calls) == 1
