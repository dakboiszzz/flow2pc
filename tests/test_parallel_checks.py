"""Parallel-launch contracts; prepared for explicit test execution only."""

import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


@pytest.fixture
def launcher():
    previous_bytecode_setting = sys.dont_write_bytecode
    path = Path(__file__).resolve().parents[1] / "scripts/run_alicfm_checks.py"
    name = "_parallel_check_contract"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)
    sys.dont_write_bytecode = previous_bytecode_setting


def test_preview_creates_no_output_and_starts_no_workers(
    launcher, tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        launcher,
        "gpu_inventory",
        lambda: {
            "0": {"uuid": "GPU-zero", "free": 16 * 2**30},
            "1": {"uuid": "GPU-one", "free": 16 * 2**30},
        },
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry-run must not start workers")

    monkeypatch.setattr(launcher, "run", forbidden)
    data = tmp_path / "data"
    data.mkdir()
    output = tmp_path / "results"
    code = launcher.main(
        [
            "--dry-run",
            "--datasets",
            "st",
            "knot",
            "--data-root",
            str(data),
            "--output-root",
            str(output),
        ]
    )
    assert code == 0 and not output.exists()
    assert '"dry_run": true' in capsys.readouterr().out


def test_selected_devices_respect_parent_visibility(launcher, monkeypatch):
    inventory = {str(i): {"uuid": f"GPU-{i}", "free": 16 * 2**30} for i in range(3)}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-1,GPU-2")
    assert launcher.selected_devices("auto", inventory) == ["1", "2"]
    with pytest.raises(ValueError, match="outside"):
        launcher.selected_devices("0,1", inventory)
    with pytest.raises(ValueError, match="at least two"):
        launcher.selected_devices("1", inventory)


def test_teacher_only_preparation_never_checks_or_queues_students(launcher, tmp_path, monkeypatch):
    args = SimpleNamespace(teachers_only=True, rerun_evaluation=False)
    phases = []
    monkeypatch.setattr(launcher, "compatible_config", lambda args, ds, phase: phases.append(phase))
    monkeypatch.setattr(launcher, "complete", lambda args, ds, phase: False)
    dataset = launcher.Dataset("cite5", tmp_path / "new")
    launcher.prepare_dataset(args, dataset)
    assert phases == ["teacher"]
    assert [task.stage for task in dataset.pending] == ["teacher"]
    with pytest.raises(ValueError, match="Teacher-only"):
        launcher.command_for(args, launcher.Task(dataset, "student"))
