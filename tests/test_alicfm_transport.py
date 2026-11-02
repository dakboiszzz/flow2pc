"""Transport loader/sweep regressions; execute only with explicit authorization."""

import importlib.util
import errno
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flow2pc.data.real.transport import TransportMarginals
from flow2pc.data.restore import restore_dataset
from flow2pc.evaluation.config import resolve_evaluation
from flow2pc.evaluation.runner import evaluate

ROOT = Path(__file__).resolve().parents[1]


def gom_bundle(tmp_path):
    folder = tmp_path / "gom"
    folder.mkdir()
    samples = np.random.default_rng(2).normal(size=(9, 111, 2)).astype(np.float32)
    samples[1::2] += 10  # Holdout statistics are used; holdout optimization is not.
    np.savez(folder / "marginals.npz", samples=samples, times=np.arange(9) / 8,
             train_indices=[0, 2, 4, 6, 8], holdout_indices=[1, 3, 5, 7])
    return TransportMarginals(name="gom", root=tmp_path, seed=42, params={
        "normalise": False, "whiten": True, "preprocessing_protocol": "native_all",
    })


def test_multiple_holdouts_are_excluded_and_restore_exactly(tmp_path):
    dataset = gom_bundle(tmp_path)
    assert dataset.observed == [0, 2, 4, 6, 8]
    assert dataset.heldout_indices == [1, 3, 5, 7]
    assert dataset.preprocessing["heldout_time_contributes_to_statistics"]
    for index in range(9):
        assert len(dataset.get_marginal(index, "train")) == (111 if index in dataset.observed else 0)
        assert len(dataset.get_marginal(index, "test")) == (0 if index in dataset.observed else 111)
        assert len(dataset.get_marginal(index, "val")) == 0
    assert dataset.contract()["has_trajectory_ids"] is False
    restored = restore_dataset(dataset.contract(), dataset.preprocessing)
    assert restored.contract() == dataset.contract()


def test_evaluation_ranks_all_four_holdouts_not_only_one(tmp_path):
    dataset = gom_bundle(tmp_path)
    class ZeroField(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.manifest = {"method": "fixture", "checkpoint_sha256": "constant"}

        def ode_velocity(self, time, values):
            return torch.zeros_like(values)

    settings = resolve_evaluation({"enabled": True, "solver": "rk4", "num_samples": 8,
                                   "metric_samples": 8, "ode_steps": 8, "metric_space": "raw"})
    result = evaluate(ZeroField(), dataset, settings, tmp_path / "eval")
    heldout = [row for row in result["scores"] if row["held_out"]]
    assert [row["index"] for row in heldout] == [1, 3, 5, 7]
    assert result["teacher_heldout_emd"] == pytest.approx(np.mean([r["teacher_emd"] for r in heldout]))
    assert (tmp_path / "eval/teacher_ode_marginals.png").is_file()


def test_beijing_one_dimensional_final_figures(tmp_path):
    from flow2pc.evaluation.figures import save_figures

    folder = tmp_path / "beijing"
    folder.mkdir()
    counts = np.array([744, 706, 710, 716, 720, 721, 743, 734, 728, 702, 717, 730, 738])
    np.savez(folder / "marginals.npz", values=np.linspace(1, 100, 9409, dtype=np.float32)[:, None],
             offsets=np.r_[0, counts.cumsum()], times=np.arange(13) / 12,
             train_indices=[0, 1, 3, 4, 6, 7, 9, 10, 12], holdout_indices=[2, 5, 8, 11],
             month_indices=np.arange(0, 25, 2))
    dataset = TransportMarginals(name="beijing", root=tmp_path, seed=42, params={
        "normalise": False, "whiten": True, "preprocessing_protocol": "native_all",
    })
    path = torch.stack([cloud[:8] for cloud in dataset.clouds])
    files = save_figures(dataset, path, None, tmp_path, {"metric_space": "raw"})
    assert dataset.labels[0] == "2013-03" and dataset.labels[-1] == "2015-03"
    assert files["teacher_marginals"].is_file() and files["teacher_quantiles"].is_file()


def test_sweep_has_512_unique_configs_per_dataset_and_evaluation_recovery(tmp_path, monkeypatch):
    path = ROOT / "scripts/run_alicfm_transport_sweep.py"
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_transport_sweep_contract", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    args = SimpleNamespace(datasets=["gom", "beijing"], seed=42, output_root=tmp_path,
                           data_root=tmp_path, pretrain_steps=2000, gan_steps=30000, cfm_steps=10000)
    runs = module.build_runs(args)
    assert len(runs) == len({run.name for run in runs}) == 1024
    assert [run.dataset for run in runs[:4]] == ["gom", "beijing", "gom", "beijing"]
    assert all(run.config["logging"]["project"] == "flow2pc" for run in runs)
    for run in runs:
        assert run.config["stage"] == "pretrain"
        assert run.config["training"]["protocol"] == "marginal"
    run = runs[0]
    run.output.mkdir(parents=True)
    import yaml

    (run.output / "resolved_config.yaml").write_text(yaml.safe_dump(run.config))
    (run.output / "checkpoints").mkdir()
    (run.output / "checkpoints/last.pt").write_bytes(b"fixture; never loaded")
    module.prepare(run)
    assert run.action == "evaluate" and run.status == "pending"
    command = module.command_for(run, tmp_path / "config.yaml")
    assert command[-2:] == ["--evaluate", "teacher"]
    assert "distill" not in json.dumps(command)


def test_transport_logger_keeps_local_metrics_on_wandb_communication_failure(tmp_path, monkeypatch):
    from flow2pc.pretrain.models.alicfm.trainer import LocalLogger

    class CommunicationError(Exception):
        pass

    class FakeRun:
        url, entity = "https://example.invalid/run", "fixture"

        def log(self, row):
            raise CommunicationError("service process busy")

        def finish(self, **kwargs):
            pass

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        errors=SimpleNamespace(CommError=CommunicationError), init=lambda **kwargs: FakeRun(),
    ))
    config = {"training": {"protocol": "marginal"},
              "logging": {"mode": "online", "project": "flow2pc", "name": "fixture"}}
    logger = LocalLogger(tmp_path, config)
    logger.log({"ali/cfm/cfm_loss": 0.5, "ali/cfm_step": 0})
    assert logger.run is None
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["ali/cfm/cfm_loss"] == 0.5
    assert json.loads((tmp_path / "tracking_status.json").read_text())["status"] == "pending_sync"
    with pytest.raises(ValueError, match="Non-finite"):
        logger.log({"ali/cfm/cfm_loss": float("nan")})
    logger.close()


def test_transport_wandb_startup_falls_back_to_offline(tmp_path, monkeypatch):
    from flow2pc.pretrain.models.alicfm.trainer import LocalLogger

    class CommunicationError(Exception):
        pass

    calls = []
    def initialize(**kwargs):
        calls.append(kwargs["mode"])
        if kwargs["mode"] == "online":
            raise CommunicationError("request timed out")
        return SimpleNamespace(url=None, entity="fixture", log=lambda row: None, finish=lambda **kwargs: None)

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        errors=SimpleNamespace(CommError=CommunicationError), init=initialize,
    ))
    config = {"training": {"protocol": "marginal"},
              "logging": {"mode": "online", "project": "flow2pc", "name": "fixture"}}
    logger = LocalLogger(tmp_path, config)
    assert calls == ["online", "offline"]
    assert logger.run is not None and logger.offline
    logger.close()


def test_wandb_publication_failure_does_not_invalidate_saved_teacher(tmp_path, monkeypatch):
    path = ROOT / "scripts/run_alicfm_transport_sweep.py"
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_transport_publication_contract", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    output = tmp_path / "gom" / "fixture"
    (output / "checkpoints").mkdir(parents=True)
    (output / "eval").mkdir()
    config = {"stage": "pretrain", "training": {"protocol": "marginal", "device": "cuda:0"},
              "evaluation": {"enabled": True, "metric_space": "raw"}}
    import yaml

    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    (output / "checkpoints/last.pt").write_bytes(b"saved teacher; never loaded")
    image = output / "eval/teacher.png"
    image.write_bytes(b"existing figure; never decoded")
    for name in ("status.json", "eval/status.json"):
        (output / name).write_text(json.dumps({"status": "failed", "error": "W&B API service timed out"}))
    metrics = {"protocol": {"metric_space": "raw"}, "teacher_heldout_emd": 0.5,
               "scores": [{"index": i, "held_out": True, "teacher_emd": 0.5} for i in [1, 3, 5, 7]]}
    (output / "eval/metrics.json").write_text(json.dumps(metrics))
    (output / "eval/media.json").write_text(json.dumps({"teacher_marginals": str(image)}))
    run = module.Run("gom", "fixture", config, output, {})
    module.prepare(run)
    assert run.status == "complete" and run.tracking_status == "pending_sync"
    with pytest.raises(ValueError, match="Refusing to archive"):
        module.archive_incomplete(run)


def test_restart_preserves_incomplete_attempt(tmp_path, monkeypatch):
    path = ROOT / "scripts/run_alicfm_transport_sweep.py"
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_transport_archive_contract", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    output = tmp_path / "gom" / "fixture"
    output.mkdir(parents=True)
    (output / "metrics.jsonl").write_text("original partial metrics\n")
    run = module.Run("gom", "fixture", {}, output, {})
    module.archive_incomplete(run)
    paths = list((tmp_path / "_attempts/gom/fixture").glob("*/metrics.jsonl"))
    assert len(paths) == 1 and paths[0].read_text() == "original partial metrics\n"
    assert not output.exists()


@pytest.fixture
def sweep_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_transport_storage_contract", ROOT / "scripts/run_alicfm_transport_sweep.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_summary_disk_full_does_not_stop_active_worker(tmp_path, monkeypatch, sweep_module):
    module = sweep_module
    args = SimpleNamespace(output_root=tmp_path, min_disk_free_gb=20,
                           workers_per_gpu=1, min_free_gb=4, max_parallel=8)
    run = module.Run("gom", "fixture", {}, tmp_path / "fixture", {})
    stopped, reports = [], []
    class Process:
        pid = 123
        polls = 0

        def poll(self):
            self.polls += 1
            return 0 if self.polls >= 2 else None

    process = Process()
    def full_summary(*args):
        reports.append(args[-1])
        raise OSError(errno.ENOSPC, "filesystem full")
    def completed(task):
        task.status = "complete"
    monkeypatch.setattr(module, "summary", full_summary)
    monkeypatch.setattr(module, "inspect_run", completed)
    monkeypatch.setattr(module, "start_worker", lambda *args: dict(
        task=run, device="0", process=process, log=io.StringIO(), started=0))
    monkeypatch.setattr(module, "gpu_inventory", lambda: {"0": {"free": 20}})
    monkeypatch.setattr(module, "choose_device", lambda *args: "0")
    monkeypatch.setattr(module, "stop_workers", lambda active: stopped.extend(active))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=100 * 2**30))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(module.signal, "signal", lambda *args: None)
    assert module.run_queue(args, [run], ["0"], None) == 0
    assert run.status == "complete" and process.polls == 2
    assert reports and not stopped


def test_summary_permission_error_is_not_hidden(tmp_path, monkeypatch, sweep_module):
    def denied(*args):
        raise PermissionError(errno.EACCES, "permission denied")
    monkeypatch.setattr(sweep_module, "summary", denied)
    with pytest.raises(PermissionError):
        sweep_module.try_summary(SimpleNamespace(output_root=tmp_path), [], [], "running")


def test_worker_temporary_and_wandb_files_use_monitored_filesystem(tmp_path, monkeypatch, sweep_module):
    monkeypatch.setenv("TMPDIR", "/unmonitored/root/tmp")
    args = SimpleNamespace(output_root=tmp_path, threads=4)
    env = sweep_module.worker_environment(args, {"uuid": "GPU-fixture"})
    for key in ("TMPDIR", "TMP", "TEMP", "WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        assert Path(env[key]).is_relative_to(tmp_path / "_runtime")
        assert Path(env[key]).is_dir()
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-fixture"
