"""Prepared-data and teacher-sweep contracts. Run only with user authorization."""

import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from flow2pc.data import mfm_snapshots
from flow2pc.data.restore import restore_dataset
from flow2pc.data.synthetic.prepared_mfm import PreparedMFMMarginals
from flow2pc.pretrain.models.alicfm.trainer import ALICFM
from flow2pc.pretrain.models.otpfm.marginals import validate as validate_otp

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sweep(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_mfm_teacher_sweep", ROOT / "scripts/run_mfm_teacher_sweep.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def metadata():
    return {name: {"identity": name, "model_times": [0., 1.] if name == "lidar" else [0., .5, 1.],
                   "heldout_indices": [] if name == "lidar" else [1]}
            for name in ("arch", "sphere", "lidar")}


def test_1536_unique_configs_reuse_splits_and_exclude_inactive_sweep_axes(tmp_path, sweep):
    args = sweep.parser().parse_args(["--data-root", str(tmp_path / "data"), "--output-root", str(tmp_path / "out")])
    runs = sweep.build_runs(args, metadata())
    assert len(runs) == len({run.name for run in runs}) == 1536
    assert set(Counter((run.method, run.dataset) for run in runs).values()) == {256}
    assert {(run.method, run.dataset) for run in runs[:6]} == set(Counter((run.method, run.dataset) for run in runs))
    for run in runs:
        assert run.config["data"]["params"] == {}
        assert run.config["data"]["root"].endswith(f"prepared/{run.dataset}/seed42")
        assert run.config["logging"]["project"] == "flow2pc"
        assert run.config["logging"]["name"].startswith(run.dataset)
        assert run.observed == ([0, 1] if run.dataset == "lidar" else [0, 2])
        if run.method == "alicfm":
            ALICFM().validate(run.config)
            assert run.config["training"]["protocol"] == "endpoints"
            assert run.config["training"]["num_ali_pretrain_steps"] == 0
            assert "gan_loss" not in run.params and "correct_coeff" not in run.params
        else:
            validate_otp(run.config)
            assert run.config["training"]["population_split"] == "train"
            assert not {"strength", "width_multiplier", "lambda_type", "otp_alpha_mean_scale"} & run.params.keys()
    assert not args.output_root.exists()


@pytest.mark.parametrize("name,dim,count", [("arch", 2, 3), ("sphere", 3, 3), ("lidar", 3, 2)])
def test_shared_loader_preserves_coordinates_splits_geometry_and_reload(tmp_path, monkeypatch, name, dim, count):
    clouds = [np.arange(60 * dim, dtype=np.float32).reshape(60, dim) + i for i in range(count)]
    info = {"labels": list(range(count)), "physical_times": None,
            "physical_time_units": "synthetic_transport_parameter", "model_times": np.linspace(0, 1, count).tolist(),
            "source_files": [], "provenance": {}}
    geometry = None
    if name == "lidar":
        info["provenance"]["upstream_normalization"] = {
            "mean": [1.] * dim, "scale": [2.] * dim, "fit_scope": "full_ground_scene_as_in_upstream"}
        geometry = [np.zeros((20, dim), np.float32), np.ones((20, dim), np.float32)]
    monkeypatch.setattr(mfm_snapshots, "read_source", lambda *args: (clouds, copy.deepcopy(info), None, geometry))
    prepared = mfm_snapshots.prepare_dataset(name, tmp_path, 42, tmp_path)
    original = mfm_snapshots.load({"data_dir": str(prepared)}, {})
    dataset = PreparedMFMMarginals(name=name, root=prepared, seed=123, params={})
    assert dataset.metadata.split_seed == 42
    for index in dataset.observed:
        torch.testing.assert_close(dataset.get_marginal(index, "train").materialize(), original.train_marginals[index], rtol=0, atol=0)
        torch.testing.assert_close(dataset.get_marginal(index, "val").materialize(), original.validation_marginals[index], rtol=0, atol=0)
    for index in dataset.heldout_indices:
        assert not len(dataset.get_marginal(index, "train"))
        assert not len(dataset.get_marginal(index, "val"))
        assert len(dataset.get_marginal(index, "test")) == len(clouds[index])
    restored = restore_dataset(dataset.contract(), dataset.preprocessing)
    assert restored.contract() == dataset.contract()
    if name == "lidar":
        torch.testing.assert_close(torch.cat(dataset.metric_clouds), torch.from_numpy(np.concatenate(geometry)))
        # Saved coordinates must not be whitened a second time.
        torch.testing.assert_close(dataset.clouds[0], torch.from_numpy(clouds[0]))


@pytest.mark.parametrize("consistency", ["meanflow", "imf"])
def test_original_otp_consistency_model_accepts_no_interior_potential(consistency):
    from flow2pc.pretrain.models.otpfm._source.factory import create_model, create_potentials

    config = yaml.safe_load((ROOT / "configs/pretrain/otpfm/otpfm_arch.yaml").read_text())["model"]["params"]["native_config"]
    config.update(tks=[], hidden_dim=8, x_emb_dim=8, t_emb_dim=8, num_hidden_layers=2,
                  consistency_loss=consistency)
    potentials = create_potentials(config)
    assert not potentials
    model = create_model(config, 2, potentials, "cpu")
    loss = model(torch.randn(4, 2, 2), otp_alpha=1., do_otp=False)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.flownet.parameters())


def test_rankings_use_validation_emd_even_if_test_order_is_reversed(tmp_path, sweep):
    args = sweep.parser().parse_args(["--datasets", "arch", "--methods", "alicfm", "--runs-per-dataset", "2",
                                      "--data-root", str(tmp_path), "--output-root", str(tmp_path / "out")])
    runs = sweep.build_runs(args, metadata())
    for run, val, test in zip(runs, (1., 2.), (100., .1), strict=True):
        run.status = "complete"
        run.selection = {"score": val}
        run.metrics = {"teacher_heldout_emd": test}
    sweep.summary(args, runs, [], "complete")
    best = json.loads((args.output_root / "alicfm/arch/best_models.json").read_text())
    assert best["test_score_used_for_selection"] is False
    assert best["best"][0]["run"] == runs[0].name


def test_dry_run_starts_no_processes_and_writes_no_outputs(tmp_path, monkeypatch, sweep):
    monkeypatch.setattr(sweep, "verify_inputs", lambda *args: metadata())
    monkeypatch.setattr(sweep, "gpu_inventory", lambda: pytest.fail("Dry run queried GPUs"))
    monkeypatch.setattr(sweep, "ParallelSweep", lambda *args, **kwargs: pytest.fail("Dry run launched jobs"))
    assert sweep.main(["--dry-run", "--runs-per-dataset", "1", "--output-root", str(tmp_path / "out")]) == 0
    assert not (tmp_path / "out").exists()
