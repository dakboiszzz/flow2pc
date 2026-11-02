"""Shared-data OTP-FM contracts; run only when the user authorizes tests."""

import copy
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from flow2pc.evaluation.config import DEFAULTS, resolve_evaluation
from flow2pc.evaluation.figures import TRACKING_FRAMES
from flow2pc.pretrain.models.otpfm._source.common_trainer import Trainer
from flow2pc.pretrain.models.otpfm.marginals import MarginalTrainer, validate
from flow2pc.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sweep(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_otpfm_dataset_contract", ROOT / "scripts/run_otpfm_dataset_sweep.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def specification_and_args(tmp_path, sweep):
    specification = copy.deepcopy(sweep.DEFAULT_SWEEP)
    args = SimpleNamespace(runs_per_dataset=100, datasets=list(specification["datasets"]),
                           grid_seed=2026, seed=42, steps=10000, mode="offline",
                           data_root=tmp_path / "data", output_root=tmp_path / "outputs")
    return specification, args


def test_exactly_100_distinct_configs_per_profile_with_fair_order(tmp_path, sweep):
    specification, args = specification_and_args(tmp_path, sweep)
    runs = sweep.build_runs(args, specification)
    assert len(runs) == len({run.name for run in runs}) == 1000
    assert set(Counter(run.dataset for run in runs).values()) == {100}
    assert [run.dataset for run in runs[:10]] == args.datasets
    assert all(run.config["logging"]["project"] == "flow2pc" for run in runs)
    assert not args.output_root.exists()
    for run in runs:
        validate(run.config)
        assert run.config["model"]["params"]["native_config"]["potential"] == "w2inf"
        assert run.config["data"]["root"] == str(args.data_root)


def test_each_dataset_inherits_its_standalone_otpfm_config(tmp_path, sweep):
    specification, args = specification_and_args(tmp_path, sweep)
    args.runs_per_dataset = 1
    for run in sweep.build_runs(args, specification):
        profile = specification["datasets"][run.dataset]
        original = load_config(ROOT / "configs/pretrain" / profile["config"])
        expected_data = copy.deepcopy(original["data"])
        expected_data["root"] = str(args.data_root)
        assert run.config["data"] == expected_data
        expected_evaluation = original["evaluation"]
        if run.dataset == "cell_tracking":
            assert expected_evaluation["time_indices"] == [0, 24, 49, 74, 99, 114]
            assert run.config["training"]["samples_per_marginal"] == 10
            assert run.ranking_metric == "teacher_mean_emd"
        assert run.config["evaluation"] == expected_evaluation
        if run.dataset == "eb":
            assert run.config["training"]["population_split"] == "train"
        if run.dataset in ("knot", "cell_tracking"):
            assert run.ranking_metric == "teacher_mean_emd"
        else:
            assert run.ranking_metric == "teacher_heldout_emd"


def test_pretrain_presets_are_independently_usable_and_keep_experiment_metrics(tmp_path, sweep):
    specification, _ = specification_and_args(tmp_path, sweep)
    for alias, profile in specification["datasets"].items():
        assert profile["config"] == f"otpfm_{alias}.yaml"
        config = load_config(ROOT / "configs/pretrain" / profile["config"])
        validate(config)
        assert config["stage"] == "pretrain"
        assert config["model"]["name"] == "otpfm"
        assert config["logging"]["project"] == "flow2pc"
        original = load_config(ROOT / "configs/pretrain" / f"alicfm_{alias}.yaml")
        assert config["data"] == original["data"]
        evaluation = {**original["evaluation"], "protocol": "shared_ode"}
        if alias == "cell_tracking":
            evaluation["time_indices"] = [0, 24, 49, 74, 99, 114]
        assert config["evaluation"] == evaluation


def test_sweep_uses_preset_budget_unless_explicitly_overridden(tmp_path, sweep):
    specification, args = specification_and_args(tmp_path, sweep)
    args.runs_per_dataset = 1
    args.steps = None
    for run in sweep.build_runs(args, specification):
        preset = load_config(ROOT / "configs/pretrain" / specification["datasets"][run.dataset]["config"])
        assert run.config["training"]["steps"] == preset["training"]["steps"]
    args.steps = 200
    assert all(run.config["training"]["steps"] == 200 for run in sweep.build_runs(args, specification))


def test_native_training_and_optimizer_loops_are_inherited_without_overrides():
    assert MarginalTrainer.train is Trainer.train
    assert MarginalTrainer.train_epoch is Trainer.train_epoch
    assert MarginalTrainer.validate is Trainer.validate
    assert MarginalTrainer._setup_optimizer is Trainer._setup_optimizer


def test_six_frame_option_does_not_change_existing_default_contract():
    original = resolve_evaluation({})
    assert original == DEFAULTS
    assert "time_indices" not in original
    assert TRACKING_FRAMES == (0, 16, 32, 48, 65, 81, 97, 114)
    chosen = resolve_evaluation({"time_indices": [0, 24, 49, 74, 99, 114]})
    assert chosen["time_indices"] == [0, 24, 49, 74, 99, 114]
    with pytest.raises(ValueError, match="time_indices"):
        resolve_evaluation({"time_indices": [0, 24, 24, 114]})


def test_final_checkpoint_recovers_evaluation_without_retraining(tmp_path, sweep):
    specification, args = specification_and_args(tmp_path, sweep)
    args.runs_per_dataset = 1
    run = sweep.build_runs(args, specification)[0]
    (run.output / "checkpoints").mkdir(parents=True)
    (run.output / "resolved_config.yaml").write_text(yaml.safe_dump(run.config))
    (run.output / "checkpoints/last.pt").write_bytes(b"existing weights; never loaded")
    (run.output / "manifest.json").write_text(json.dumps({"method": "otpfm"}))
    (run.output / "preprocessing.json").write_text("{}")
    sweep.inspect_run(run)
    assert run.action == "evaluate" and run.status == "pending"
    with pytest.raises(ValueError, match="Refusing to archive"):
        sweep.archive_incomplete(run)
