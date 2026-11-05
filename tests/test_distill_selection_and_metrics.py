"""Distillation regressions. Execute only when the user authorizes tests."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flow2pc.distill import runner
from flow2pc.evaluation import metric_flow
from flow2pc.evaluation.ode import emd
from flow2pc.pretrain.models.alicfm.query_policy import resolve_sampler
from flow2pc.utils.checkpoint import restore_best_checkpoint
from flow2pc.utils.experiment import ExperimentConfig
from flow2pc.utils.experiment_protocols import validation_scope
from flow2pc.utils.mfm_config import validate_resume_training
from flow2pc.utils.tracking import metric_logger_config


def test_wandb_name_identifies_the_dataset_and_honors_explicit_name():
    config = {"training": {"lr": .002, "seed": 7},
              "student": {"num_latents": 128}, "wandb": {"project": "flow2pc"}}
    logger_config = metric_logger_config(config, method="mmfm", stage="distill", dataset="cite50")
    assert logger_config["logging"]["project"] == "flow2pc"
    assert logger_config["logging"]["name"] == "cite50-mmfm-hclt-h128-lr0.002-s7"
    config["wandb"]["run_name"] = "custom"
    assert metric_logger_config(config, method="mmfm", stage="distill")["logging"]["name"] == "custom"


def test_resume_preserves_selected_weights_in_new_directory(tmp_path):
    source = tmp_path / "original/checkpoints"
    source.mkdir(parents=True)
    selected = dict(identity={"teacher": "a"}, model_type="metric_pc_distill",
                    model_config={"name": "gaussian"}, step=2,
                    best_validation_loss=.1, state_dict={"weight": torch.tensor(2.)})
    last = {**selected, "step": 8, "state_dict": {"weight": torch.tensor(8.)}}
    torch.save(selected, source / "best.pt")
    torch.save(last, source / "last.pt")
    destination = tmp_path / "resumed"
    restore_best_checkpoint(last, source / "last.pt", destination)
    copied = torch.load(destination / "checkpoints/best.pt", weights_only=False)
    assert copied["step"] == 2
    assert copied["state_dict"]["weight"].item() == 2.


def test_resume_rejects_missing_best_and_changed_learning_rate(tmp_path):
    checkpoint = dict(step=5, best_validation_loss=.1,
                      training_config={"training": {"lr": .001}})
    with pytest.raises(ValueError, match="matching best.pt"):
        restore_best_checkpoint(checkpoint, tmp_path / "last.pt", tmp_path / "new")
    with pytest.raises(ValueError, match="lr changed"):
        validate_resume_training(checkpoint, {"lr": .002})


@pytest.mark.parametrize("dataset,scope", [
    ("cite50", "observed_rows"), ("multi5", "observed_rows"),
    ("st", "observed_rows"), ("cell_tracking", "observed_rows"),
    ("gom", "heldout_times"), ("beijing", "heldout_times"),
])
def test_validation_policy_is_dataset_specific(dataset, scope):
    data = SimpleNamespace(source_info={"profile": dataset})
    assert validation_scope(data, {}) == scope
    assert validation_scope(data, {"validation_scope": "observed_rows"}) == "observed_rows"


@pytest.mark.parametrize("coupling,expected", [
    ("border", "border_ot"), ("full", "border_ot"),
    ("mmot", "chain_ot"), ("none", "independent"),
])
def test_ali_coupling_uses_cfm_training_branch_over_stale_profile_tag(coupling, expected):
    manifest = {"model_config": {"profile": "single_cell", "cfm_ot": coupling},
                "distillation_sampler": "mmot_middle"}
    settings = {"name": "teacher_interpolant", "params": {}}
    assert resolve_sampler(manifest, settings) == expected
    if coupling != "none":
        with pytest.raises(ValueError, match="conflicts"):
            resolve_sampler(manifest, {**settings, "params": {"coupling": "none"}})


def test_emd_w1_and_w2_have_distinct_formulas_and_rematch_particles():
    real = torch.tensor([[0.], [2.]])
    assert emd(real, real.flip(0)) == pytest.approx(0.)
    predicted = torch.zeros(2, 1)
    assert emd(real, predicted) == pytest.approx(1.)
    assert emd(real, predicted, order=2) == pytest.approx(2**.5)


@pytest.mark.parametrize("profile,coupling,expected", [
    ("st", "full", "border_ot"), ("st", "mmot", "independent"),
    ("cell_tracking", "mmot", "mmot_middle"),
    ("cell_tracking", "none", "independent"),
    ("knot", "none", "independent"),
])
def test_ali_profile_specific_coupling(profile, coupling, expected):
    manifest = {"model_config": {"profile": profile, "cfm_ot": coupling}}
    assert resolve_sampler(manifest, {"name": "teacher_interpolant"}) == expected


class Student(torch.nn.Module):
    def __init__(self, *, broken_samples=False, broken_velocity=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.))
        self.broken_samples = broken_samples
        self.broken_velocity = broken_velocity

    def velocity(self, x, t):
        if self.broken_velocity:
            raise FloatingPointError("Nonfinite test velocity")
        return self.weight.expand_as(x)

    def sample(self, count, time):
        # Exercise all three RNG streams without leaking their states.
        random.random()
        np.random.random()
        values = torch.randn(count, 2)
        return values * float("nan") if self.broken_samples else values


def dataset():
    frames = [torch.zeros(8, 2), torch.ones(8, 2)]
    return SimpleNamespace(marginals=frames, train_marginals=frames,
        validation_marginals=frames, observed_indices=[0, 1], heldout_indices=[],
        times=torch.tensor([0., 1.]), source_info={}, normalization={}, split_indices={}, dim=2)


@pytest.mark.parametrize("broken_samples,broken_velocity", [(True, False), (False, True)])
def test_prediction_failures_are_isolated_and_rng_is_restored(monkeypatch, broken_samples, broken_velocity):
    # This test concerns orchestration; exact OT is tested separately above.
    monkeypatch.setattr(metric_flow, "emd", lambda real, pred, **kw:
                        float((real.mean(0) - pred.mean(0)).norm()))
    student = Student(broken_samples=broken_samples, broken_velocity=broken_velocity)
    teacher = SimpleNamespace(velocity=lambda x, t: torch.zeros_like(x))
    torch_state, numpy_state, python_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    report = metric_flow.evaluate(student, teacher, dataset(),
                                  {"samples": 4, "ode_substeps": 1}, "cpu")
    row = report["by_time"]["1"]
    assert "teacher_ode_emd" in row
    assert ("direct_pc_emd" in row) is not broken_samples
    assert ("student_ode_emd" in row) is not broken_velocity
    assert student.training
    assert torch.equal(torch_state, torch.get_rng_state())
    assert numpy_state[0] == np.random.get_state()[0]
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert numpy_state[2:] == np.random.get_state()[2:]
    assert python_state == random.getstate()


def test_final_evaluation_loads_best_student_instead_of_last(tmp_path, monkeypatch):
    output = tmp_path / "run"
    config = ExperimentConfig(
        teacher={"name": "metric_flow", "checkpoint_path": "flow.pt",
                 "options": {"geopath_checkpoint": "path.pt"}},
        dataset={"name": "arch"}, student={"pc_structure": "gaussian"},
        training={"steps": 3, "validate_every": 1, "batch_size": 4,
                  "device": "cpu", "output_dir": str(output),
                  "validation_population_metrics": False},
        evaluation={"enabled": True}, wandb={"mode": "disabled"}, task="distill")
    teacher = SimpleNamespace(model=torch.nn.Identity().eval(), metadata={},
        velocity=lambda x, t: torch.ones_like(x),
        path_sampler=SimpleNamespace(couple=lambda x, y: (x, y),
            sample=lambda x, y, *args: (torch.full((len(x),), .5), x)))
    monkeypatch.setattr(runner, "load_resources", lambda *args: (teacher, dataset()))
    monkeypatch.setattr(runner, "_identity", lambda *args: {})
    monkeypatch.setattr(runner, "student_config", lambda *args: {"name": "gaussian", "dim": 2})
    monkeypatch.setattr(runner, "build_student_from_config", lambda *args: Student())
    validation_scores = iter([.1, .4, .8])
    monkeypatch.setattr(runner, "validate_metric", lambda *args: next(validation_scores))
    seen = []
    def evaluate(student, *args):
        seen.append(student.weight.item())
        return {}
    monkeypatch.setattr(metric_flow, "evaluate", evaluate)
    runner.run(config)
    best = torch.load(output / "checkpoints/best.pt", weights_only=False)
    last = torch.load(output / "checkpoints/last.pt", weights_only=False)
    assert best["step"] == 1 and last["step"] == 3
    assert seen == [best["state_dict"]["weight"].item()]
    assert seen[0] != last["state_dict"]["weight"].item()
