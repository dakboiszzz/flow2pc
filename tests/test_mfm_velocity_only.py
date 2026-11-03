"""Contracts for shared student construction and velocity-only MFM artifacts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from flow2pc.distill import runner, sampling
from flow2pc.distill.student import build_student_from_config
from flow2pc.pc.gaussian_pc import TimeDependentGaussianPC
from flow2pc.pc.hclt_pc import TimeDependentHCLTPC
from flow2pc.utils.mfm_config import checkpoint_training_config


@pytest.mark.parametrize("kind", ["hclt", "gaussian"])
def test_factory_restores_original_model_layout_and_outputs(kind):
    cfg = dict(name=kind, dim=2, parents=[-1, 0], num_latents=4,
               num_components=4, time_order=3, time_embedding_dim=6,
               time_feature_seed=9)
    if kind == "hclt":
        original = TimeDependentHCLTPC(
            cfg["parents"], num_latents=4, time_order=3, sum_layer="dense",
            time_conditioning="fourier", time_embedding_dim=6,
            time_feature_seed=9, fourier_zero_center=False,
        )
    else:
        original = TimeDependentGaussianPC(4, 2, 3, polynomial_zero_center=False)
    restored = build_student_from_config(cfg)
    restored.load_state_dict(original.state_dict(), strict=True)
    assert runner.build_student_from_config is build_student_from_config
    x, t = torch.randn(8, 2), torch.linspace(0, 1, 8)
    torch.testing.assert_close(restored.velocity(x, t), original.velocity(x, t), rtol=0, atol=0)
    torch.testing.assert_close(restored.log_prob(x, t), original.log_prob(x, t), rtol=0, atol=0)


def test_loss_never_reads_source_density_or_draws_extra_batch(monkeypatch):
    student = TimeDependentGaussianPC(3, 2, 2, polynomial_zero_center=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("Density must not be queried")
    monkeypatch.setattr(student, "log_prob", forbidden)
    samples, times = torch.randn(5, 2), torch.linspace(0.1, 0.9, 5)
    teacher = SimpleNamespace(
        path_sampler=SimpleNamespace(couple=lambda x, y: (x, y),
                                     sample=lambda *args: (times, samples)),
        velocity=lambda x, t: torch.ones_like(x),
    )
    frames = [torch.randn(10, 2), torch.randn(10, 2)]
    data = SimpleNamespace(train_marginals=frames, validation_marginals=frames,
                           observed_indices=[0, 1], times=torch.tensor([0., 1.]))
    draws = []
    def draw(values, count, device):
        draws.append(count)
        return values[:count]
    monkeypatch.setattr(sampling, "_draw", draw)
    expected = (student.velocity(samples, times) - 1).square().mean()
    actual, components = sampling.mfm_distillation_loss(student, teacher, data, 5, "cpu")
    torch.testing.assert_close(actual, expected)
    assert draws == [5, 5]
    assert set(components) == {"velocity_mse"}
    actual.backward()
    grads = [p.grad for p in student.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


@pytest.mark.parametrize("weight", [0., 1., float("nan")])
def test_legacy_normalization_preserves_input_and_checks_resume(weight):
    checkpoint = {"training_config": {"training": {"lr": .001, "source_nll_weight": weight}},
                  "source_nll_weight": weight}
    config = checkpoint_training_config(checkpoint)
    assert config == {"training": {"lr": .001}}
    assert "source_nll_weight" in checkpoint["training_config"]["training"]
    if weight == 0:
        assert checkpoint_training_config(checkpoint, resume=True) == config
    else:
        with pytest.raises(ValueError, match="nonzero"):
            checkpoint_training_config(checkpoint, resume=True)


def test_new_checkpoint_config_needs_no_legacy_fields():
    checkpoint = {"training_config": {"training": {"lr": .001}}}
    original = deepcopy(checkpoint)
    assert checkpoint_training_config(checkpoint, resume=True) == original["training_config"]
    assert checkpoint == original


def test_legacy_csv_is_preserved_and_new_schema_checked(tmp_path):
    old = tmp_path / "metrics.csv"
    content = "step,loss,velocity_mse,source_nll_per_dimension,gradient_norm,validation_loss\n1,2,2,3,4,5\n"
    old.write_text(content)
    target = runner.metrics_path(tmp_path)
    assert target.name == "metrics_velocity.csv"
    assert old.read_text() == content
    target.write_text(",".join(runner.METRIC_FIELDS) + "\n")
    assert runner.metrics_path(tmp_path) == target
    target.write_text("invalid,header\n")
    with pytest.raises(ValueError, match="header"):
        runner.metrics_path(tmp_path)


def test_new_csv_uses_standard_filename(tmp_path):
    assert runner.metrics_path(tmp_path) == tmp_path / "metrics.csv"


def test_evaluation_does_not_query_density(monkeypatch):
    from flow2pc.evaluation.metric_flow import evaluate

    student = TimeDependentGaussianPC(3, 2, 2, polynomial_zero_center=False)
    def forbidden(*args, **kwargs):
        raise AssertionError("Evaluation must not query NLL")
    monkeypatch.setattr(student, "log_prob", forbidden)
    teacher = SimpleNamespace(velocity=lambda x, t: torch.zeros_like(x))
    data = SimpleNamespace(
        validation_marginals=[torch.randn(8, 2), torch.randn(8, 2)],
        times=torch.tensor([0., 1.]), heldout_indices=[], normalization={},
    )
    report = evaluate(student, teacher, data,
                      {"samples": 4, "projections": 2, "ode_substeps": 1},
                      torch.device("cpu"))
    assert all("nll_per_dimension" not in row for row in report["by_time"].values())


def test_new_run_emits_only_velocity_metadata(tmp_path, monkeypatch):
    from flow2pc.utils.experiment import ExperimentConfig

    config = ExperimentConfig(
        teacher={"name": "metric_flow", "checkpoint_path": "flow.ckpt",
                 "options": {"geopath_checkpoint": "geo.ckpt"}},
        dataset={"name": "arch"}, student={"pc_structure": "gaussian"},
        training={"steps": 1, "batch_size": 4, "device": "cpu", "output_dir": str(tmp_path)},
        evaluation={"enabled": False}, wandb={"mode": "disabled"}, task="distill",
    )
    frames = [torch.randn(8, 2), torch.randn(8, 2)]
    data = SimpleNamespace(
        train_marginals=frames, validation_marginals=frames,
        observed_indices=[0, 1], times=torch.tensor([0., 1.]),
        source_info={}, normalization={}, split_indices={},
    )
    teacher = SimpleNamespace(
        metadata={}, velocity=lambda x, t: torch.ones_like(x),
        path_sampler=SimpleNamespace(couple=lambda x, y: (x, y),
                                     sample=lambda x, y, *args: (torch.full((len(x),), .5), x)),
    )
    monkeypatch.setattr(runner, "load_resources", lambda *args: (teacher, data))
    monkeypatch.setattr(runner, "_identity", lambda *args: {})
    monkeypatch.setattr(runner, "student_config", lambda *args: {
        "name": "gaussian", "dim": 2, "num_components": 3, "time_order": 2,
    })
    runner.run_mfm(config)
    checkpoint = torch.load(tmp_path / "checkpoints/last.pt", weights_only=False)
    assert "source_nll_weight" not in checkpoint
    assert "source_nll_weight" not in checkpoint["training_config"]["training"]
    assert (tmp_path / "metrics.csv").read_text().splitlines()[0] == ",".join(runner.METRIC_FIELDS)
