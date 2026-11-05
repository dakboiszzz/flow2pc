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
from flow2pc.utils.experiment import ExperimentConfig


def test_runner_dispatches_existing_config_contracts(monkeypatch):
    ali_config = {"stage": "distill"}
    mfm_config = ExperimentConfig(teacher={}, dataset={})
    monkeypatch.setattr(runner, "_run_ali", lambda config: ("ali", config))
    monkeypatch.setattr(runner, "_run_metric", lambda config: ("mfm", config))
    assert runner.run(ali_config) == ("ali", ali_config)
    assert runner.run(mfm_config) == ("mfm", mfm_config)
    with pytest.raises(TypeError, match="Unsupported distillation config type"):
        runner.run(None)


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
        model=torch.nn.Identity().eval(),
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
    query = sampling.sample_mfm_batch(teacher, data, 5, "cpu")
    actual = sampling.velocity_mse(student, teacher, query)
    torch.testing.assert_close(actual, expected)
    assert draws == [5, 5]
    torch.testing.assert_close(query.x_t, samples)
    torch.testing.assert_close(query.model_t, times)
    torch.testing.assert_close(query.target_velocity, torch.ones_like(samples))
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
        model=torch.nn.Identity().eval(),
        metadata={}, velocity=lambda x, t: torch.ones_like(x),
        path_sampler=SimpleNamespace(couple=lambda x, y: (x, y),
                                     sample=lambda x, y, *args: (torch.full((len(x),), .5), x)),
    )
    monkeypatch.setattr(runner, "load_resources", lambda *args: (teacher, data))
    monkeypatch.setattr(runner, "_identity", lambda *args: {})
    monkeypatch.setattr(runner, "student_config", lambda *args: {
        "name": "gaussian", "dim": 2, "num_components": 3, "time_order": 2,
    })
    runner.run(config)
    checkpoint = torch.load(tmp_path / "checkpoints/last.pt", weights_only=False)
    assert "source_nll_weight" not in checkpoint
    assert "source_nll_weight" not in checkpoint["training_config"]["training"]
    assert (tmp_path / "metrics.csv").read_text().splitlines()[0] == ",".join(runner.METRIC_FIELDS)


def test_ali_sampler_keeps_independent_endpoints_and_time_draws():
    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
            self.eval()

        def interpolate(self, x0, x1, times):
            return x0 * (1 - times[:, None]) + x1 * times[:, None]

        def velocity(self, x, times, *, context=None):
            return torch.ones_like(x)

    teacher = Teacher()
    pools = [torch.arange(12, dtype=torch.float32).reshape(6, 2),
             torch.arange(12, 24, dtype=torch.float32).reshape(6, 2)]
    dataset = SimpleNamespace(metadata=SimpleNamespace(name="cite"), observed=[0, 1])
    rng = torch.Generator().manual_seed(37)
    query = sampling.sample_ali_batch(teacher, dataset, pools, 4, rng, "independent")
    replay = torch.Generator().manual_seed(37)
    x0 = pools[0][torch.randint(6, (4,), generator=replay)]
    x1 = pools[1][torch.randint(6, (4,), generator=replay)]
    times = torch.rand(4, generator=replay)
    torch.testing.assert_close(query.x_t, teacher.interpolate(x0, x1, times))
    torch.testing.assert_close(query.model_t, times)
    assert query.target_velocity is None
    student = TimeDependentGaussianPC(3, 2, 2, polynomial_zero_center=False)
    expected = (student.velocity(query.x_t, times) - 1).square().mean()
    torch.testing.assert_close(sampling.velocity_mse(student, teacher, query), expected)


def test_shared_loss_rejects_unfrozen_mfm_teacher():
    student = TimeDependentGaussianPC(3, 2, 2, polynomial_zero_center=False)
    teacher = SimpleNamespace(model=torch.nn.Linear(2, 2),
                              velocity=lambda x, t: torch.zeros_like(x))
    query = sampling.DistillationBatch(torch.zeros(3, 2), torch.zeros(3),
                                       target_velocity=torch.zeros(3, 2))
    with pytest.raises(ValueError, match="frozen"):
        sampling.velocity_mse(student, teacher, query)


def test_mfm_sampler_uses_adjacent_observed_marginals_across_holdout(monkeypatch):
    source = torch.zeros(6, 2)
    heldout = torch.full((6, 2), 99.)
    target = torch.ones(6, 2)
    calls = []
    def couple(x0, x1):
        calls.append((x0.clone(), x1.clone()))
        return x0, x1
    def sample(x0, x1, left, right):
        assert (left, right) == (0., 1.)
        return torch.full((len(x0),), .5), (x0 + x1) / 2
    teacher = SimpleNamespace(
        model=torch.nn.Identity().eval(),
        path_sampler=SimpleNamespace(couple=couple, sample=sample),
        velocity=lambda values, times: torch.ones_like(values),
    )
    data = SimpleNamespace(
        validation_marginals=[source, heldout, target],
        observed_indices=[0, 2], times=torch.tensor([0., .5, 1.]),
    )
    monkeypatch.setattr(sampling, "_draw", lambda values, count, device: values[:count])
    batch = sampling.sample_mfm_batch(
        teacher, data, 4, "cpu", validation=True, interval_index=0
    )
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], source[:4])
    torch.testing.assert_close(calls[0][1], target[:4])
    torch.testing.assert_close(batch.x_t, torch.full((4, 2), .5))
    torch.testing.assert_close(batch.target_velocity, torch.ones(4, 2))
