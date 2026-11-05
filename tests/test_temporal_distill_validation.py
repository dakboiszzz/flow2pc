"""Temporal validation regressions; run only when the user authorizes tests."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from flow2pc.distill import validation
from flow2pc.distill.sampling import DistillationBatch


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.calls = []

    def velocity(self, x, t):
        assert not torch.is_grad_enabled()
        self.calls.append((x.clone(), t.clone()))
        return x + 2 * t[:, None]


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(1.0))

    def velocity(self, x, t):
        assert not torch.is_grad_enabled()
        error = torch.where(t < 0.5, self.offset, 3 * self.offset)
        return x + 2 * t[:, None] + error[:, None]


class TemporalDataset:
    def __init__(self, sizes=(2, 3)):
        self.metadata = SimpleNamespace(
            name="beijing", model_times=(0.0, 0.25, 0.5, 0.75, 1.0),
            physical_times=(0, 6, 12, 18, 24),
        )
        self.observed = [0, 2, 4]
        self.heldout_indices = [1, 3]
        self.values = {
            index: torch.arange(size, dtype=torch.float32)[:, None] + index * 100
            for index, size in zip(self.heldout_indices, sizes, strict=True)
        }
        self.requests = []

    def get_marginal(self, index, split):
        self.requests.append((index, split))
        assert split == "test" and index in self.heldout_indices
        return SimpleNamespace(materialize=lambda: self.values[index])


def empty_validation_pools(dataset):
    return [torch.empty(0, 1) for _ in dataset.metadata.model_times]


def test_temporal_validation_uses_only_heldout_cells_at_model_times(monkeypatch):
    dataset, teacher = TemporalDataset(), Teacher()
    pools = empty_validation_pools(dataset)
    monkeypatch.setattr(
        validation, "sample_ali_batch",
        lambda *args: pytest.fail("Empty row-validation pools must not be sampled"),
    )
    groups, _ = validation.prepare_validation(
        teacher, dataset, pools, {}, 123, "mmot_middle"
    )
    assert set(groups) == {"heldout_time_1", "heldout_time_3"}
    assert dataset.requests == [(1, "test"), (3, "test")]
    for index in dataset.heldout_indices:
        x, times, target = groups[f"heldout_time_{index}"]
        assert torch.equal(x, dataset.values[index])
        assert torch.equal(times, torch.full((len(x),), index / 4))
        assert torch.equal(target, x + 2 * times[:, None])
        assert not target.requires_grad
    assert all(len(pool) == 0 for pool in pools)
    assert dataset.observed == [0, 2, 4]


@pytest.mark.parametrize("training", [True, False])
def test_selection_weights_times_equally_and_preserves_student_state(training):
    dataset, teacher, student = TemporalDataset(), Teacher(), Student()
    student.train(training)
    groups, energy = validation.prepare_validation(
        teacher, dataset, empty_validation_pools(dataset), {}, 123, "mmot_middle"
    )
    scores = validation.validate(student, groups, energy, batch_size=2)
    assert scores["heldout_time_1_velocity_mse"] == pytest.approx(1.0)
    assert scores["heldout_time_3_velocity_mse"] == pytest.approx(9.0)
    # Equal-time mean is 5, rather than cell-weighted (2 * 1 + 3 * 9) / 5.
    assert scores["selection_score"] == pytest.approx(5.0)
    assert scores["heldout_time_velocity_mse"] == pytest.approx(5.0)
    assert student.training is training
    assert student.offset.grad is None


def test_temporal_subsets_are_fixed_seeded_and_teacher_queries_are_batched():
    dataset = TemporalDataset(sizes=(20, 30))
    settings = {"validation_batches": 2, "validation_batch_size": 3}
    teacher = Teacher()
    first, _ = validation.prepare_validation(
        teacher, dataset, empty_validation_pools(dataset), settings, 123, "mmot_middle"
    )
    second, _ = validation.prepare_validation(
        Teacher(), dataset, empty_validation_pools(dataset), settings, 123,
        "mmot_middle",
    )
    assert all(len(x) <= 3 for x, _ in teacher.calls)
    for name in first:
        assert len(first[name][0]) == 6
        assert len(torch.unique(first[name][0])) == 6
        assert torch.equal(first[name][0], second[name][0])
    contract = validation.temporal_validation_contract(dataset, settings)
    assert contract["time_indices"] == [1, 3]
    assert contract["max_cells_per_time"] == 6


def test_row_diagnostic_does_not_override_temporal_selection(monkeypatch):
    dataset, teacher = TemporalDataset(), Teacher()
    pools = [torch.ones(2, 1) for _ in dataset.metadata.model_times]
    monkeypatch.setattr(
        validation, "sample_ali_batch",
        lambda *args: DistillationBatch(torch.ones(2, 1), torch.full((2,), 0.9)),
    )
    groups, energy = validation.prepare_validation(
        teacher, dataset, pools, {}, 123, "mmot_middle"
    )
    scores = validation.validate(Student(), groups, energy)
    assert scores["global_velocity_mse"] == pytest.approx(9.0)
    assert scores["selection_score"] == pytest.approx(5.0)


def test_no_temporal_holdouts_preserves_row_validation(monkeypatch):
    dataset, teacher = TemporalDataset(), Teacher()
    dataset.heldout_indices = []
    dataset.metadata.name = "knot"
    dataset.observed = list(range(5))
    pools = [torch.ones(2, 1) for _ in dataset.metadata.model_times]
    monkeypatch.setattr(
        validation, "sample_ali_batch",
        lambda *args: DistillationBatch(torch.ones(2, 1), torch.full((2,), 0.9)),
    )
    groups, energy = validation.prepare_validation(
        teacher, dataset, pools, {}, 123, "independent"
    )
    assert set(groups) == {"global"}
    assert dataset.requests == []
    assert validation.temporal_validation_contract(dataset, {}) is None
    scores = validation.validate(Student(), groups, energy)
    assert scores["selection_score"] == pytest.approx(9.0)


def test_tracking_late_time_selection_is_preserved_without_temporal_holdouts():
    groups = {}
    for name, time in (("global", 0.9), ("late", 0.1), ("final", 0.9)):
        values = torch.ones(2, 1)
        times = torch.full((2,), time)
        groups[name] = (values, times, values + 2 * times[:, None])
    scores = validation.validate(Student(), groups, energy=2.0)
    assert scores["selection_score"] == pytest.approx(9.0 / 2.0)


def test_empty_heldout_population_fails_without_training_data_fallback():
    dataset = TemporalDataset(sizes=(0, 3))
    with pytest.raises(ValueError, match="Temporal validation population 1 is empty"):
        validation.prepare_validation(
            Teacher(), dataset, empty_validation_pools(dataset), {}, 123, "mmot_middle"
        )


def test_temporal_holdouts_cannot_overlap_observed_training_times():
    dataset = TemporalDataset()
    dataset.heldout_indices = [1, 2]
    with pytest.raises(ValueError, match="unique held-out times"):
        validation.temporal_validation_contract(dataset, {})
