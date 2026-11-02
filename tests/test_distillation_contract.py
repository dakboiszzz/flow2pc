"""Regression coverage for the recovered student boundary; not run automatically."""

import torch
from torch import nn

from flow2pc.data.restore import restore_dataset
from flow2pc.data.temporal import SpatialTranscriptomics
from flow2pc.distill.sampling import DistillationBatch, velocity_mse
from flow2pc.distill.student import build_student, initialize_source
from flow2pc.pretrain.models.alicfm.adapter import FrozenALICFM


class LinearInterpolant(nn.Module):
    def forward(self, x0, x1, t, training=False):
        return (1 - t) * x0 + t * x1


def test_student_optimizes_exactly_velocity_mse_with_frozen_teacher(monkeypatch):
    torch.manual_seed(8)
    teacher = FrozenALICFM(
        nn.Linear(3, 2),
        LinearInterpolant(),
        {"model_config": {"dim": 2}},
        {"shift": [0.0, 0.0], "scale": [1.0, 1.0]},
    )
    student = build_student([-1, 0], {"num_latents": 4, "time_embedding_dim": 4})
    source = torch.randn(12, 2)
    initialize_source(student, source, torch.Generator().manual_seed(3))

    def forbidden_sample(*args, **kwargs):
        raise AssertionError("The objective must not sample from the PC")

    monkeypatch.setattr(student, "sample", forbidden_sample)
    batch = DistillationBatch(torch.randn(4, 2), torch.tensor([0.1, 0.3, 0.6, 0.9]))
    saved = {name: value.clone() for name, value in teacher.state_dict().items()}
    expected = nn.functional.mse_loss(
        student.velocity(batch.x_t, batch.model_t),
        teacher.velocity(batch.x_t, batch.model_t),
    )
    actual = velocity_mse(student, teacher, batch)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters()
    )
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


def test_st_saved_preprocessing_and_splits_are_restored_without_refit(tmp_path):
    import numpy as np

    directory = tmp_path / "spatial_transcriptomics/aligned/aligned_spots"
    directory.mkdir(parents=True)
    for index, slide in enumerate(range(2, 6)):
        values = np.arange(40, dtype=np.float32).reshape(20, 2) + index
        if slide == 3:
            values += 100_000  # Held-out U3 must not influence the fitted range.
        np.savetxt(directory / f"U{slide}_tumor_coordinates.csv", values, delimiter=",")
    dataset = SpatialTranscriptomics(
        name="st",
        root=tmp_path,
        seed=42,
        params={"heldout_index": 1, "normalise": True, "whiten": False},
    )
    assert dataset.preprocessing["fit_time_indices"] == [0, 2, 3]
    assert dataset.get_marginal(1, "test").materialize().min() > 1000
    restored = restore_dataset(dataset.contract(), dataset.preprocessing)
    assert restored.contract() == dataset.contract()
    for index in dataset.observed:
        torch.testing.assert_close(
            restored.get_marginal(index, "train").materialize(),
            dataset.get_marginal(index, "train").materialize(),
            rtol=0,
            atol=0,
        )
