"""Recovered evaluation behavior. These definitions require explicit test execution."""

import numpy as np
import torch
from torch import nn

from flow2pc.data.temporal import SpatialTranscriptomics
from flow2pc.evaluation.config import resolve_evaluation
from flow2pc.evaluation.ode import emd, rollout
from flow2pc.evaluation.runner import evaluate


class ConstantField(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            "anchor", nn.Parameter(torch.zeros(1), requires_grad=False)
        )
        self.manifest = {"method": "fixture", "checkpoint_sha256": "constant-field"}

    def velocity(self, values, time):
        return torch.ones_like(values)

    def ode_velocity(self, time, values):
        return self.velocity(values, time)


def test_emd_uses_population_matching_and_euclidean_cost():
    cloud = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert emd(cloud, cloud.flip(0)) == 0
    assert abs(emd(cloud, cloud + torch.tensor([0.0, 2.0])) - 2.0) < 1e-10


def test_teacher_student_rollouts_use_identical_source_particles_and_times():
    model = ConstantField()
    source, times = torch.randn(5, 2), torch.tensor([0.0, 1 / 3, 2 / 3, 1.0])
    settings = resolve_evaluation({"solver": "rk4", "batch_size": 2, "ode_steps": 7})
    teacher, nfe = rollout(model, source, times, settings)
    student, student_nfe = rollout(model, source, times, settings, student=True)
    torch.testing.assert_close(teacher, source[None] + times[:, None, None])
    torch.testing.assert_close(student, teacher, rtol=0, atol=0)
    assert nfe == student_nfe


def test_st_evaluation_saves_clean_he_comparison_and_teacher_reference(tmp_path):
    from PIL import Image

    directory = tmp_path / "spatial_transcriptomics/aligned"
    (directory / "aligned_spots").mkdir(parents=True)
    (directory / "ref_U5_warped_images").mkdir()
    for slide in range(2, 6):
        cloud = np.tile([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], (3, 1))
        np.savetxt(
            directory / f"aligned_spots/U{slide}_tumor_coordinates.csv",
            cloud,
            delimiter=",",
        )
        Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8)).save(
            directory / f"ref_U5_warped_images/U{slide}_aligned_to_U5.tif"
        )
    dataset = SpatialTranscriptomics(
        name="st", root=tmp_path, seed=3, params={"heldout_index": 1}
    )
    settings = resolve_evaluation(
        {
            "enabled": True,
            "solver": "rk4",
            "num_samples": 4,
            "metric_samples": 4,
            "ode_steps": 4,
        }
    )
    teacher = ConstantField()
    result = evaluate(teacher, dataset, settings, tmp_path / "teacher/eval")
    comparison = evaluate(
        teacher,
        dataset,
        settings,
        tmp_path / "student/eval",
        student=ConstantField(),
        reference_dir=tmp_path / "teacher/eval",
    )
    assert result["teacher_heldout_emd"] == comparison["teacher_heldout_emd"]
    assert comparison["student_heldout_emd"] == comparison["teacher_heldout_emd"]
    assert comparison["teacher_student_max_paired_rmse_raw"] == 0
    for name in (
        "st_he_overlay.png",
        "teacher_overlay.png",
        "student_overlay.png",
        "metrics.json",
        "media.json",
        "marginal_metrics.csv",
        "predictions.pt",
    ):
        assert (tmp_path / "student/eval" / name).is_file()


def test_whitened_single_cell_figure_accepts_empty_heldout_splits(tmp_path):
    import h5py

    from flow2pc.data.real.single_cell import SingleCell
    from flow2pc.evaluation.figures import save_figures

    directory = tmp_path / "single_cell"
    directory.mkdir()
    with h5py.File(directory / "op_cite_inputs_0.h5ad", "w") as handle:
        handle.create_dataset("obsm/X_pca", data=np.arange(240, dtype=np.float32).reshape(48, 5))
        handle.create_dataset("obs/day", data=np.repeat([2, 3, 4, 7], 12))
    dataset = SingleCell(
        name="cite", root=tmp_path, seed=42,
        params={"dim": 5, "heldout_index": 1, "normalise": True,
                "whiten": True, "preprocessing_protocol": "native_all"},
    )
    for split in ("train", "val"):
        empty = dataset.get_marginal(1, split).materialize()
        assert empty.shape == (0, 5)
        assert empty.dtype == torch.float32
    paths = torch.stack([
        dataset.get_marginal(index, "all").materialize()[:8]
        for index in range(4)
    ])
    files = save_figures(
        dataset, paths, None, tmp_path, {"metric_space": "historical"}
    )
    assert files["teacher_marginals"].is_file()
