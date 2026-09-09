import sys
import tempfile
import unittest
from pathlib import Path

# Ensure local repository root takes precedence over any installed site-packages
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path or sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from torch import nn

from repc.data import (
    MultiMarginalDataset,
    generate_knot_distribution,
    multimarginal_loader,
)
from repc.flow import ALICFMTeacherWrapper, build_teacher
from repc.hclt_pc import TimeDependentHCLTPC
from repc.pc import TimeDependentGaussianPC
from repc.region_pc import RegionGraphPC
from train import (
    ConditionalFlowMatcher,
    distillation_loss,
    load_teacher,
    parse_args,
    sample_distillation_timesteps,
)


class MultiMarginalDataTest(unittest.TestCase):
    def test_generate_knot_distribution_shape_and_validation(self):
        with self.assertRaises(ValueError):
            generate_knot_distribution(K=10)  # not multiple of 3

        knot_data = generate_knot_distribution(K=60, samples_per_marginal=10, seed=42)
        self.assertEqual(knot_data.shape, (60, 10, 2))
        self.assertEqual(knot_data.dtype, torch.float32)
        self.assertTrue(torch.all(torch.isfinite(knot_data)))

        # Segment 1 (k=0): t=0 -> t_tilde = -1.5 -> mu_x1 = 3*(-1.5 + 0.5) = -3.0
        # Check empirical mean around k=0 is close to -3.0
        self.assertAlmostEqual(knot_data[0, :, 0].mean().item(), -3.0, delta=0.2)

    def test_multi_marginal_dataset_and_loader(self):
        K, N, D = 6, 20, 2
        data = torch.randn(K, N, D)
        dataset = MultiMarginalDataset(data)

        self.assertEqual(len(dataset), K * N)
        point, k_idx = dataset[25]  # k = 25 // 20 = 1, n = 5
        self.assertEqual(k_idx.item(), 1)
        self.assertTrue(torch.allclose(point, data[1, 5]))

        loader = multimarginal_loader("knot", batch_size=16, K=60, samples_per_marginal=10)
        batch = next(iter(loader))
        points, k_indices = batch
        self.assertEqual(points.shape, (16, 2))
        self.assertEqual(k_indices.shape, (16,))
        self.assertTrue(torch.all((k_indices >= 0) & (k_indices < 60)))


class ALICFMTeacherTest(unittest.TestCase):
    def test_alicfm_teacher_wrapper_and_interval_mapping(self):
        # Create a simple mock MLP matching ALI-CFM architecture: 2D space + 1D time -> 2D velocity
        in_dim = 2 + 1
        out_dim = 2
        mock_mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.SELU(),
            nn.Linear(32, out_dim),
        )

        teacher = ALICFMTeacherWrapper(mock_mlp, num_marginals=60)

        # Call with global times and path samples
        batch_size = 8
        times = torch.linspace(0.0, 1.0, batch_size)
        path_samples = torch.randn(batch_size, 2)

        velocity = teacher(times, path_samples)
        self.assertEqual(velocity.shape, (batch_size, 2))
        self.assertTrue(torch.all(torch.isfinite(velocity)))

    def test_build_teacher_alicfm_checkpoint(self):
        in_dim = 3
        out_dim = 2
        mock_mlp = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.SELU(),
            nn.Linear(16, out_dim),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "alicfm_model.pt"
            torch.save({"ot_cfm_model": mock_mlp.state_dict()}, ckpt_path)

            teacher = build_teacher({"name": "alicfm", "checkpoint_path": ckpt_path, "num_marginals": 30})
            self.assertIsInstance(teacher, ALICFMTeacherWrapper)
            self.assertEqual(teacher.num_marginals, 30)

            v = teacher(torch.tensor([0.2, 0.5]), torch.randn(2, 2))
            self.assertEqual(v.shape, (2, 2))

    def test_load_teacher_alicfm_dispatch(self):
        in_dim = 3
        out_dim = 2
        mock_mlp = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.SELU(),
            nn.Linear(16, out_dim),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "alicfm_model.pt"
            torch.save(mock_mlp, ckpt_path)

            loaded_teacher, flow_config = load_teacher(
                ckpt_path,
                torch.device("cpu"),
                teacher_arch="alicfm",
            )
            self.assertIsInstance(loaded_teacher, ALICFMTeacherWrapper)
            self.assertEqual(flow_config["name"], "vanilla_cfm")


class ZeroCenteringTest(unittest.TestCase):
    def test_fourier_zero_centering_in_hclt(self):
        parents = [-1, 0, 1]  # 3 variables
        # Zero centered: cos(angles) - 1.0 -> 0.0 at t=0
        hclt_centered = TimeDependentHCLTPC(
            parents=parents,
            num_latents=4,
            time_conditioning="fourier",
            time_embedding_dim=8,
            fourier_zero_center=True,
        )
        t_zero = torch.tensor([0.0])
        basis_centered, _ = hclt_centered._time_basis(t_zero)
        self.assertTrue(torch.allclose(basis_centered, torch.zeros_like(basis_centered)))

        # Uncentered: cos(0) = 1.0
        hclt_uncentered = TimeDependentHCLTPC(
            parents=parents,
            num_latents=4,
            time_conditioning="fourier",
            time_embedding_dim=8,
            fourier_zero_center=False,
        )
        basis_uncentered, _ = hclt_uncentered._time_basis(t_zero)
        # Check that cosine terms (even columns) are 1.0
        self.assertTrue(torch.any(basis_uncentered != 0.0))

    def test_polynomial_zero_centering_in_gaussian_pc(self):
        pc_centered = TimeDependentGaussianPC(
            num_components=2,
            dim=2,
            time_order=3,
            polynomial_zero_center=True,
        )
        # Shape has time_order=3 features
        self.assertEqual(pc_centered.mean_coefficients.shape, (2, 2, 3))
        t_zero = torch.tensor([0.0])
        basis_c, dt_c = pc_centered._time_basis(t_zero)
        self.assertEqual(basis_c.shape, (1, 3))
        self.assertTrue(torch.all(torch.isfinite(dt_c)))

        pc_uncentered = TimeDependentGaussianPC(
            num_components=2,
            dim=2,
            time_order=3,
            polynomial_zero_center=False,
        )
        # Shape has time_order + 1 = 4 features
        self.assertEqual(pc_uncentered.mean_coefficients.shape, (2, 2, 4))
        basis_u, dt_u = pc_uncentered._time_basis(t_zero)
        self.assertEqual(basis_u.shape, (1, 4))
        # Power 0 term at t=0 is 1.0
        self.assertAlmostEqual(basis_u[0, 0].item(), 1.0)
        # Power 0 derivative is 0.0, no NaNs
        self.assertAlmostEqual(dt_u[0, 0].item(), 0.0)
        self.assertTrue(torch.all(torch.isfinite(dt_u)))

    def test_polynomial_zero_centering_in_region_pc(self):
        rpc_centered = RegionGraphPC(
            height=4,
            width=4,
            num_channels=2,
            time_order=2,
            polynomial_zero_center=True,
        )
        self.assertEqual(rpc_centered.mean_coefficients.shape[-1], 2)

        rpc_uncentered = RegionGraphPC(
            height=4,
            width=4,
            num_channels=2,
            time_order=2,
            polynomial_zero_center=False,
        )
        self.assertEqual(rpc_uncentered.mean_coefficients.shape[-1], 3)
        basis, dt = rpc_uncentered._time_basis(torch.tensor([0.0]))
        self.assertTrue(torch.all(torch.isfinite(dt)))


class DistillationTimestepsAndLossTest(unittest.TestCase):
    def test_interval_bound_timestep_sampling(self):
        K_total = 60
        k_indices = torch.tensor([0, 10, 25, 58, 59])
        count = len(k_indices)

        times = sample_distillation_timesteps(
            count=count,
            device=torch.device("cpu"),
            dtype=torch.float32,
            sampling="multimarginal_interval",
            k_indices=k_indices,
            K_total=K_total,
        )

        for i, k in enumerate(k_indices):
            k_clamped = min(int(k), K_total - 2)
            t_low = k_clamped / (K_total - 1)
            t_high = (k_clamped + 1) / (K_total - 1)
            self.assertGreaterEqual(times[i].item(), t_low - 1e-6)
            self.assertLessEqual(times[i].item(), t_high + 1e-6)

    def test_end_to_end_distillation_loss_on_knot(self):
        torch.manual_seed(0)
        student = TimeDependentGaussianPC(
            num_components=2,
            dim=2,
            time_order=2,
            polynomial_zero_center=False,
        )
        mock_teacher_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.SELU(),
            nn.Linear(16, 2),
        )
        teacher = ALICFMTeacherWrapper(mock_teacher_mlp, num_marginals=60)
        flow_matcher = ConditionalFlowMatcher(sigma=0.0)

        data = torch.randn(4, 2)
        k_indices = torch.tensor([0, 5, 10, 15])

        loss = distillation_loss(
            student,
            teacher,
            flow_matcher,
            data,
            timestep_sampling="multimarginal_interval",
            k_indices=k_indices,
            K_total=60,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(student.mean_coefficients.grad)


if __name__ == "__main__":
    unittest.main()
