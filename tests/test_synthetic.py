import math
import unittest

import torch

from repc.pc import TimeDependentGaussianPC
from repc.synthetic import sample_path_times, standard_branching_flow


class BranchingGaussianFlowTest(unittest.TestCase):
    def test_path_starts_at_exact_standard_normal(self):
        source = standard_branching_flow()
        values = torch.tensor([[0.0, 0.0], [1.0, -2.0], [-0.5, 0.25]])
        log_probability = source.log_prob(values, torch.tensor(0.0))
        expected = -0.5 * (
            values.square().sum(dim=1) + values.shape[1] * math.log(2.0 * math.pi)
        )
        torch.testing.assert_close(log_probability, expected)

    def test_scalar_and_batched_time_agree(self):
        source = standard_branching_flow()
        values = torch.tensor([[0.2, -0.3], [1.0, 0.5], [-0.7, 1.1]])
        scalar_time = torch.tensor(0.4)
        batched_time = torch.full((len(values),), 0.4)
        torch.testing.assert_close(
            source.log_prob(values, scalar_time),
            source.log_prob(values, batched_time),
        )
        torch.testing.assert_close(
            source.velocity(values, scalar_time),
            source.velocity(values, batched_time),
        )

    def test_samples_have_the_exact_endpoint_mean(self):
        source = standard_branching_flow()
        generator = torch.Generator().manual_seed(7)
        samples = source.sample(100_000, torch.tensor(1.0), generator)
        expected = (source.weights[:, None] * source.endpoint_means).sum(dim=0)
        torch.testing.assert_close(samples.mean(dim=0), expected, atol=0.02, rtol=0.0)

    def test_distillation_only_objective_backpropagates(self):
        source = standard_branching_flow()
        student = TimeDependentGaussianPC(num_components=2, dim=2, time_order=4)
        with torch.no_grad():
            student.mean_coefficients.normal_(0.0, 0.01)
            student.log_scale_coefficients.normal_(0.0, 0.01)
        times = torch.rand(16)
        values = source.sample(16, times, torch.Generator().manual_seed(9))
        target = source.velocity(values, times)
        loss = torch.nn.functional.mse_loss(student.velocity(values, times), target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(
            any(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in student.parameters()
            )
        )

    def test_beta_timestep_scheme_biases_toward_data_endpoint(self):
        torch.manual_seed(11)
        times = sample_path_times(
            100_000,
            torch.device("cpu"),
            torch.Generator().manual_seed(13),
            "beta",
            1.0,
            0.8,
        )
        self.assertGreater(float(times.mean()), 0.54)
        self.assertLess(float(times.mean()), 0.57)


if __name__ == "__main__":
    unittest.main()
