import math
import unittest

import torch

from repc.region_pc import RegionGraphPC


class RegionGraphPCTest(unittest.TestCase):
    def test_vectorized_sampling_and_standard_normal_boundary(self):
        torch.manual_seed(2)
        model = RegionGraphPC(height=4, width=4, num_channels=2, time_order=2)
        generator = torch.Generator().manual_seed(5)
        times = torch.tensor([0.0, 0.5, 1.0])
        samples = model.sample_path(17, times, generator=generator)
        self.assertEqual(samples.shape, (3, 17, 16))
        self.assertTrue(torch.isfinite(samples).all())

        values = torch.randn(3, 16)
        expected = -0.5 * (values.square().sum(dim=1) + 16 * math.log(2.0 * math.pi))
        torch.testing.assert_close(model.log_prob(values, torch.tensor(0.0)), expected)


if __name__ == "__main__":
    unittest.main()
