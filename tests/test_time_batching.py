import unittest

import torch

from repc.hclt_pc import TimeDependentHCLTPC
from repc.pc import TimeDependentGaussianPC
from repc.region_pc import RegionGraphPC


class IndependentTimeBatchingTest(unittest.TestCase):
    def test_vector_times_match_individual_velocity_evaluations(self):
        torch.manual_seed(7)
        models = (
            TimeDependentHCLTPC([-1, 0, 1, 1], 4, 2, "dense"),
            TimeDependentHCLTPC([-1, 0, 1, 1], 4, 2, "monarch"),
            RegionGraphPC(2, 2, 2, 2),
            TimeDependentGaussianPC(4, 4, 2),
        )
        values = torch.randn(3, 4)
        times = torch.tensor([0.1, 0.5, 0.9])
        for model in models:
            with self.subTest(model=type(model).__name__):
                model.eval()
                batched = model.velocity(values, times)
                individual = torch.cat(
                    [
                        model.velocity(values[index : index + 1], times[index])
                        for index in range(len(times))
                    ]
                )
                torch.testing.assert_close(batched, individual)


if __name__ == "__main__":
    unittest.main()
