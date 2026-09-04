import unittest

import torch

from repc.hclt import (
    categorical_mutual_information,
    parents_from_mutual_information,
)
from train_endpoint_hclt import (
    add_statistics,
    linear_em_step_size,
    parse_args,
)


class ICLR2022EndpointTrainingTests(unittest.TestCase):
    def test_published_table2_defaults(self):
        args = parse_args([])
        self.assertEqual(args.num_latents, 16)
        self.assertEqual(args.batch_size, 1_024)
        self.assertEqual(args.mini_batch_epochs, 100)
        self.assertEqual(args.full_batch_epochs, 20)
        self.assertEqual(args.validation_size, 0)
        self.assertEqual(args.em_step_start, 0.15)
        self.assertEqual(args.em_step_end, 0.05)
        self.assertEqual(args.topology_source, "iclr2022")

    def test_linear_em_schedule_includes_both_endpoints(self):
        self.assertEqual(linear_em_step_size(0, 11, 0.15, 0.05), 0.15)
        self.assertAlmostEqual(linear_em_step_size(5, 11, 0.15, 0.05), 0.10)
        self.assertEqual(linear_em_step_size(10, 11, 0.15, 0.05), 0.05)

    def test_batch_statistics_are_added(self):
        first = {"count": torch.tensor([1.0, 2.0]), "ll": torch.tensor(-3.0)}
        second = {"count": torch.tensor([4.0, 8.0]), "ll": torch.tensor(-5.0)}
        accumulated = add_statistics(None, first)
        accumulated = add_statistics(accumulated, second)
        torch.testing.assert_close(accumulated["count"], torch.tensor([5.0, 10.0]))
        torch.testing.assert_close(accumulated["ll"], torch.tensor(-8.0))
        self.assertIsNot(accumulated["count"], first["count"])

    def test_empirical_categorical_mutual_information(self):
        data = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ],
            dtype=torch.uint8,
        )
        mutual_information = categorical_mutual_information(
            data,
            num_categories=2,
            sample_chunk_size=2,
            variable_chunk_size=2,
        )
        self.assertAlmostEqual(mutual_information[0, 1].item(), 1.0, places=6)
        self.assertAlmostEqual(mutual_information[0, 2].item(), 0.0, places=6)
        torch.testing.assert_close(mutual_information, mutual_information.T)
        torch.testing.assert_close(mutual_information.diag(), torch.zeros(3))

    def test_chow_liu_tree_is_center_rooted(self):
        mutual_information = torch.tensor(
            [
                [0.0, 3.0, 1.0],
                [3.0, 0.0, 2.0],
                [1.0, 2.0, 0.0],
            ]
        )
        self.assertEqual(
            parents_from_mutual_information(mutual_information),
            [1, -1, 1],
        )


if __name__ == "__main__":
    unittest.main()
