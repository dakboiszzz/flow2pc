import unittest

import torch

from repc.data import quantize_mnist
from repc.evaluation import (
    class_distribution_metrics,
    frechet_distance,
    improved_precision_recall,
)
from repc.pc import TimeDependentGaussianPC, discretized_gaussian_log_mass


class EvaluationTest(unittest.TestCase):
    def test_mnist_quantization(self):
        values = torch.tensor([-2.0, -1.0, -0.3, 0.3, 1.0, 2.0])
        quantized = quantize_mnist(values)
        self.assertGreaterEqual(quantized.min().item(), -1.0)
        self.assertLessEqual(quantized.max().item(), 1.0)
        bin_indices = (quantized + 1.0) * 255.0 / 2.0
        torch.testing.assert_close(bin_indices, bin_indices.round())

    def test_discretized_gaussian_bins_are_normalized(self):
        centers = torch.linspace(-1.0, 1.0, 256).reshape(256, 1)
        log_mass = discretized_gaussian_log_mass(
            centers,
            torch.zeros(1, 1),
            torch.zeros(1, 1),
        )
        torch.testing.assert_close(
            log_mass[:, 0, 0].exp().sum(),
            torch.tensor(1.0),
            atol=1e-5,
            rtol=1e-5,
        )

        model = TimeDependentGaussianPC(num_components=1, dim=1)
        model_mass = model.discrete_log_prob(centers, torch.tensor(0.0))
        torch.testing.assert_close(
            model_mass.exp().sum(),
            torch.tensor(1.0),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_identical_feature_sets_have_zero_fid_and_full_coverage(self):
        torch.manual_seed(4)
        features = torch.randn(40, 6)
        self.assertLess(frechet_distance(features, features), 1e-8)
        precision, recall = improved_precision_recall(
            features,
            features,
            torch.device("cpu"),
            nearest_neighbors=3,
            chunk_size=10,
        )
        self.assertEqual(precision, 1.0)
        self.assertEqual(recall, 1.0)

    def test_balanced_confident_classes(self):
        probabilities = torch.eye(10).repeat(2, 1)
        labels = torch.arange(10).repeat(2)
        metrics = class_distribution_metrics(probabilities, labels)
        self.assertAlmostEqual(metrics["classifier_confidence"], 1.0)
        self.assertAlmostEqual(metrics["class_entropy_normalized"], 1.0, places=6)
        self.assertAlmostEqual(metrics["class_jsd_to_test"], 0.0)


if __name__ == "__main__":
    unittest.main()
