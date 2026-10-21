"""Contract and objective tests; no downloads, W&B sessions, or real training."""

import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from repc.data import mnist_latent_loader
from repc.flow import build_teacher
from repc.hclt_pc import TimeDependentHCLTPC
from repc.mnist_latent import (check_contract, decode_codes, latent_dimension,
                               learn_latent_topology, load_latent_bundle, normalize_codes)
from repc.mnist_latent_eval import (comparison_tables, fidelity_metrics, generate_images,
                                    resolve_classifier_checkpoint, student_samples,
                                    teacher_samples, teacher_student_metrics, validation_metrics)
from train import build_flow_matcher, distillation_loss, flow_matching_loss, load_teacher
from train_mnist_latent import build_latent_student, load_latent_teacher, resolve_device, should_log_images


class LatentMNISTTest(unittest.TestCase):
    def test_representation_supports_98_and_196_dimensions(self):
        for channels in (2, 4):
            manifest = {"dim": channels * 49, "latent_shape": [channels, 7, 7],
                        "encode_mode": "posterior_mean", "flatten_order": "CHW"}
            self.assertEqual(latent_dimension(manifest), channels * 49)
        for invalid in ({"dim": 196}, {"latent_shape": [2, 14, 7]},
                        {"flatten_order": "HWC"}, {"encode_mode": "sample"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                latent_dimension({"dim": 98, "latent_shape": [2, 7, 7],
                                  "encode_mode": "posterior_mean", "flatten_order": "CHW", **invalid})

    def test_98d_bundle_checks_normalizer_shape_and_decodes_two_channels(self):
        from repc.mnist_latent import atomic_save, sha256_file, write_json

        class Decoder(nn.Module):
            def decode(self, values):
                return values

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            codes = {split: {"values": torch.zeros(count, 98), "labels": torch.zeros(count, dtype=torch.long),
                             "indices": torch.arange(count)}
                     for split, count in (("train", 55000), ("validation", 5000), ("test", 10000))}
            normalizer = {"mean": torch.zeros(1, 98), "scale": torch.ones(1, 98),
                          "latent_shape": [2, 7, 7], "flatten_order": "CHW"}
            atomic_save(codes, directory / "codes.pt")
            atomic_save(normalizer, directory / "normalizer.pt")
            manifest = {"dim": 98, "latent_shape": [2, 7, 7], "encode_mode": "posterior_mean",
                        "flatten_order": "CHW", "codes_sha256": sha256_file(directory / "codes.pt"),
                        "normalizer_sha256": sha256_file(directory / "normalizer.pt")}
            write_json(manifest, directory / "manifest.json")
            loaded, contract = load_latent_bundle(directory)
            self.assertEqual(loaded["train"]["values"].shape, (55000, 98))
            self.assertEqual(decode_codes(Decoder(), torch.ones(2, 98), contract).shape, (2, 2, 7, 7))
            normalizer["latent_shape"] = [4, 7, 7]
            atomic_save(normalizer, directory / "normalizer.pt")
            manifest["normalizer_sha256"] = sha256_file(directory / "normalizer.pt")
            write_json(manifest, directory / "manifest.json")
            with self.assertRaisesRegex(ValueError, "normalization"):
                load_latent_bundle(directory)

    def test_98d_teacher_loads_and_rejects_an_inconsistent_architecture(self):
        config = {"name": "latent_residual_mlp", "dim": 98, "hidden_dim": 16,
                  "num_blocks": 1, "time_embedding_dim": 8}
        teacher = build_teacher(config)
        contract = {"manifest_sha256": "z98", "mean": torch.zeros(1, 98), "scale": torch.ones(1, 98)}
        checkpoint = {"model_type": "flow", "model_config": config, "state_dict": teacher.state_dict(),
                      "flow_matching_config": {"name": "vanilla_cfm", "sigma": 0.0}, "data_contract": contract}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            torch.save(checkpoint, path)
            loaded, _ = load_latent_teacher(path, contract, torch.device("cpu"))
            self.assertEqual(loaded(torch.tensor(0.5), torch.ones(2, 98)).shape, (2, 98))
            checkpoint["model_config"] = {**config, "dim": 196}
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "architecture dimension"):
                load_latent_teacher(path, contract, torch.device("cpu"))

    def test_zero_image_interval_logs_only_after_training_finishes(self):
        self.assertEqual([step for step in range(1, 11) if should_log_images(step, 10, 0)], [10])
        self.assertEqual([step for step in range(1, 11) if should_log_images(step, 10, 3)], [3, 6, 9, 10])

    def test_cuda_resolution_uses_explicit_visible_device_index(self):
        for requested, expected in (("cuda", 2), ("auto", 2), ("cuda:1", 1)):
            with self.subTest(requested=requested):
                with (patch("torch.cuda.is_available", return_value=True),
                      patch("torch.cuda.current_device", return_value=2),
                      patch("torch.cuda.set_device") as set_device):
                    self.assertEqual(resolve_device(requested), torch.device("cuda", expected))
                    set_device.assert_called_once_with(expected)

    def test_cpu_resolution_and_unavailable_cuda(self):
        with (patch("torch.cuda.is_available", return_value=False),
              patch("torch.cuda.set_device") as set_device):
            self.assertEqual(resolve_device("auto"), torch.device("cpu"))
            self.assertEqual(resolve_device("cpu"), torch.device("cpu"))
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")
            set_device.assert_not_called()

    def test_normalization_uses_train_only_and_inverts_without_clipping(self):
        train = torch.tensor([[-2., 7.], [2., 7.]])
        validation = torch.tensor([[20., 8.]])
        test = torch.tensor([[40., 9.]])
        values, mean, scale, _ = normalize_codes(train, validation, test)
        torch.testing.assert_close(mean, torch.tensor([[0., 7.]]))
        self.assertEqual(scale[0, 0].item(), 2.)
        self.assertGreater(values[1][0, 0].item(), 1.)
        torch.testing.assert_close(values[2] * scale + mean, test)
        loader = mnist_latent_loader({"validation": {"values": values[1], "labels": torch.tensor([3])}},
                                    "validation", batch_size=1)
        torch.testing.assert_close(next(iter(loader))[0], values[1])

    def test_teacher_checkpoint_and_frozen_distillation(self):
        config = {"name": "latent_residual_mlp", "dim": 196, "hidden_dim": 16,
                  "num_blocks": 1, "time_embedding_dim": 8}
        teacher = build_teacher(config)
        data = torch.randn(3, 196)
        matcher_config = {"name": "vanilla_cfm", "sigma": 0.0}
        matcher = build_flow_matcher(matcher_config)
        loss = flow_matching_loss(teacher, matcher, data)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(teacher.output[-1].weight.grad)
        contract = {"manifest_sha256": "example", "mean": torch.zeros(1, 196), "scale": torch.ones(1, 196)}
        checkpoint = {"model_type": "flow", "model_config": config, "state_dict": teacher.state_dict(),
                      "flow_matching_config": matcher_config, "data_contract": contract}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            torch.save(checkpoint, path)
            loaded, _ = load_latent_teacher(path, contract, torch.device("cpu"))
            generic_loaded, path_config = load_teacher(path, torch.device("cpu"))
            torch.testing.assert_close(loaded(torch.tensor(0.5), data), generic_loaded(torch.full((3,), 0.5), data))
            self.assertEqual(path_config, matcher_config)
        student = TimeDependentHCLTPC(parents=[-1] + [0] * 195, num_latents=2,
                                      time_conditioning="fourier", time_embedding_dim=4)
        loss = distillation_loss(student, loaded, matcher, data)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(parameter.grad is None for parameter in loaded.parameters()))
        self.assertTrue(any(parameter.grad is not None for parameter in student.parameters()))

    def test_contract_mismatch_is_rejected(self):
        saved = {"manifest_sha256": "a", "mean": torch.zeros(1, 2), "scale": torch.ones(1, 2)}
        with self.assertRaises(ValueError):
            check_contract(saved, {**saved, "manifest_sha256": "b"})
        with self.assertRaises(ValueError):
            check_contract(saved, {**saved, "scale": torch.ones(1, 2) * 2})

    def test_decode_inverts_normalization_and_preserves_chw(self):
        class Decoder(nn.Module):
            def decode(self, values):
                return values

        contract = {"mean": torch.arange(196).reshape(1, 196).float(),
                    "scale": torch.ones(1, 196) * 2, "latent_shape": [4, 7, 7]}
        values = torch.ones(2, 196) * 3
        actual = decode_codes(Decoder(), values, contract)
        torch.testing.assert_close(actual.flatten(1), values * 2 + contract["mean"])
        self.assertEqual(actual.shape, (2, 4, 7, 7))

    def test_continuous_topology_handles_constant_coordinates(self):
        generator = torch.Generator().manual_seed(0)
        first = torch.randn(100, generator=generator)
        values = torch.stack([first, first + 0.01 * torch.randn(100, generator=generator),
                              torch.randn(100, generator=generator), torch.ones(100)], dim=1)
        parents, metadata = learn_latent_topology(values, 100, 0)
        self.assertEqual(len(parents), 4)
        self.assertEqual(parents.count(-1), 1)
        self.assertTrue(parents[0] == 1 or parents[1] == 0)
        self.assertEqual(metadata["constant_coordinates"], 1)
        self.assertEqual(metadata["fit_split"], "train")

    def test_gaussian_control_retains_normal_source_and_breaks_symmetry(self):
        model = build_latent_student({"name": "gaussian", "dim": 3, "num_components": 4, "time_order": 2})
        values = torch.randn(5, 3)
        expected = -0.5 * (values.square().sum(dim=1) + 3 * math.log(2 * math.pi))
        torch.testing.assert_close(model.log_prob(values, torch.tensor(0.)), expected)
        self.assertFalse(torch.equal(model.mean_coefficients[0], model.mean_coefficients[1]))

    def test_teacher_ode_direction_and_deterministic_validation(self):
        class ConstantTeacher(nn.Module):
            def forward(self, times, values):
                return torch.ones_like(values) * 2

        teacher = ConstantTeacher()
        seed = 3
        noise = torch.randn(4, 3, generator=torch.Generator().manual_seed(seed))
        samples = teacher_samples(teacher, 4, 3, torch.device("cpu"),
                                   torch.Generator().manual_seed(seed), 4)
        torch.testing.assert_close(samples, noise + 2)
        first = validation_metrics(teacher, None, noise, 2, torch.device("cpu"), seed)
        second = validation_metrics(teacher, None, noise, 2, torch.device("cpu"), seed)
        self.assertEqual(first, second)

    def test_student_ode_integrates_velocity_from_gaussian_noise(self):
        class LinearStudent(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def velocity(self, values, times):
                self.assert_eval = not self.training
                self.calls += 1
                return values

            def sample(self, *args, **kwargs):
                raise AssertionError("Student generation must integrate its velocity")

        student = LinearStudent()
        generator = torch.Generator().manual_seed(7)
        noise = torch.randn(5, 3, generator=torch.Generator().manual_seed(7))
        samples = student_samples(student, 5, 3, torch.device("cpu"), generator, 100)
        torch.testing.assert_close(samples, math.e * noise)
        self.assertEqual(student.calls, 400)
        self.assertTrue(student.assert_eval)
        self.assertTrue(student.training)
        self.assertFalse(samples.requires_grad)

    def test_image_generation_uses_matching_source_and_ode_for_both_models(self):
        class ZeroTeacher(nn.Module):
            def forward(self, times, values):
                return torch.zeros_like(values)

        class ConstantStudent(nn.Module):
            def velocity(self, values, times):
                return torch.full_like(values, 2)

            def sample(self, *args, **kwargs):
                raise AssertionError("Direct sampling is forbidden in latent image generation")

        class Decoder(nn.Module):
            def decode(self, values):
                return values[:, :1].repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)

        contract = {"mean": torch.zeros(1, 196), "scale": torch.ones(1, 196),
                    "latent_shape": [4, 7, 7], "manifest": {"dim": 196}}
        images, metrics = generate_images(ZeroTeacher(), ConstantStudent(), Decoder(), contract,
                                          5, 3, torch.device("cpu"), 7, 100)
        torch.testing.assert_close(images["student"], images["teacher"] + 2)
        self.assertEqual(images["student"].shape, (5, 1, 28, 28))
        self.assertEqual(metrics["student/ode_steps"], 100)
        self.assertEqual(metrics["student/sampling"], "velocity_ode")

    def test_mnist_fidelity_metrics_include_inception_score_and_coverage(self):
        features = torch.randn(40, 6, generator=torch.Generator().manual_seed(4))
        labels = torch.arange(40) % 10
        probabilities = torch.eye(10)[labels]
        metrics = fidelity_metrics(features, features, probabilities, labels, torch.device("cpu"),
                                   kid_subsets=2, kid_subset_size=20, is_splits=1, metric_chunk_size=16)
        self.assertLess(metrics["fid_mnist"], 1e-8)
        self.assertAlmostEqual(metrics["inception_score_mnist_mean"], 10., places=5)
        self.assertEqual(metrics["precision_mnist"], 1.)
        self.assertEqual(metrics["recall_mnist"], 1.)
        self.assertEqual(metrics["classes_covered"], 10)
        self.assertEqual(metrics["low_confidence_fraction"], 0.)
        self.assertAlmostEqual(metrics["class_jsd_to_reference"], 0.)
        uniform = torch.full((40, 10), 0.1)
        uniform_metrics = fidelity_metrics(features, features, uniform, labels, torch.device("cpu"),
                                           kid_subsets=2, kid_subset_size=20, is_splits=1, metric_chunk_size=16)
        self.assertAlmostEqual(uniform_metrics["inception_score_mnist_mean"], 1., places=5)
        self.assertEqual(uniform_metrics["low_confidence_fraction"], 1.)

    def test_identical_teacher_and_student_have_zero_paired_errors(self):
        features = torch.randn(40, 6, generator=torch.Generator().manual_seed(4))
        probabilities = torch.eye(10)[torch.arange(40) % 10]
        images = torch.zeros(40, 1, 28, 28)
        metrics = teacher_student_metrics(
            {"teacher": images, "student": images},
            {"teacher": features, "student": features},
            {"teacher": probabilities, "student": probabilities}, torch.device("cpu"),
            kid_subsets=2, kid_subset_size=20)
        self.assertLess(metrics["comparison/student_to_teacher_fid_mnist"], 1e-8)
        for key in ("paired_pixel_mse_01", "paired_pixel_mae_01", "paired_feature_mse", "paired_probability_jsd"):
            self.assertEqual(metrics[f"comparison/{key}"], 0.)
        self.assertEqual(metrics["comparison/paired_class_agreement"], 1.)

    def test_comparison_table_records_direction_and_signed_gap(self):
        metrics = {"teacher/fid_mnist": 8., "student/fid_mnist": 10.,
                   "teacher/inception_score_mnist_mean": 4., "student/inception_score_mnist_mean": 5.,
                   "real/class_distribution": [0.1] * 10, "teacher/class_distribution": [0.1] * 10,
                   "student/class_distribution": [0.1] * 10}
        tables = comparison_tables(metrics)
        rows = {row[0]: row for row in tables["teacher_student_comparison"]["data"]}
        self.assertFalse(rows["fid_mnist"][1])
        self.assertEqual(rows["fid_mnist"][-1], 2.)
        self.assertTrue(rows["inception_score_mnist_mean"][1])
        self.assertEqual(rows["inception_score_mnist_mean"][-1], 1.)
        self.assertEqual(len(tables["class_distribution"]["data"]), 10)

    def test_missing_classifier_is_not_silently_skipped(self):
        with patch("repc.mnist_latent_eval.Path.is_file", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "trained classifier"):
                resolve_classifier_checkpoint(Path("missing-classifier.pt"))


if __name__ == "__main__":
    unittest.main()
