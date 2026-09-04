import sys
import unittest
from unittest.mock import patch

from repc.flow import build_teacher, torchcfm_mnist_teacher_config
from repc.hclt_pc import TimeDependentHCLTPC
from repc.pc import TimeDependentGaussianPC
from repc.region_pc import RegionGraphPC
from train import parse_args


class CapacityProfileTest(unittest.TestCase):
    def test_standard_profile_uses_power_of_two_widths(self):
        configurations = (
            (["--pc-structure", "hclt", "--pc-layer", "dense"], "num_latents", 64),
            (
                ["--pc-structure", "hclt", "--pc-layer", "monarch"],
                "num_latents",
                64,
            ),
            (["--pc-structure", "pd"], "num_channels", 8),
            (["--pc-structure", "gaussian"], "num_components", 256),
        )
        for options, attribute, expected in configurations:
            with self.subTest(options=options):
                argv = [
                    "train.py",
                    "distill",
                    "--teacher-checkpoint",
                    "teacher.pt",
                    *options,
                ]
                with patch.object(sys, "argv", argv):
                    args = parse_args()
                self.assertEqual(args.capacity_profile, "standard")
                self.assertEqual(getattr(args, attribute), expected)
                self.assertEqual(expected & (expected - 1), 0)

    def test_standard_students_have_aligned_capacity_headroom(self):
        parents = [-1] + list(range(783))
        teacher = build_teacher(torchcfm_mnist_teacher_config())
        teacher_parameters = sum(
            parameter.numel() for parameter in teacher.parameters()
        )
        self.assertEqual(teacher_parameters, 1_075_361)

        factories_and_counts = (
            (
                lambda: TimeDependentHCLTPC(parents, 64, 4, "dense"),
                3_608_640,
            ),
            (
                lambda: TimeDependentHCLTPC(parents, 64, 4, "monarch"),
                1_604_160,
            ),
            (lambda: RegionGraphPC(28, 28, 8, 4), 1_570_824),
            (lambda: TimeDependentGaussianPC(256, 784, 4), 1_605_888),
        )
        for factory, expected_parameters in factories_and_counts:
            with self.subTest(expected_parameters=expected_parameters):
                student = factory()
                student_parameters = sum(
                    parameter.numel() for parameter in student.parameters()
                )
                self.assertEqual(student_parameters, expected_parameters)
                ratio = student_parameters / teacher_parameters
                self.assertGreater(ratio, 1.0)
                self.assertLessEqual(ratio, 3.36)

    def test_beta_timestep_configuration_is_exposed_by_cli(self):
        argv = [
            "train.py",
            "distill",
            "--teacher-checkpoint",
            "teacher.pt",
            "--timestep-sampling",
            "beta",
            "--timestep-beta-alpha",
            "1.0",
            "--timestep-beta-beta",
            "0.8",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.timestep_sampling, "beta")
        self.assertEqual(args.timestep_beta_alpha, 1.0)
        self.assertEqual(args.timestep_beta_beta, 0.8)

    def test_endpoint_em_configuration_is_exposed_by_cli(self):
        argv = [
            "train.py",
            "distill",
            "--teacher-checkpoint",
            "teacher.pt",
            "--endpoint-pretraining",
            "em",
            "--endpoint-em-epochs",
            "20",
            "--endpoint-em-pseudocount",
            "0.1",
            "--endpoint-em-min-std",
            "0.01",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.endpoint_pretraining, "em")
        self.assertEqual(args.endpoint_em_epochs, 20)
        self.assertEqual(args.endpoint_em_pseudocount, 0.1)
        self.assertEqual(args.endpoint_em_min_std, 0.01)

    def test_iclr2022_topology_protocol_is_exposed_by_cli(self):
        argv = [
            "train.py",
            "distill",
            "--teacher-checkpoint",
            "teacher.pt",
            "--hclt-topology-protocol",
            "iclr2022-categorical",
            "--hclt-mi-sample-chunk-size",
            "1024",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.hclt_topology_protocol, "iclr2022-categorical")
        self.assertEqual(args.hclt_mi_sample_chunk_size, 1_024)


if __name__ == "__main__":
    unittest.main()
