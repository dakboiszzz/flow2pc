import tempfile
import unittest
from pathlib import Path

import torch

from repc.flow import build_teacher, torchcfm_mnist_teacher_config
from repc.pc import TimeDependentGaussianPC
from train import (
    build_flow_matcher,
    distillation_loss,
    flow_matching_loss,
    load_teacher,
    sample_distillation_timesteps,
)


class TorchCFMTeacherTest(unittest.TestCase):
    def test_beta_timestep_sampling_biases_toward_data_endpoint(self):
        torch.manual_seed(0)
        times = sample_distillation_timesteps(
            100_000,
            torch.device("cpu"),
            torch.float32,
            sampling="beta",
            beta_alpha=1.0,
            beta_beta=0.8,
        )

        self.assertTrue(torch.all((times > 0.0) & (times < 1.0)))
        self.assertAlmostEqual(times.mean().item(), 1.0 / 1.8, delta=0.005)
        self.assertGreater((times >= 0.8).float().mean().item(), 0.26)

    def test_training_checkpoint_and_distillation_path(self):
        torch.manual_seed(0)
        model_config = torchcfm_mnist_teacher_config()
        teacher = build_teacher(model_config)
        images = torch.randn(2, 1, 28, 28)

        for matcher_name in ("vanilla_cfm", "exact_ot_cfm"):
            with self.subTest(matcher=matcher_name):
                path_config = {"name": matcher_name, "sigma": 0.0}
                matcher = build_flow_matcher(path_config)
                teacher_loss = flow_matching_loss(teacher, matcher, images)
                self.assertTrue(torch.isfinite(teacher_loss))
                teacher_loss.backward()
                teacher.zero_grad(set_to_none=True)

                checkpoint = {
                    "model_type": "flow",
                    "model_config": model_config,
                    "flow_matching_config": path_config,
                    "state_dict": teacher.state_dict(),
                }
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "teacher.pt"
                    torch.save(checkpoint, path)
                    loaded_teacher, loaded_path_config = load_teacher(
                        path,
                        torch.device("cpu"),
                    )

                self.assertEqual(loaded_path_config, path_config)
                student = TimeDependentGaussianPC(
                    num_components=2,
                    dim=28 * 28,
                    time_order=2,
                )
                student_loss = distillation_loss(
                    student,
                    loaded_teacher,
                    build_flow_matcher(loaded_path_config),
                    images,
                )
                self.assertTrue(torch.isfinite(student_loss))
                student_loss.backward()


if __name__ == "__main__":
    unittest.main()
