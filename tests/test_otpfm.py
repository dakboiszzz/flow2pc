"""Numerical parity, holdout exclusion, anchored boundary, and sweep contracts."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from repc.otpfm import (DEFAULT_OTP_ROOT, OTPVelocity, build_cache, native_factories,
                        sample_cache)
from repc.otpfm_data import couple_observed, load_data, physical
from repc.otpfm_eval import finite_teacher_path, log_comparison, score_paths, w2
from repc.otpfm_student import (AnchoredHCLT, fit_source, build_student,
                                initialize_student, build_optimizer)
from repc.single_cell import atomic_json, sha256_file, build_single_cell_student
from scripts import run_otpfm_sweep as sweep
from scripts import train_otpfm as train


class MetricAndBoundaryTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(10)

    def test_default_recipe_reuses_ali_student_without_freezing_parameters(self):
        torch.manual_seed(17)
        actual = build_student(2, 4, 8, 3)
        torch.manual_seed(17)
        expected = build_single_cell_student({"name": "hclt", "parents": [-1, 0],
            "num_latents": 4, "time_embedding_dim": 8, "time_feature_seed": 3,
            "fourier_zero_center": False})
        for key, value in expected.state_dict().items():
            self.assertTrue(torch.equal(actual.state_dict()[key], value), key)
        self.assertTrue(all(p.requires_grad for p in actual.parameters()))
        self.assertFalse(actual.fourier_zero_center)
        source = torch.randn(45, 2)
        with patch("repc.otpfm_student.fit_source", side_effect=AssertionError("unexpected source MLE")):
            info = initialize_student(actual, source[:40], source[40:], "ali", 3)
        self.assertEqual(info["fit_steps"], 0)
        loss = actual.velocity(source, torch.rand(45)).square().mean()
        loss.backward()
        self.assertIsNotNone(actual.root_logits.grad)
        self.assertIsNotNone(actual.transition_logits.grad)

    def test_ali_optimizer_decay_and_scheduler_resume(self):
        student = build_student(1, 4, 8)
        optimizer, scheduler = build_optimizer(student, "ali", .001, 4)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], .01)
        optimizer.step()
        scheduler.step()
        self.assertLess(optimizer.param_groups[0]["lr"], .001)
        clone_optimizer, clone_scheduler = build_optimizer(student, "ali", .001, 4)
        clone_optimizer.load_state_dict(optimizer.state_dict())
        clone_scheduler.load_state_dict(scheduler.state_dict())
        optimizer.step(); scheduler.step()
        clone_optimizer.step(); clone_scheduler.step()
        self.assertEqual(optimizer.param_groups[0]["lr"], clone_optimizer.param_groups[0]["lr"])
        anchored = build_student(1, 4, 8, recipe="anchored")
        _, anchored_scheduler = build_optimizer(anchored, "anchored", .001, 4)
        self.assertIsNone(anchored_scheduler)

    def test_training_defaults_and_recipe_configuration(self):
        required = ["--reference-dir", "/tmp/reference", "--output-dir", "/tmp/student"]
        args = train.parse_args(required)
        self.assertEqual(args.student_recipe, "ali")
        self.assertFalse(args.direct_sampling)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text(json.dumps({"student_recipe": "anchored", "direct_sampling": True}))
            args = train.parse_args(required + ["--config", str(config)])
            self.assertEqual(args.student_recipe, "anchored")
            self.assertTrue(args.direct_sampling)
            args = train.parse_args(required + ["--config", str(config), "--student-recipe", "ali", "--no-direct-sampling"])
            self.assertEqual(args.student_recipe, "ali")
            self.assertFalse(args.direct_sampling)

    def test_w2_uses_squared_cost_root_and_all_unequal_samples(self):
        a = torch.tensor([[0.], [2.]])
        b = torch.tensor([[0.], [1.], [2.]])
        self.assertAlmostEqual(w2(a, b), np.sqrt(1 / 3))
        self.assertEqual(w2(a, a), 0)
        x = torch.randn(8, 2)
        self.assertAlmostEqual(w2(x, x + torch.tensor([3., 4.])), 5., places=5)
        with self.assertRaises(ValueError):
            w2(a, torch.full((3, 1), float("nan")))

    def test_teacher_student_comparison_keeps_pairwise_and_population_errors_separate(self):
        contract = {"times": [0., .5, 1.], "holdout_indices": [1], "dim": 1,
                    "normalization": {"scale": [1.], "mean": [0.]}}
        teacher = torch.tensor([[[0.], [2.]]] * 3)
        student = teacher.flip(1)
        result = score_paths({"contract": contract, "clouds": list(teacher)},
                             {"teacher_ode": teacher, "student_ode": student})
        self.assertEqual(result["student_ode_vs_teacher_ode_holdout_w2"], 0.)
        self.assertEqual(result["scores"][1]["paired_ode_position_rmse"], 2.)

    def test_wandb_logs_comparisons_validation_images_and_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["metrics.json", "predictions.pt", "marginal_comparison.png", "w2_comparison.png"]:
                (root / name).write_text("test artifact")
            run = SimpleNamespace(log=Mock(), summary={}, dir=str(root / "wandb"), save=Mock())
            metrics = {"student_direct_holdout_w2": 1., "validation": {"relative_velocity_mse": .2},
                       "source_fit": {"validation_nll_normalized": .3},
                       "scores": [{"index": 1, "student_direct_w2": .8, "teacher_ode_w2": 1.1}]}
            with patch.dict("sys.modules", {"wandb": SimpleNamespace(Image=lambda path: path)}):
                log_comparison(run, metrics, root)
            self.assertEqual(run.summary["eval/validation/relative_velocity_mse"], .2)
            self.assertEqual(run.summary["eval/time_1/teacher_ode_w2"], 1.1)
            self.assertEqual(run.log.call_count, 2)
            self.assertTrue(any("predictions.pt" in call.args[0] for call in run.save.call_args_list))

    def test_source_law_is_exactly_unchanged_after_velocity_updates(self):
        student = AnchoredHCLT(2, 4, 8)
        source = torch.randn(45, 2) * .4 + 2
        info = fit_source(student, source[:40], source[40:], steps=25)
        self.assertLessEqual(info["best_step"], 25)
        zero = torch.tensor(0.)
        before = student.log_prob(source, zero).detach().clone()
        boundary = {key: value.clone() for key, value in student.state_dict().items()
                    if key in ("source_mean", "source_log_scale", "root_logits", "transition_logits")}
        optimizer = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=.01)
        student.train()
        for _ in range(4):
            loss = (student.velocity(source, torch.rand(len(source))) - 1.).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertTrue(torch.equal(student.log_prob(source, zero).detach(), before))
        for key, values in boundary.items():
            self.assertTrue(torch.equal(student.state_dict()[key], values))
        self.assertFalse(student.root_logits.requires_grad)
        self.assertFalse(student.transition_logits.requires_grad)

    def test_saturated_scales_have_correct_zero_time_derivative(self):
        student = AnchoredHCLT(1, 4, 4)
        student.source_log_scale.fill_(30.)
        with torch.no_grad():
            student.log_scale_coefficients.fill_(.1)
        _, _, logs, derivative = student.leaf_parameters(torch.tensor(.3))
        self.assertTrue(torch.equal(logs, torch.full_like(logs, 20.)))
        self.assertTrue(torch.equal(derivative, torch.zeros_like(derivative)))

    def test_cache_coupling_never_reads_heldout_clouds(self):
        clouds = [torch.randn(n, 2) for n in (9, 7, 11, 8, 10)]
        tuples, indices = couple_observed(clouds, [0, 2, 4])
        poisoned = copy.deepcopy(clouds)
        poisoned[1].fill_(float("nan"))
        poisoned[3].fill_(float("nan"))
        other, other_indices = couple_observed(poisoned, [0, 2, 4])
        self.assertTrue(torch.equal(tuples, other))
        self.assertTrue(torch.equal(indices, other_indices))
        self.assertTrue(torch.equal(tuples[:, 0], clouds[0]))

    def test_native_scaler_arithmetic_ragged_clouds_and_source_row_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = np.empty(5, dtype=object)
            raw[:] = [np.arange(n, dtype=np.float32)[:, None] + i for i, n in enumerate([20, 13, 15, 16, 17])]
            path = root / "dingling_pm25_13marginals.npy"
            np.save(path, raw)
            atomic_json(root / "manifest.json", {"files_sha256": {path.name: sha256_file(path)}})
            entry = {"dataset": "beijing", "dim": 1, "data_root": str(root),
                     "data_manifest_sha256": sha256_file(root / "manifest.json"),
                     "observed_indices": [0, 2, 4], "holdout_indices": [1, 3],
                     "normalized_times": [0, .25, .5, .75, 1.]}
            data = load_data(entry, {"tks": [.5], "train_times": [0, 2, 4], "holdout_times": [1, 3]})
            scaler = StandardScaler().fit(np.concatenate(list(raw)))
            for index, cloud in enumerate(data["clouds"]):
                self.assertTrue(np.array_equal(cloud.numpy(), scaler.transform(raw[index]).astype(np.float32)))
                np.testing.assert_allclose(physical(cloud, data["contract"]).numpy(), raw[index], atol=1e-5)
            train, val = data["train_rows"].tolist(), data["validation_rows"].tolist()
            self.assertFalse(set(train) & set(val))
            self.assertEqual(sorted(train + val), list(range(20)))


@unittest.skipUnless((DEFAULT_OTP_ROOT / "experiments/train.py").exists(), "native OTP checkout unavailable")
class NativeParityTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        factory, potential_factory = native_factories(DEFAULT_OTP_ROOT)
        self.config = {"tks": [.25, .5, .75], "potential": "w2inf", "strength": 20.,
                       "lambda_type": "box", "width": .2, "hidden_dim": 8,
                       "x_emb_dim": 8, "t_emb_dim": 8, "num_hidden_layers": 2}
        self.model = factory(self.config, 2, potential_factory(self.config), "cpu")

    def test_diagonal_ema_adapter_and_frozen_parameters(self):
        teacher = OTPVelocity(self.model)
        x, times = torch.randn(9, 2), torch.rand(9)
        torch.testing.assert_close(teacher(times, x), self.model.v_func_ema(x, times, times), rtol=0, atol=0)
        torch.testing.assert_close(teacher(.3, x), teacher(torch.full((9,), .3), x))
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        with self.assertRaises(ValueError):
            teacher(torch.ones(2), x)

    def test_cached_forces_match_native_direct_solver(self):
        from otpfm.solvers import DirectSolver

        tuples = torch.randn(11, 5, 2)
        cache = build_cache(self.model, tuples, .7)
        knots = torch.tensor(self.config["tks"])
        potentials = list(self.model.potentials.values())
        h = torch.stack([self.model.x_time_dep(p, knots[:, None]).reshape(3) for p in potentials], 1)
        bases = tuples[:, 0][None] + knots[:, None, None] * (tuples[:, -1] - tuples[:, 0])[None]
        _, forces, _ = DirectSolver(5)(bases, bases, h[:, :, None, None], potentials,
                                       tuples[:, 1:-1].transpose(0, 1))
        torch.testing.assert_close(cache["forces"], forces, rtol=2e-5, atol=2e-5)
        self.assertLess(cache["relative_linear_residual"], 1e-5)
        rows = torch.arange(len(tuples))
        for t, endpoint in [(0., 0), (1., -1)]:
            values, _ = sample_cache(self.model, cache, rows, torch.full((len(rows),), t))
            torch.testing.assert_close(values, tuples[:, endpoint], atol=2e-5, rtol=1e-5)

    def test_conditional_velocity_is_full_vector_derivative(self):
        tuples = torch.randn(6, 5, 2)
        cache = build_cache(self.model, tuples, .9)
        times = torch.tensor([.13, .32, .42, .57, .68, .88], requires_grad=True)
        values, velocity = sample_cache(self.model, cache, torch.arange(6), times)
        derivative = torch.stack([torch.autograd.grad(values[:, dim].sum(), times, retain_graph=True)[0]
                                  for dim in range(2)], 1)
        torch.testing.assert_close(derivative, velocity, atol=1e-5, rtol=1e-5)

    def test_native_row_argmax_chain_parity(self):
        from experiments.gulfofmexico.data import compute_gom_ot_alignments, GoMMultiMarginalDataset

        clouds = [torch.randn(n, 2) for n in (9, 7, 11, 8, 10)]
        alignments = compute_gom_ot_alignments([cloud.numpy() for cloud in clouds], [0, 2, 4])
        native = GoMMultiMarginalDataset([cloud.numpy() for cloud in clouds], [1, 3], alignments)
        _, indices = couple_observed(clouds, [0, 2, 4])
        self.assertTrue(np.array_equal(indices.numpy(), native.ot_chains))

    def test_finite_teacher_includes_exact_nonuniform_observation_times(self):
        teacher = OTPVelocity(self.model)
        times = [0., .08333333333333, .25, .66666666666667, 1.]
        path, info = finite_teacher_path(teacher, torch.zeros(8, 2), times, 2)
        self.assertEqual(tuple(path.shape), (5, 8, 2))
        self.assertTrue(torch.isfinite(path).all())
        self.assertIn("exact_observation", info["method"])


class SweepPlanTest(unittest.TestCase):
    def test_shipped_sweeps_use_saved_teachers_and_requested_output_root(self):
        args = sweep.parse_args([])
        self.assertEqual(str(args.output_root), "/data/lthung/flow2pc/otp-fm/ali-recipe")
        fake = {"seed": 44, "checkpoint_sha256": "abc", "config_sha256": "def"}
        with patch.object(sweep, "select_teacher", return_value=(fake, {})):
            plans = sweep.build_plan(args)
        self.assertEqual(set(plans), {"gom", "beijing"})
        for dataset, plan in plans.items():
            self.assertEqual(len(plan["jobs"]), 162)
            for name, settings in plan["jobs"]:
                self.assertEqual(settings["wandb_project"], "flow2pc")
                self.assertEqual(settings["wandb_group"], f"otpfm-{dataset}-w2-teacher44")
                self.assertEqual(settings["steps"], 20000)
                self.assertEqual(settings["student_recipe"], "ali")
                self.assertFalse(settings["direct_sampling"])

    def test_student_command_cannot_call_native_teacher_training(self):
        args = sweep.parse_args([])
        command = sweep.task_command(args, ("gom", "train", "example", {}, 0), "cuda:3")
        self.assertTrue(command[2].endswith("scripts/train_otpfm.py"))
        self.assertIn("--reference-dir", command)
        self.assertNotIn("experiments.train", command)
        self.assertEqual(command[command.index("--device") + 1], "cuda:3")


if __name__ == "__main__":
    unittest.main()
