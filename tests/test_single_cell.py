import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
import torch
from torch import nn

from repc.flow import ALICFMGeneratorWrapper, ALICFMTeacherWrapper
from repc.single_cell import (build_single_cell_student, initialize_source,
                             learn_topology, load_single_cell_data,
                             mmot_endpoints, observed_batch, sample_paths,
                             sha256_file, validation_metrics)
from repc.single_cell_eval import direct_diagnostics, emd
from repc.single_cell_tracking import (emd_scalars, init_tracking, log_evaluation,
                                       log_files, log_training, retryable_startup_error)
from scripts.run_single_cell_sweep import automatic_devices, load_jobs


class SweepGridTest(unittest.TestCase):
    def test_intensive_grid_covers_all_combinations_and_seeds(self):
        grid = Path(__file__).resolve().parents[1] / "configs/single_cell/cite5d_intensive_sweep.json"
        jobs = load_jobs(grid, {})
        self.assertEqual(len(jobs), 288)
        self.assertEqual(len({name for name, _ in jobs}), 288)
        combinations = {}
        for _, settings in jobs:
            key = tuple(settings[field] for field in
                        ("num_latents", "time_embedding_dim", "lr", "batch_size"))
            combinations.setdefault(key, set()).add(settings["seed"])
            self.assertEqual(settings["steps"], 50000)
            self.assertEqual(settings["wandb_project"], "flow2pc")
        self.assertEqual(len(combinations), 96)
        self.assertTrue(all(seeds == {0, 1, 2} for seeds in combinations.values()))
        self.assertEqual([settings["num_latents"] for _, settings in jobs[:4]], [32, 64, 128, 256])

    def test_axis_override_does_not_launch_duplicate_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            grid = Path(directory) / "grid.json"
            grid.write_text(json.dumps({"common": {"lr": .001},
                                        "parameters": {"steps": [10, 20], "seed": [0, 1]}}))
            jobs = load_jobs(grid, {"steps": 30})
            self.assertEqual(len(jobs), 2)
            self.assertEqual({settings["seed"] for _, settings in jobs}, {0, 1})
            self.assertTrue(all(settings["steps"] == 30 for _, settings in jobs))
            self.assertEqual(jobs, load_jobs(grid, {"steps": 30}))

    def test_invalid_grids_fail_before_launching_workers(self):
        invalid = [{"parameters": {}}, {"parameters": {"seed": []}},
                   {"parameters": {"seed": [0, 0]}},
                   {"parameters": {"seed": [[0]]}},
                   {"parameters": {"name": ["bad"]}},
                   {"runs": [{"name": "one"}], "parameters": {"seed": [0]}}]
        with tempfile.TemporaryDirectory() as directory:
            grid = Path(directory) / "grid.json"
            for contents in invalid:
                with self.subTest(grid=contents):
                    grid.write_text(json.dumps(contents))
                    with self.assertRaises(ValueError):
                        load_jobs(grid, {})


class SingleCellTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(42)
        self.pca = rng.normal(size=(80, 5)).astype(np.float32)
        self.days = np.repeat([2, 3, 4, 7], 20)
        input_path = self.root / "cite.h5ad"
        with h5py.File(input_path, "w") as f:
            f.create_dataset("obsm/X_pca", data=self.pca)
            f.create_dataset("obs/day", data=self.days)
        source = {"dataset": "cite", "file": "data/cite.h5ad",
                  "file_size_bytes": input_path.stat().st_size,
                  "sha256": sha256_file(input_path), "day_order_used_by_loader": [2, 3, 4, 7]}
        manifest_bytes = json.dumps({"datasets": [source]}).encode()
        (self.root / "data_manifest.json").write_bytes(manifest_bytes)
        self.contract = {"source": source, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                         "time_index": 1, "heldout_day": 3, "observed_days": [2, 4, 7],
                         "pseudo_time_convention": "equally_spaced_ordered_marginals_on_[0,1]",
                         "preprocessing": {"dataset": "cite", "n_pca_components_used": 5,
                                           "time_order": [2, 3, 4, 7], "cells_per_time": [20] * 4,
                                           "whiten": True, "whitening": {"mean": [1.] * 5, "scale": [2.] * 5},
                                           "normalise": True, "min_max": {"min": [-3.] * 5, "max": [3.] * 5}}}
        self.teacher_net = nn.Sequential(nn.Linear(6, 8), nn.SELU(), nn.Linear(8, 5))
        self.correction = nn.Sequential(nn.Linear(11, 8), nn.ELU(), nn.Linear(8, 5))
        self.path = self.root / "combined.pth"
        torch.save({"ot_cfm_model": {"net." + k: v for k, v in self.teacher_net.state_dict().items()},
                    "interpolant": {"interpolant_net." + k: v for k, v in self.correction.state_dict().items()},
                    "data_contract": self.contract}, self.path)

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_checkpoint_reconstruction_and_global_time(self):
        teacher = ALICFMTeacherWrapper(self.path)
        generator = ALICFMGeneratorWrapper(self.path)
        samples, terminal = torch.randn(7, 5), torch.randn(7, 5)
        times = torch.linspace(0, 1, 7)
        torch.testing.assert_close(teacher(times, samples), self.teacher_net(torch.cat([samples, times[:, None]], 1)))
        t = times[:, None]
        correction = self.correction(torch.cat([samples, terminal, t], 1))
        torch.testing.assert_close(generator(samples, terminal, times),
                                   (1-t)*samples+t*terminal+t*(1-t)*correction)
        torch.testing.assert_close(teacher(torch.tensor(.4), samples),
                                   self.teacher_net(torch.cat([samples, torch.full((7, 1), .4)], 1)))

    def test_saved_scalers_and_heldout_exclusion(self):
        data = load_single_cell_data(self.path, self.root, validation_fraction=.1)
        expected = (((self.pca - 1) / 2) + 3) / 6
        torch.testing.assert_close(data.clouds[0], torch.tensor(expected[:20]))
        self.assertEqual(len(data.train[1]), 0)
        self.assertEqual(data.observed, [0, 2, 3])
        self.assertEqual([len(x) for x in data.validation], [2, 0, 2, 2])
        points, times = observed_batch(data, data.train, 100, torch.Generator().manual_seed(0), torch.device("cpu"))
        self.assertFalse(torch.isclose(times, torch.tensor(1/3)).any())
        torch.testing.assert_close(data.metric_coordinates(data.clouds[0]), torch.tensor((self.pca[:20] - 1) / 2))
        original = learn_topology(data)
        data.clouds[1].fill_(1e6)
        self.assertEqual(learn_topology(data), original)

    def test_checksum_rejects_modified_data(self):
        with h5py.File(self.root / "cite.h5ad", "r+") as f:
            f["obsm/X_pca"][0, 0] = 123.
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_single_cell_data(self.path, self.root)

    def test_mmot_alignment(self):
        middle = torch.arange(5).float()[:, None]
        source = middle[torch.tensor([3, 1, 4, 0, 2])]
        target = middle[torch.tensor([1, 3, 0, 4, 2])]
        aligned_source, aligned_target = mmot_endpoints(source, middle, target)
        torch.testing.assert_close(aligned_source, middle)
        torch.testing.assert_close(aligned_target, middle)

    def test_distillation_backward_frozen_targets_and_validation(self):
        data = load_single_cell_data(self.path, self.root)
        teacher = ALICFMTeacherWrapper(self.path).requires_grad_(False)
        generator = ALICFMGeneratorWrapper(self.path).requires_grad_(False)
        student = build_single_cell_student({"name": "hclt", "parents": learn_topology(data),
                                             "num_latents": 4, "time_embedding_dim": 8,
                                             "time_feature_seed": 0, "fourier_zero_center": False})
        rng = torch.Generator().manual_seed(1)
        initialize_source(student, data.train[0], rng)
        samples, times = sample_paths(data, data.train, generator, 8, rng, torch.device("cpu"))
        with torch.no_grad():
            targets = teacher(times, samples)
        loss = (student.velocity(samples, times) - targets).square().mean()
        loss.backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters()))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in generator.parameters()))
        before = student.training
        metrics = validation_metrics(student, teacher, generator, data, torch.device("cpu"), 9, batches=1, batch_size=8)
        self.assertTrue(all(np.isfinite(x) for x in metrics.values()))
        self.assertEqual(student.training, before)

    def test_config_identity_and_duplicate_names(self):
        grid = self.root / "grid.json"
        grid.write_text(json.dumps({"common": {"steps": 10}, "runs": [{"name": "a", "lr": .001}]}))
        a = load_jobs(grid, {})[0][0]
        b = load_jobs(grid, {"steps": 20})[0][0]
        self.assertNotEqual(a, b)
        grid.write_text(json.dumps({"runs": [{"name": "a"}, {"name": "a"}]}))
        with self.assertRaisesRegex(ValueError, "unique"):
            load_jobs(grid, {})

    def test_emd_detaches_inputs_and_uses_euclidean_cost(self):
        real = torch.tensor([[0., 0.], [1., 0.]], requires_grad=True)
        predicted = real + torch.tensor([0., 2.])
        self.assertAlmostEqual(emd(real, predicted), 2.)
        with self.assertRaisesRegex(ValueError, "finite"):
            emd(real, torch.full((2, 2), float("nan")))

    def test_automatic_devices_use_visible_logical_indices(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"}), \
                patch("torch.cuda.is_available", return_value=True), \
                patch("torch.cuda.device_count", return_value=2), \
                patch("torch.cuda.mem_get_info", side_effect=[(2**30, 16 * 2**30), (12 * 2**30, 16 * 2**30)]):
            self.assertEqual(automatic_devices(2), ["cuda:1"])

    def test_emd_monitor_restores_training_mode_and_preserves_global_rng(self):
        data = load_single_cell_data(self.path, self.root)
        student = build_single_cell_student({"name": "hclt", "parents": learn_topology(data),
                                             "num_latents": 4, "time_embedding_dim": 8,
                                             "time_feature_seed": 0, "fourier_zero_center": False})
        initialize_source(student, data.train[0], torch.Generator().manual_seed(2))
        before = torch.random.get_rng_state().clone()
        result = direct_diagnostics(student, data, torch.device("cpu"), num_samples=8)
        self.assertTrue(student.training)
        torch.testing.assert_close(torch.random.get_rng_state(), before)
        self.assertEqual(len(result["scores"]), 4)
        self.assertEqual(len(data.train[data.heldout]), 0)
        self.assertTrue(np.isfinite(result["heldout_direct_emd"]))
        self.assertTrue(all(parameter.grad is None for parameter in student.parameters()))

    def test_tracking_uses_custom_axes_instead_of_internal_steps(self):
        run = Mock()
        log_training(run, {"step": 500, "loss": .25, "validation_score": .1})
        run.log.assert_called_once_with({"train/step": 500, "train/loss": .25,
                                         "validation/score": .1})
        metrics = emd_scalars({"scores": [{"day": 3, "ode_emd": .9, "direct_emd": 1.1}],
                              "heldout_direct_emd": 1.1, "heldout_ode_emd": .9,
                              "reference_heldout_ode_emd": 1.}, "eval")
        self.assertEqual(metrics["eval/day_3/direct_emd"], 1.1)
        self.assertAlmostEqual(metrics["eval/heldout_ode_emd_minus_teacher"], -.1)
        self.assertEqual(metrics["eval/heldout_direct_emd_ratio_to_teacher"], 1.1)

    def test_tracking_ids_are_independent_and_resume_the_student(self):
        args = SimpleNamespace(wandb_mode="online", wandb_project="flow2pc",
                               wandb_entity=None, wandb_group="test", wandb_service_wait=300)
        def create_run(**kwargs):
            return SimpleNamespace(id=kwargs["id"], name=kwargs["name"], entity="test-team",
                                   url="https://wandb.ai/test", define_metric=Mock())
        fake_wandb = SimpleNamespace(
            init=Mock(side_effect=create_run), Settings=lambda **kwargs: kwargs)
        with patch.dict("sys.modules", {"wandb": fake_wandb}):
            first = init_tracking(args, self.root / "a", {}, "distillation")
            second = init_tracking(args, self.root / "b", {}, "distillation")
            resumed = init_tracking(args, self.root / "a", {}, "student-evaluation")
        self.assertNotEqual(first.id, second.id)
        self.assertRegex(first.id, r"^[0-9a-f]{16}$")
        self.assertEqual(first.id, resumed.id)
        self.assertIsNone(fake_wandb.init.call_args_list[0].kwargs["resume"])
        self.assertIsNone(fake_wandb.init.call_args_list[1].kwargs["resume"])
        self.assertEqual(fake_wandb.init.call_args.kwargs["resume"], "allow")
        self.assertEqual(fake_wandb.init.call_args.kwargs["settings"]["x_service_wait"], 300)
        self.assertEqual(fake_wandb.init.call_args.kwargs["settings"]["init_timeout"], 300)

    def test_tracking_retries_communication_failure_with_the_same_id(self):
        class CommError(Exception):
            pass
        class AuthenticationError(CommError):
            pass
        args = SimpleNamespace(wandb_mode="online", wandb_project="flow2pc",
                               wandb_entity=None, wandb_group="test", wandb_service_wait=300)
        attempts = []
        def create_run(**kwargs):
            attempts.append(kwargs)
            if len(attempts) == 1:
                raise CommError("context deadline exceeded")
            return SimpleNamespace(id=kwargs["id"], name=kwargs["name"], entity="test-team",
                                   url="https://wandb.ai/test", define_metric=Mock())
        def settings_factory(**kwargs):
            return kwargs
        settings_factory.model_fields = {key: None for key in (
            "x_graphql_timeout_seconds", "x_graphql_retry_max",
            "x_graphql_retry_wait_min_seconds", "x_graphql_retry_wait_max_seconds")}
        fake_wandb = SimpleNamespace(init=Mock(side_effect=create_run), Settings=settings_factory,
                                    teardown=Mock(),
                                    errors=SimpleNamespace(CommError=CommError,
                                                           AuthenticationError=AuthenticationError))
        with patch.dict("sys.modules", {"wandb": fake_wandb}), \
                patch("repc.single_cell_tracking.time.sleep") as sleep:
            run = init_tracking(args, self.root / "retry", {}, "distillation")
        self.assertEqual(attempts[0]["id"], attempts[1]["id"])
        self.assertEqual(run.id, attempts[0]["id"])
        self.assertIsNone(attempts[0]["resume"])
        self.assertEqual(attempts[1]["resume"], "allow")
        self.assertEqual(attempts[0]["settings"]["x_graphql_timeout_seconds"], 60.)
        sleep.assert_called_once_with(5)
        fake_wandb.teardown.assert_called_once_with()
        metadata = json.loads((self.root / "retry/wandb_run.json").read_text())
        self.assertTrue(metadata["initialized"])
        self.assertEqual(metadata["startup_status"], "ready")

    def test_tracking_does_not_retry_authentication_errors(self):
        class CommError(Exception):
            pass
        class AuthenticationError(CommError):
            pass
        args = SimpleNamespace(wandb_mode="online", wandb_project="flow2pc",
                               wandb_entity=None, wandb_group="test", wandb_service_wait=300)
        fake_wandb = SimpleNamespace(init=Mock(side_effect=AuthenticationError("bad credentials")),
                                    Settings=lambda **kwargs: kwargs,
                                    errors=SimpleNamespace(CommError=CommError,
                                                           AuthenticationError=AuthenticationError))
        with patch.dict("sys.modules", {"wandb": fake_wandb}), \
                patch("repc.single_cell_tracking.time.sleep") as sleep:
            with self.assertRaises(AuthenticationError):
                init_tracking(args, self.root / "auth", {}, "distillation")
        fake_wandb.init.assert_called_once()
        sleep.assert_not_called()

    def test_verification_timeout_is_retryable_but_rejected_key_is_not(self):
        class AuthenticationError(Exception):
            pass
        fake_wandb = SimpleNamespace(errors=SimpleNamespace(AuthenticationError=AuthenticationError))
        timeout = AuthenticationError("Failed to verify credentials: the service process is busy "
                                      "and did not respond in time")
        self.assertTrue(retryable_startup_error(fake_wandb, timeout))
        self.assertFalse(retryable_startup_error(fake_wandb, AuthenticationError("bad credentials")))
        self.assertFalse(retryable_startup_error(fake_wandb, AuthenticationError("HTTP 401 unauthorized")))
        self.assertFalse(retryable_startup_error(fake_wandb, AuthenticationError("HTTP 403 forbidden")))

    def test_file_uploads_use_run_save_without_artifact_api(self):
        source = self.root / "source.json"
        source.write_text('{"heldout_emd": 1.0}\n')
        run = SimpleNamespace(dir=str(self.root / "wandb/files"), save=Mock(), summary={})
        log_files(run, "teacher-evaluation", {"metrics.json": source})
        destination = self.root / "wandb/files/attachments/teacher-evaluation/metrics.json"
        self.assertEqual(destination.read_bytes(), source.read_bytes())
        run.save.assert_called_once_with(str(destination), base_path=run.dir, policy="now")
        self.assertEqual(run.summary["uploads/teacher-evaluation/files"], 1)

    def test_evaluation_logs_emd_before_media_and_uses_csv_instead_of_table(self):
        output = self.root / "eval"
        output.mkdir()
        result = {"model": "teacher", "heldout_ode_emd": 1.25,
                  "scores": [{"day": 3, "held_out": True, "ode_emd": 1.25}]}
        (output / "metrics.json").write_text(json.dumps(result))
        (output / "ode_marginals.png").write_bytes(b"mock image")
        run = SimpleNamespace(dir=str(self.root / "wandb/files"), save=Mock(), log=Mock(), summary={})
        # Deliberately provide no Table, Artifact, Api, or log_artifact method.
        fake_wandb = SimpleNamespace(Image=Mock(return_value="image-object"))
        with patch.dict("sys.modules", {"wandb": fake_wandb}):
            log_evaluation(run, result, output)
        first = run.log.call_args_list[0].args[0]
        self.assertEqual(first["teacher/heldout_ode_emd"], 1.25)
        self.assertNotIn("teacher/ode_marginals", first)
        self.assertEqual(run.log.call_args_list[1].args[0]["teacher/ode_marginals"], "image-object")
        self.assertTrue((output / "marginal_metrics.csv").exists())
        self.assertEqual(run.save.call_count, 3)
        self.assertEqual(run.summary["teacher/status"], "complete")


if __name__ == "__main__":
    unittest.main()
