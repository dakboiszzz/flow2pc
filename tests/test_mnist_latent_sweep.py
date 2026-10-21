"""Simulated worker-pool tests; no GPUs, downloads, models, or W&B calls."""

from io import StringIO
from collections import deque
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import run_mnist_latent_sweep as sweep


class LatentSweepTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_full_grid_has_192_unique_configurations(self):
        jobs = sweep.load_jobs(sweep.ROOT / "configs/mnist_latent/full_sweep.json")
        self.assertEqual(len(jobs), 192)
        self.assertEqual(len({job["name"] for job in jobs}), 192)
        self.assertEqual({job["settings"]["num_latents"] for job in jobs}, {128, 256, 512, 1024})
        self.assertEqual({job["settings"]["time_embedding_dim"] for job in jobs}, {16, 32, 64})
        self.assertEqual(len({job["settings"]["lr"] for job in jobs}), 4)
        self.assertEqual(jobs, sweep.load_jobs(sweep.ROOT / "configs/mnist_latent/full_sweep.json"))

    def test_z98_grid_varies_requested_lr_values_and_uses_50000_steps(self):
        jobs = sweep.load_jobs(sweep.ROOT / "configs/mnist_latent/z98_sweep.json")
        self.assertEqual(len(jobs), 192)
        self.assertEqual({job["settings"]["lr"] for job in jobs}, {0.0001, 0.0005, 0.001, 0.005})
        self.assertTrue(all(job["settings"]["steps"] == 50000 and job["settings"]["weight_decay"] == 0 for job in jobs))
        self.assertTrue(any(all(job["settings"][key] == value for key, value in {
            "num_latents": 128, "time_embedding_dim": 16, "batch_size": 64, "lr": 0.0005, "seed": 0,
        }.items()) for job in jobs))

    def test_training_and_evaluation_names_begin_with_mnist_and_identify_latent_size(self):
        args = SimpleNamespace(latent_dim=98)
        job = {"name": "h128-f16-lr0.003-b64-s0", "settings": {"steps": 50000}}
        for stage in ("train", "validation"):
            name = sweep.student_run_name(args, job, stage)
            self.assertTrue(name.startswith("mnist-z98-"))
            self.assertTrue(name.endswith(f"steps50000-{stage}"))
        self.assertEqual(sweep.mnist_run_name("custom-teacher"), "mnist-custom-teacher")
        self.assertEqual(sweep.mnist_run_name("mnist-selected-test"), "mnist-selected-test")

    def test_distinct_visible_gpu_slots_are_required(self):
        self.assertEqual(sweep.parse_devices("auto", 8, verify=False), [f"cuda:{i}" for i in range(8)])
        for devices, workers in (("cuda:0,cuda:0", 2), ("cuda:0", 8), ("cuda", 1)):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                sweep.parse_devices(devices, workers, verify=False)

    def test_visible_indices_are_validated_without_physical_gpu_assumptions(self):
        with (patch("torch.cuda.is_available", return_value=True),
              patch("torch.cuda.device_count", return_value=2)):
            self.assertEqual(sweep.parse_devices("cuda:0,cuda:1", 2), ["cuda:0", "cuda:1"])
            with self.assertRaisesRegex(ValueError, "only 2 logical"):
                sweep.parse_devices("auto", 8)

    def test_busy_gpu_is_skipped_until_enough_memory_is_available(self):
        devices = ["cuda:0", "cuda:1"]
        with patch("torch.cuda.mem_get_info", side_effect=lambda index: ((8 if index == 0 else 40) * 1024**3, 80 * 1024**3)):
            self.assertEqual(sweep.ready_device(devices, 24), "cuda:1")
            self.assertIsNone(sweep.ready_device(devices, 48))
        with patch("torch.cuda.mem_get_info") as memory:
            self.assertEqual(sweep.ready_device(devices, 0), "cuda:0")
        memory.assert_not_called()

    def test_grid_rejects_image_intervals_and_invalid_monarch_capacity(self):
        path = self.root / "grid.json"
        for common in ({"image_every": 1000}, {"time_embedding_dim": 3},
                       {"pc_layer": "monarch", "num_latents": 32}, {"output_dir": "unsafe"}):
            path.write_text(json.dumps({"common": common, "parameters": {"seed": [0]}}))
            with self.subTest(common=common), self.assertRaises(ValueError):
                sweep.load_jobs(path)

    def test_automatic_assignment_skips_large_jobs_and_preserves_roomy_gpus(self):
        args = SimpleNamespace(min_free_gb=None)
        large = {"name": "large", "settings": {**sweep.DEFAULTS, "num_latents": 1024, "time_embedding_dim": 64}}
        medium = {"name": "medium", "settings": {**sweep.DEFAULTS, "num_latents": 512}}
        small = {"name": "small", "settings": {**sweep.DEFAULTS, "num_latents": 128}}
        queue = deque([large, medium, small])
        devices = deque(["cuda:0", "cuda:1"])
        with patch("torch.cuda.mem_get_info", side_effect=lambda index: ((3 if index == 0 else 8) * 1024**3, 80 * 1024**3)):
            self.assertEqual(sweep.choose_pending_job(args, queue, devices), (medium, "cuda:1"))
            queue.remove(medium)
            self.assertEqual(sweep.choose_pending_job(args, queue, devices), (small, "cuda:0"))
            queue.remove(small)
            self.assertIsNone(sweep.choose_pending_job(args, queue, devices))

    def test_freshly_trained_evaluation_keeps_its_reserved_gpu(self):
        args = SimpleNamespace(min_free_gb=None)
        trained = {"name": "trained", "settings": sweep.DEFAULTS, "preferred_device": "cuda:1"}
        with patch("torch.cuda.mem_get_info") as memory:
            self.assertEqual(sweep.choose_pending_job(args, deque([trained]), deque(["cuda:0", "cuda:1"])),
                             (trained, "cuda:1"))
        memory.assert_not_called()

    def run_fake_pool(self, failed_job=None):
        args = SimpleNamespace(output_dir=self.root / "sweep", retry_failed=False, min_free_gb=0)
        jobs = [{"name": f"job{i}", "settings": {**sweep.DEFAULTS, "seed": i}} for i in range(19)]
        occupied, calls, high_water = set(), [], [0]

        def start(args, shared, job, device):
            self.assertNotIn(device, occupied)
            occupied.add(device)
            high_water[0] = max(high_water[0], len(occupied))
            calls.append((job["name"], job["stage"], device))
            if job["stage"] == "train":
                sweep.atomic_json(job["train_output"] / "training_summary.json",
                                  {"status": "complete", "best_validation_score": 0.1})
                checkpoint = job["train_output"] / "checkpoints/best.pt"
                checkpoint.parent.mkdir()
                checkpoint.touch()
            else:
                job["eval_output"] = job["train_output"].parent / "validation"
                sweep.atomic_json(job["eval_output"] / "metrics.json", {"student/fid_mnist": 1.})
            sweep.set_job_state(args, job, status="running", attempt=job["attempt"],
                                stage=job["stage"], train_output=str(job["train_output"]),
                                eval_output=str(job.get("eval_output", "")))

            class Process:
                def poll(self):
                    occupied.remove(device)
                    return 1 if job["name"] == failed_job else 0

            return {"job": job, "device": device, "process": Process(), "log": StringIO()}

        with (patch.object(sweep, "start_worker", side_effect=start),
              patch.object(sweep, "summarize"), patch.object(sweep.time, "sleep")):
            failures = sweep.run_pool(args, {}, jobs, [f"cuda:{i}" for i in range(8)])
        self.assertEqual(high_water[0], 8)
        self.assertFalse(occupied)
        return args, jobs, calls, failures

    def test_eight_concurrent_workers_refill_and_evaluate_on_their_gpu(self):
        args, jobs, calls, failures = self.run_fake_pool()
        self.assertEqual(failures, [])
        self.assertEqual(len(calls), 2 * len(jobs))
        training_devices = {name: device for name, stage, device in calls if stage == "train"}
        for name, stage, device in calls:
            if stage == "evaluate":
                self.assertEqual(device, training_devices[name])
        self.assertTrue(all(sweep.job_state(args, job)["status"] == "complete" for job in jobs))

    def test_failed_configuration_does_not_stop_other_gpu_workers(self):
        args, jobs, calls, failures = self.run_fake_pool(failed_job="job3")
        self.assertEqual(failures, ["job3"])
        self.assertEqual(len(calls), 2 * len(jobs) - 1)
        self.assertTrue(all(sweep.job_state(args, job)["status"] == "complete"
                            for job in jobs if job["name"] != "job3"))

    def test_child_environments_preserve_gpu_mapping_and_isolate_wandb(self):
        args = SimpleNamespace(threads=1, output_dir=self.root)
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b", "WANDB_RUN_ID": "parent"}):
            first = sweep.child_environment(args, self.root / "worker1")
            second = sweep.child_environment(args, self.root / "worker2")
        self.assertEqual(first["CUDA_VISIBLE_DEVICES"], "GPU-a,GPU-b")
        self.assertNotIn("WANDB_RUN_ID", first)
        self.assertNotEqual(first["WANDB_DIR"], second["WANDB_DIR"])
        self.assertEqual(first["OMP_NUM_THREADS"], "1")

    def test_dry_run_never_checks_cuda_writes_files_or_spawns_workers(self):
        output = self.root / "untouched"
        with (patch.object(sweep, "ARTIFACT_ROOT", self.root),
              patch.object(sweep, "prepare_shared") as prepare,
              patch.object(sweep.subprocess, "Popen") as popen,
              patch("torch.cuda.device_count") as count):
            sweep.main(["--dry-run", "--output-dir", str(output),
                        "--bundle-dir", str(self.root / "bundle"), "--data-dir", str(self.root / "data")])
        prepare.assert_not_called()
        popen.assert_not_called()
        count.assert_not_called()
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
