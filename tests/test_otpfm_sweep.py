"""Exercise the two-dataset scheduler using fake processes, never CUDA training."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from repc.single_cell import atomic_json
from scripts import run_otpfm_sweep as sweep


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.grid = self.root / "grid.json"
        atomic_json(self.grid, {"common": {"steps": 3, "num_latents": 4, "time_embedding_dim": 4,
                                           "batch_size": 8, "lr": .001, "seed": 0},
                                "runs": [{"name": f"student{i}"} for i in range(3)]})
        self.args = sweep.parse_args(["--grid", str(self.grid), "--output-root", str(self.root / "results"),
                                      "--wandb-mode", "disabled"])
        teacher = {"seed": 44, "checkpoint_sha256": "hash", "config_sha256": "config"}
        with patch.object(sweep, "select_teacher", return_value=(teacher, {})):
            self.plans = sweep.build_plan(self.args)

    def processes(self, fail_beijing=False):
        owner = self
        created = []

        class Process:
            def __init__(self, command, **kwargs):
                self.pid = 800000 + len(created)
                self.command = command
                self.stage = "prepare" if command[2].endswith("prepare_otpfm.py") else "train"
                self.output = Path(command[command.index("--output-dir") + 1])
                self.dataset = (command[command.index("--dataset") + 1] if self.stage == "prepare"
                                else Path(command[command.index("--reference-dir") + 1]).name)
                self.device = command[command.index("--device") + 1]
                self.returncode, self.polls = None, 0
                if self.stage == "train":
                    owner.assertTrue((owner.args.output_root / "reference" / self.dataset / "metrics.json").exists())
                created.append(self)
                running = [p for p in created if p.returncode is None]
                owner.assertLessEqual(len(running), 4)
                for device in ("cuda:0", "cuda:1"):
                    owner.assertLessEqual(sum(p.device == device for p in running), 2)

            def poll(self):
                if self.returncode is not None:
                    return self.returncode
                self.polls += 1
                duration = 5 if self.dataset == "beijing" and self.stage == "prepare" else 2
                if self.polls < duration:
                    return None
                self.returncode = 1 if fail_beijing and self.dataset == "beijing" and self.stage == "prepare" else 0
                if self.returncode:
                    return self.returncode
                if self.stage == "prepare":
                    atomic_json(self.output / "metrics.json", {"prepared": True})
                else:
                    atomic_json(self.output / "status.json", {"status": "complete", "best_validation_mse": .01})
                    atomic_json(self.output / "eval/metrics.json", {"checkpoint_step": 3})
                return 0

            def wait(self, timeout=None):
                return self.returncode

        return Process, created

    def run_fake(self, process):
        with patch.object(sweep.subprocess, "Popen", process), \
                patch.object(sweep, "gpu_free_memory", return_value={"cuda:0": 30 * 2**30, "cuda:1": 30 * 2**30}), \
                patch.object(sweep.time, "sleep"), patch.object(sweep.os, "killpg"):
            sweep.run(self.args, self.plans, ["cuda:0", "cuda:1"])

    def test_transient_memory_query_failure_does_not_stop_active_workers(self):
        process, created = self.processes()
        calls = 0

        def memory(devices):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("temporary monitoring failure")
            return dict.fromkeys(devices, 30 * 2**30)

        with patch.object(sweep.subprocess, "Popen", process), \
                patch.object(sweep, "gpu_free_memory", side_effect=memory), \
                patch.object(sweep.torch.cuda, "mem_get_info", side_effect=RuntimeError("CUDA error: out of memory")) as cuda_query, \
                patch.object(sweep.time, "sleep"), patch.object(sweep.os, "killpg") as kill:
            sweep.run(self.args, self.plans, ["cuda:0", "cuda:1"])
        cuda_query.assert_not_called()
        kill.assert_not_called()
        self.assertEqual(len(created), 8)
        self.assertEqual(sweep.read_json(self.args.output_root / "sweep_status.json")["status"], "complete")

    def test_parallel_preparation_training_and_resume(self):
        process, created = self.processes()
        self.run_fake(process)
        self.assertEqual(len(created), 8)
        self.assertEqual({p.device for p in created if p.stage == "train"}, {"cuda:0", "cuda:1"})
        self.assertEqual(sweep.read_json(self.args.output_root / "sweep_status.json")["status"], "complete")
        for dataset in self.plans:
            self.assertTrue((self.args.output_root / "sweeps" / dataset / "selected_student.json").exists())
        self.run_fake(process)
        self.assertEqual(sum(p.stage == "train" for p in created), 6, "resume duplicated a completed student")

    def test_reference_failure_isolated_to_one_dataset(self):
        process, created = self.processes(fail_beijing=True)
        with self.assertRaises(SystemExit):
            self.run_fake(process)
        self.assertEqual(sum(p.stage == "train" and p.dataset == "gom" for p in created), 3)
        self.assertFalse(any(p.stage == "train" and p.dataset == "beijing" for p in created))

    def test_external_references_are_shared_by_prepare_and_train_commands(self):
        args = sweep.parse_args(["--output-root", str(self.root / "new_students"),
                                 "--reference-root", str(self.root / "existing_references")])
        prepare = sweep.task_command(args, ("gom", "prepare", "reference", {}, 0), "cuda:0")
        train = sweep.task_command(args, ("gom", "train", "student", {}, 0), "cuda:1")
        expected = str(self.root / "existing_references" / "gom")
        self.assertEqual(prepare[prepare.index("--output-dir") + 1], expected)
        self.assertEqual(train[train.index("--reference-dir") + 1], expected)
        self.assertEqual(train[train.index("--output-dir") + 1],
                         str(self.root / "new_students/sweeps/gom/runs/student"))

    def test_dry_run_never_launches_or_creates_results(self):
        with patch.object(sweep, "parse_args", return_value=sweep.parse_args([
                    "--grid", str(self.grid), "--output-root", str(self.args.output_root), "--dry-run"])), \
                patch.object(sweep, "build_plan", return_value={
                    ds: {**plan, "teacher": {**plan["teacher"], "checkpoint": "existing-best.pt", "fixed_interpolant_alpha": 1.}}
                    for ds, plan in self.plans.items()}), \
                patch.object(sweep.torch.cuda, "device_count", return_value=2), \
                patch.object(sweep.subprocess, "Popen") as process, patch("builtins.print"):
            sweep.main()
        process.assert_not_called()
        self.assertFalse(self.args.output_root.exists())


class MemoryQueryTest(unittest.TestCase):
    def test_numeric_visibility_mapping_and_full_gpu_without_cuda_allocations(self):
        response = SimpleNamespace(stdout="0, 4096\n1, 0\n2, 24576\n")
        with patch.object(sweep.subprocess, "run", return_value=response), \
                patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2,1,0"}), \
                patch.object(sweep.torch.cuda, "mem_get_info", side_effect=RuntimeError("CUDA out of memory")) as cuda_query:
            free = sweep.gpu_free_memory(["cuda:0", "cuda:1", "cuda:2"])
        self.assertEqual(free, {"cuda:0": 24576 * 2**20, "cuda:1": 0, "cuda:2": 4096 * 2**20})
        cuda_query.assert_not_called()

    def test_uuid_visibility_mapping_and_unavailable_gpu(self):
        response = SimpleNamespace(stdout="0, N/A\n1, 12288\n")
        with patch.object(sweep.subprocess, "run", return_value=response), \
                patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-bbbb,GPU-aaaa"}), \
                patch.object(sweep.torch.cuda, "_raw_device_uuid_nvml", return_value=["GPU-aaaa-1111", "GPU-bbbb-2222"]):
            free = sweep.gpu_free_memory(["cuda:0", "cuda:1"])
        self.assertEqual(free, {"cuda:0": 12288 * 2**20, "cuda:1": 0})

    def test_query_timeout_has_clear_error(self):
        with patch.object(sweep.subprocess, "run", side_effect=sweep.subprocess.TimeoutExpired("nvidia-smi", 10)):
            with self.assertRaisesRegex(RuntimeError, "nvidia-smi memory query failed"):
                sweep.gpu_free_memory(["cuda:0", "cuda:1"])


if __name__ == "__main__":
    unittest.main()
