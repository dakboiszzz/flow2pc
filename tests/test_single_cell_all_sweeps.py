"""Scheduler tests use simulated subprocesses; no training/evaluation is run."""

from collections import deque
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from repc.single_cell import atomic_json, sha256_file
from repc.single_cell_tracking import dataset_run_name
from scripts import run_single_cell_all_sweeps as sweep


class AllSweepsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.teachers = self.root / "teachers"
        self.teachers.mkdir()
        self.grid = self.root / "grid.json"
        atomic_json(self.grid, {"common": {"steps": 100}, "runs": [
            {"name": "h32-lr0.001", "num_latents": 32, "lr": .001,
             "time_embedding_dim": 16, "batch_size": 64, "seed": 0}]})
        entries = []
        for dataset, dim in sorted(sweep.EXPECTED):
            folder = self.teachers / f"{dataset}_{dim}D" / "t=1"
            folder.mkdir(parents=True)
            contract = {"time_index": 1, "heldout_day": 3,
                        "pseudo_time_convention": "equally_spaced_ordered_marginals_on_[0,1]",
                        "preprocessing": {"dataset": dataset, "n_pca_components_used": dim,
                                          "time_order": [2, 3, 4, 7], "whiten": dim == 5}}
            torch.save({"data_contract": contract, "time_format": "global",
                        "ot_cfm_model": {"weight": torch.zeros(1)},
                        "interpolant": {"weight": torch.zeros(1)}}, folder / "teacher.pth")
            atomic_json(folder / "metadata.json", {
                "data_contract": contract, "seed": 43, "heldout_index": 1,
                "cfm": {"w": 64 if dim == 5 else 1024},
                "generator": {"h_dim": 64 if dim == 5 else 1024},
                "export_sha256": {"teacher.pth": sha256_file(folder / "teacher.pth")}})
            for holdout in (1, 2):
                entries.append({"dataset": dataset, "dim": dim, "seed": 43,
                                "heldout_index": holdout})
        atomic_json(self.teachers / "index.json", {"teachers": entries})
        self.args = sweep.parse_args([
            "--teachers-root", str(self.teachers), "--output-root", str(self.root / "output"),
            "--grid", str(self.grid), "--wandb-mode", "disabled"])
        self.datasets = sweep.discover_datasets(self.args)

    def test_exactly_six_t1_bundles_and_dataset_names(self):
        self.assertEqual(len(self.datasets), 6)
        self.assertEqual({(ds.contract["preprocessing"]["dataset"],
                           ds.contract["preprocessing"]["n_pca_components_used"])
                          for ds in self.datasets}, sweep.EXPECTED)
        for ds in self.datasets:
            self.assertEqual(ds.teacher.parent.name, "t=1")
            name, settings = ds.jobs[0]
            self.assertEqual(settings["wandb_group"], ds.tag)
            prefix = ds.tag.split("-")[0]
            self.assertTrue(dataset_run_name(ds.contract, name).startswith(prefix + "-h32"))
            self.assertEqual(dataset_run_name(ds.contract, prefix + "-h128"), prefix + "-h128")
        self.assertFalse(self.args.output_root.exists())

    def test_duplicate_or_missing_t1_teacher_is_rejected(self):
        path = self.teachers / "index.json"
        index = sweep.read_json(path)
        index["teachers"][0]["heldout_index"] = 2
        atomic_json(path, index)
        with self.assertRaisesRegex(ValueError, "exactly one t=1"):
            sweep.discover_datasets(self.args)

    def test_commands_use_matching_reference_and_best_checkpoint(self):
        for ds in self.datasets:
            name, settings = ds.jobs[0]
            for stage in ("teacher", "train", "evaluate"):
                command = sweep.command_for(self.args, sweep.Task(ds, stage, name, settings), "cuda:3")
                self.assertEqual(command[command.index("--device") + 1], "cuda:3")
                self.assertEqual(command[command.index("--teacher-checkpoint") + 1], str(ds.teacher))
                if stage != "teacher":
                    self.assertEqual(command[command.index("--reference") + 1], str(ds.reference))
                if stage == "evaluate":
                    self.assertTrue(command[command.index("--student-checkpoint") + 1].endswith("best.pt"))
                    self.assertIn("--wandb-run-dir", command)
                if stage == "teacher":
                    self.assertNotIn("--student-checkpoint", command)

    def test_gpu_capacity_memory_gate_and_launch_reservation(self):
        devices = ["cuda:0", "cuda:1"]
        free = {"cuda:0": 30 * 2**30, "cuda:1": 20 * 2**30}
        self.assertEqual(sweep.choose_device(devices, [], free, 2, 8, 100), "cuda:0")
        active = [{"device": "cuda:0", "started": 0}]
        self.assertEqual(sweep.choose_device(devices, active, free, 2, 8, 100), "cuda:1")
        active += [{"device": "cuda:1", "started": 99}]
        low = {"cuda:0": 7 * 2**30, "cuda:1": 12 * 2**30}
        self.assertIsNone(sweep.choose_device(devices, active, low, 2, 8, 100))
        full = [{"device": device, "started": 0} for device in devices for _ in range(2)]
        self.assertIsNone(sweep.choose_device(devices, full, free, 2, 8, 100))

    def test_round_robin_interleaves_datasets(self):
        for ds in self.datasets:
            ds.pending.extend([sweep.Task(ds, "train", "first"), sweep.Task(ds, "train", "second")])
        queue = deque(self.datasets)
        first_round = [sweep.next_task(queue).dataset.tag for _ in range(6)]
        second_round = [sweep.next_task(queue).dataset.tag for _ in range(6)]
        self.assertEqual(first_round, [ds.tag for ds in self.datasets])
        self.assertEqual(first_round, second_round)
        self.assertIsNone(sweep.next_task(queue))

    def simulated_processes(self, fail_teacher=False, oom_train=False):
        owner = self
        processes = []
        events = []

        class Process:
            def __init__(self, command, **kwargs):
                self.command = command
                self.pid = 900000 + len(processes)
                self.polls = 0
                self.returncode = None
                self.ds = next(ds for ds in owner.datasets
                               if str(ds.teacher) == command[command.index("--teacher-checkpoint") + 1])
                self.output = Path(command[command.index("--output-dir") + 1])
                self.stage = ("train" if command[2].endswith("train_single_cell.py") else
                              "evaluate" if "--student-checkpoint" in command else "teacher")
                self.device = command[command.index("--device") + 1]
                self.log = kwargs["stdout"]
                self.other_teacher_active = any(p.stage == "teacher" and p.returncode is None
                                                for p in processes)
                processes.append(self)
                events.append((self.ds.tag, self.stage))
                running = [p for p in processes if p.returncode is None]
                owner.assertLessEqual(len(running), 4)
                for device in ("cuda:0", "cuda:1"):
                    owner.assertLessEqual(sum(p.device == device for p in running), 2)
                if self.stage != "teacher":
                    owner.assertTrue(self.ds.reference.exists(), "student started before teacher reference")

            def poll(self):
                self.polls += 1
                if self.returncode is not None:
                    return self.returncode
                if self.polls < 2:
                    return None
                fail = ((fail_teacher and self.ds is owner.datasets[0] and self.stage == "teacher")
                        or (oom_train and self.stage == "train"))
                self.returncode = 1 if fail else 0
                if fail:
                    if oom_train and self.stage == "train":
                        self.log.write("CUDA out of memory\n")
                        self.log.flush()
                    return self.returncode
                identity = {"checkpoint_sha256": self.ds.checkpoint_hash}
                if self.stage == "teacher":
                    atomic_json(self.ds.reference, {
                        "model": "teacher", "identity": identity, "data_contract": self.ds.contract,
                        "protocol": sweep.evaluation_protocol(owner.args, self.ds), "heldout_ode_emd": 3})
                if self.stage == "evaluate":
                    atomic_json(self.output / "metrics.json", {
                        "identity": identity, "protocol": sweep.evaluation_protocol(owner.args, self.ds),
                        "checkpoint_step": 100, "heldout_direct_emd": 4,
                        "heldout_ode_emd": 2, "reference_heldout_ode_emd": 3})
                atomic_json(self.output / "status.json", {"status": "complete", "best_validation_score": .1})
                return self.returncode

            def wait(self, timeout=None):
                return self.returncode

        return Process, processes, events

    def run_simulation(self, process):
        with patch.object(sweep.subprocess, "Popen", process), \
                patch.object(sweep.torch.cuda, "mem_get_info", return_value=(30 * 2**30, 80 * 2**30)), \
                patch.object(sweep.time, "sleep"), patch.object(sweep.os, "killpg"):
            sweep.run(self.args, self.datasets, ["cuda:0", "cuda:1"])

    def test_full_scheduler_and_resume_without_real_workers(self):
        process, processes, events = self.simulated_processes()
        self.run_simulation(process)
        self.assertEqual(len(processes), 18)  # Six references, six trains, six evaluations.
        self.assertEqual({p.device for p in processes}, {"cuda:0", "cuda:1"})
        self.assertEqual(sweep.read_json(self.args.output_root / "all_sweeps_status.json")["status"], "complete")
        # A dataset can start students while another dataset still awaits its reference.
        self.assertTrue(any(p.stage == "train" and p.other_teacher_active for p in processes))
        before = len(processes)
        self.run_simulation(process)
        self.assertEqual(len(processes), before, "resume relaunched completed jobs")

    def test_one_teacher_failure_does_not_stop_other_five_sweeps(self):
        process, processes, _ = self.simulated_processes(fail_teacher=True)
        with self.assertRaises(SystemExit):
            self.run_simulation(process)
        self.assertEqual(len(processes), 16)
        self.assertFalse(any(p.ds is self.datasets[0] and p.stage != "teacher" for p in processes))
        for ds in self.datasets[1:]:
            self.assertEqual(sweep.read_json(ds.output / "sweep_status.json")["status"], "complete")

    def test_oom_retries_are_bounded(self):
        process, processes, _ = self.simulated_processes(oom_train=True)
        with self.assertRaises(SystemExit):
            self.run_simulation(process)
        self.assertEqual(sum(p.stage == "train" for p in processes), 12)
        self.assertFalse(any(p.stage == "evaluate" for p in processes))
        self.assertTrue(all(len(ds.failures) == 1 for ds in self.datasets))

    def test_changed_reference_protocol_is_rejected(self):
        ds = self.datasets[0]
        atomic_json(ds.reference, {"model": "teacher", "identity": {"checkpoint_sha256": ds.checkpoint_hash},
                                   "data_contract": ds.contract, "protocol": {"num_samples": 16}})
        with self.assertRaisesRegex(ValueError, "reference differs"):
            sweep.verify_reference(self.args, ds)

    def test_dry_run_creates_no_outputs_or_subprocesses(self):
        with patch.object(sweep.sys, "argv", ["launcher", "--dry-run", "--teachers-root",
                                               str(self.teachers), "--output-root",
                                               str(self.args.output_root), "--grid", str(self.grid)]), \
                patch.object(sweep.torch.cuda, "device_count", return_value=2), \
                patch.object(sweep.subprocess, "Popen") as process, patch("builtins.print"):
            sweep.main()
        process.assert_not_called()
        self.assertFalse(self.args.output_root.exists())

    def test_interrupt_stops_all_worker_groups_and_records_status(self):
        process, processes, _ = self.simulated_processes()
        with patch.object(sweep.subprocess, "Popen", process), \
                patch.object(sweep.torch.cuda, "mem_get_info", return_value=(30 * 2**30, 80 * 2**30)), \
                patch.object(sweep.time, "sleep", side_effect=KeyboardInterrupt), \
                patch.object(sweep.os, "killpg") as kill:
            with self.assertRaises(KeyboardInterrupt):
                sweep.run(self.args, self.datasets, ["cuda:0", "cuda:1"])
        self.assertEqual(len(processes), 4)
        self.assertEqual(kill.call_count, 4)
        self.assertEqual(sweep.read_json(self.args.output_root / "all_sweeps_status.json")["status"],
                         "interrupted")


if __name__ == "__main__":
    unittest.main()
