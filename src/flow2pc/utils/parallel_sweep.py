"""Torch-free independent-worker queue with GPU and filesystem admission guards."""

import errno
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import deque

import yaml

from flow2pc.utils.gpu_queue import choose_device, stop_workers


def storage_error(error):
    return isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT)


class ParallelSweep:
    def __init__(self, args, runs, *, repo, inventory, inspect, report, archive):
        self.args, self.runs, self.repo = args, runs, repo
        self.inventory, self.inspect, self.report, self.archive = inventory, inspect, report, archive
        self.active = []
        self.interrupted = False
        self.disk_retry = 0.0

    def write_report(self, state):
        try:
            self.report(self.args, self.runs, self.active, state)
        except OSError as error:
            if not storage_error(error):
                raise
            self.disk_retry = time.monotonic() + 30
            print(f"Storage full: report deferred; retaining active workers: {error}", file=sys.stderr, flush=True)

    def environment(self, device):
        runtime = self.args.output_root / "_runtime"
        folders = {"TMPDIR": runtime / "tmp", "WANDB_CACHE_DIR": runtime / "wandb-cache",
                   "WANDB_DATA_DIR": runtime / "wandb-staging"}
        for path in folders.values():
            path.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, **{key: str(path) for key, path in folders.items()},
               "CUDA_VISIBLE_DEVICES": device["uuid"], "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "PYTHONPATH": str(self.repo / "src"), "MPLBACKEND": "Agg",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
               "FLOW2PC_WANDB_INIT_LOCK": str(self.args.output_root / ".wandb-init.lock")}
        env.update(TMP=env["TMPDIR"], TEMP=env["TMPDIR"])
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[key] = str(self.args.threads)
        for key in ("WANDB_RUN_ID", "WANDB_NAME", "WANDB_RESUME", "WANDB_SERVICE"):
            env.pop(key, None)
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        return env

    def launch(self, task, device, inventory, lock):
        if task.action == "restart":
            self.archive(task)
            task.action = "pretrain"
        path = self.args.output_root / "_configs" / task.dataset / f"{task.name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(task.config, sort_keys=False))
        command = [sys.executable, str(self.repo / "main.py"), "--config", str(path)]
        if task.action == "evaluate":
            command += ["--evaluate", "teacher"]
        env = self.environment(inventory[device])
        path = self.args.output_root / "_launcher/logs" / f"{task.name}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = path.open("a")
        try:
            process = subprocess.Popen(command, cwd=self.repo, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True,
                                       pass_fds=(lock.fileno(),))
        except BaseException:
            stream.close()
            raise
        task.status, task.error = "running", ""
        self.active.append(dict(task=task, process=process, log=stream, device=device, started=time.monotonic()))
        print(f"Started {task.action}: {task.name}, GPU {device}, pid {process.pid}", flush=True)

    def run(self, devices, lock):
        pending = deque(run for run in self.runs if run.status == "pending")
        previous = {}
        next_report, warned = 0.0, False
        def stop(signum, frame):
            self.interrupted = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.signal(sig, stop)
        try:
            while pending or self.active:
                if self.interrupted:
                    break
                for worker in self.active[:]:
                    code = worker["process"].poll()
                    if code is None:
                        continue
                    self.active.remove(worker)
                    worker["log"].close()
                    task = worker["task"]
                    task.status = "pending"
                    self.inspect(task, False)
                    if task.status != "complete":
                        checkpoint = (task.output / "checkpoints/last.pt").is_file()
                        infrastructure = any(term in task.error.lower() for term in
                                             ("out of memory", "wandb", "no space left", "disk quota"))
                        if (checkpoint and task.status != "blocked" or infrastructure) and task.retries < self.args.worker_retries:
                            task.retries += 1
                            task.status = "pending"
                            task.action = "evaluate" if checkpoint else "restart"
                            task.not_before = time.monotonic() + 60
                            pending.append(task)
                        else:
                            task.status = "failed" if task.status != "blocked" else "blocked"
                            task.error = task.error or f"Worker exit {code}; inspect its log"
                    print(f"{task.status}: {task.name}", flush=True)
                    next_report = 0.0
                try:
                    inventory = self.inventory()
                except RuntimeError as error:
                    print(f"GPU query failed; keeping workers and retrying: {error}", flush=True)
                    time.sleep(5)
                    continue
                free = {device: inventory[device]["free"] for device in devices}
                storage_ready = shutil.disk_usage(self.args.output_root).free / 2**30 >= self.args.min_disk_free_gb and time.monotonic() >= self.disk_retry
                if pending and not storage_ready and not warned:
                    print("New launches paused for disk space; active workers continue", file=sys.stderr, flush=True)
                    warned = True
                if storage_ready:
                    warned = False
                while pending and storage_ready and len(self.active) < self.args.max_parallel and not self.interrupted:
                    if shutil.disk_usage(self.args.output_root).free / 2**30 < self.args.min_disk_free_gb:
                        storage_ready = False
                        break
                    device = choose_device(devices, self.active, free, self.args.workers_per_gpu,
                                           self.args.min_free_gb, time.monotonic())
                    task = next((run for run in pending if run.not_before <= time.monotonic()), None)
                    if device is None or task is None:
                        break
                    pending.remove(task)
                    try:
                        self.launch(task, device, inventory, lock)
                    except OSError as error:
                        if not storage_error(error):
                            raise
                        pending.appendleft(task)
                        self.disk_retry = time.monotonic() + 30
                        storage_ready = False
                        print(f"Storage full: launch deferred; active workers retained: {error}", file=sys.stderr, flush=True)
                    next_report = 0.0
                if time.monotonic() >= next_report:
                    self.write_report("running" if storage_ready else "waiting_for_disk")
                    next_report = time.monotonic() + 30
                time.sleep(2)
        finally:
            aborted = sys.exc_info()[0] is not None
            if self.active:
                stop_workers(self.active)
                for worker in self.active:
                    worker["task"].status = "interrupted"
                self.active.clear()
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            state = ("interrupted" if self.interrupted else "aborted" if aborted else
                     "failed" if any(r.status in ("failed", "blocked") for r in self.runs) else "complete")
            self.write_report(state)
        return 130 if self.interrupted else int(any(r.status in ("failed", "blocked") for r in self.runs))
