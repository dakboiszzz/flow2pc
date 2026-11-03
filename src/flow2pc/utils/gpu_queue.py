"""Proven multi-GPU queue helpers recovered from the successful real-data sweeps.

See docs/SCHEDULER_SOURCE.json. The controller uses nvidia-smi, never a CUDA
memory query. These helpers do not import Torch or start workers at import time.
"""

import os
import signal
import subprocess
import time
from collections import Counter


def choose_device(devices, active, free_bytes, workers_per_gpu, min_free_gb, now):
    """Balance workers across GPUs, reserving memory while new workers warm up."""
    counts = Counter(worker["device"] for worker in active)
    required = min_free_gb * 2**30
    candidates = []
    for device in devices:
        # Avoid launching a burst before CUDA contexts/allocations become visible.
        warming = sum(
            worker["device"] == device and now - worker["started"] < 30
            for worker in active
        )
        available = free_bytes[device] - warming * required
        if counts[device] < workers_per_gpu and available >= required:
            candidates.append((counts[device], -available, device))
    return min(candidates)[2] if candidates else None


def next_task(datasets):
    """One task per dataset per round; finished datasets do not strand GPUs."""
    for _ in range(len(datasets)):
        dataset = datasets[0]
        datasets.rotate(-1)
        if dataset.pending:
            return dataset.pending.popleft()
    return None


def stop_workers(active):
    """Signal every process group before waiting, including SDK child services."""
    for worker in active:
        try:
            os.killpg(worker["process"].pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 15
    for worker in active:
        try:
            worker["process"].wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(worker["process"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker["process"].wait()
        worker["log"].close()
