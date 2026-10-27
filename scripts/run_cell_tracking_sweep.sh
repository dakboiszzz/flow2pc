#!/usr/bin/env bash
# Prepare the frozen reference once, then run 108 students on many GPUs.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${FLOW2PC_PYTHON:-/data/nvidia/conda_envs/uffm/bin/python}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-120}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "$python_bin" -u scripts/run_cell_tracking_sweep.py "$@"
