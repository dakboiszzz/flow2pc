#!/usr/bin/env bash
# Explicit invocation only: evaluate the selected teacher, never train models.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${FLOW2PC_PYTHON:-/data/nvidia/conda_envs/uffm/bin/python}"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-120}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
exec "$python_bin" -u scripts/evaluate_cell_tracking_teacher.py "$@"
