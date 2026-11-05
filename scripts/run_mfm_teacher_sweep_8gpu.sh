#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${FLOW2PC_PYTHON:-/data/nvidia/conda_envs/uffm/bin/python}" \
  "$repo_root/scripts/run_mfm_teacher_sweep.py" \
  --gpus "${GPUS:-0,1,2,3,4,5,6,7}" \
  --workers-per-gpu "${WORKERS_PER_GPU:-2}" \
  --max-parallel "${MAX_PARALLEL:-16}" \
  --threads "${CPU_THREADS_PER_RUN:-4}" \
  --min-free-gb "${MIN_FREE_GB:-6}" \
  --min-disk-free-gb "${MIN_DISK_FREE_GB:-20}" \
  --data-root "${FLOW2PC_MFM_DATA_ROOT:-/data/pqhung_group/pdtung/flow2pc/data/mfm}" \
  --output-root "${MFM_TEACHER_SWEEP_ROOT:-/data/lthung/flow2pc/mfm-teachers/sweep-256}" \
  "$@"
