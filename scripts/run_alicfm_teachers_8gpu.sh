#!/usr/bin/env bash
# Six single-cell teachers + ST + tracking. No distillation is queued.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${FLOW2PC_PYTHON:-/data/nvidia/conda_envs/uffm/bin/python}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

exec "$python_bin" "$repo_root/scripts/run_alicfm_checks.py" \
  --teachers-only \
  --datasets cite5 cite50 cite100 multi5 multi50 multi100 st cell_tracking \
  --gpus "${GPUS:-0,1,2,3,4,5,6,7}" \
  --workers-per-gpu 1 \
  --max-parallel 8 \
  --threads "${CPU_THREADS_PER_RUN:-4}" \
  --teacher-min-free-gb "${TEACHER_MIN_FREE_GB:-2}" \
  --data-root "${FLOW2PC_DATA_ROOT:-/data/lthung/ali-cfm-data}" \
  --output-root "${FLOW2PC_TEACHER_ROOT:-/data/lthung/flow2pc/framework/native-teachers}" \
  "$@"
