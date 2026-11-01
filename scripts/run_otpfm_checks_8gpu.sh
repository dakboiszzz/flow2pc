#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${FLOW2PC_PYTHON:-/data/nvidia/conda_envs/uffm/bin/python}" \
  "$repo_root/scripts/run_otpfm_checks.py" \
  --gpus "${GPUS:-0,1,2,3,4,5,6,7}" --max-parallel "${MAX_PARALLEL:-8}" \
  --threads "${CPU_THREADS_PER_RUN:-4}" --min-free-gb "${MIN_FREE_GB:-4}" \
  --data-root "${OTP_FM_DATA_ROOT:-/data/lthung/otp-fm}" \
  --output-root "${OTP_FM_CHECK_ROOT:-/data/lthung/flow2pc/otp-fm/migration-check}" "$@"
