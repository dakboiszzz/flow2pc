#!/usr/bin/env bash
# Full 324-student hyperparameter sweep with frozen OTP-FM teachers.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# The historical flag value "ali" selects the standard PC student settings.
# Both the pretrained teacher and the conditional interpolant remain OTP-FM.
exec bash "$repo_root/scripts/run_otpfm_sweep.sh" \
  --datasets gom,beijing \
  --grid "$repo_root/configs/otpfm/sweep.json" \
  --student-recipe ali \
  --reference-root /data/lthung/flow2pc/otp-fm/reference \
  --output-root /data/lthung/flow2pc/otp-fm/corrected \
  --devices auto \
  --workers-per-gpu 2 \
  --wandb-mode online \
  --ode-comparison \
  "$@"
