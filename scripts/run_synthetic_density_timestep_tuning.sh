#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python_bin="${FLOW2PC_PYTHON:-python}"
gpu="${FLOW2PC_GPU:-1}"

run_candidate() {
  local density_weight="$1"
  local timestep_sampling="$2"
  local density_label="${density_weight//./p}"
  local time_label="${timestep_sampling}"
  local timestep_args=(--timestep-sampling "${timestep_sampling}")
  if [[ "${timestep_sampling}" == "beta" ]]; then
    time_label="beta-a1-b0p8"
    timestep_args+=(--timestep-beta-alpha 1.0 --timestep-beta-beta 0.8)
  fi
  local label="branching-tune-lambda${density_label}-${time_label}-k8-seed0"
  local run_dir="outputs/runs/synthetic/${label}"
  if [[ -f "${run_dir}/results.json" ]]; then
    echo "Skipping completed ${label}"
    return
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" synthetic_experiment.py \
    --device cuda:0 \
    --steps 5000 \
    --capacities 8 \
    --batch-size 2048 \
    --time-order 4 \
    --lr 0.001 \
    --weight-decay 0.01 \
    --gradient-clip 1.0 \
    --density-loss-weight "${density_weight}" \
    "${timestep_args[@]}" \
    --eval-samples 16384 \
    --certificate-samples 256 \
    --certificate-times 17 \
    --certificate-ode-steps 24 \
    --max-pruned-variants 0 \
    --seed 0 \
    --wandb-project flow2pc \
    --wandb-group synthetic-density-timestep-tuning \
    --wandb-run-name "${label}" \
    --output-dir "${run_dir}"
}

# The lambda=0/uniform and lambda=0.1/uniform controls already exist.
run_candidate 1.0 uniform
run_candidate 10.0 uniform
run_candidate 0.0 beta
run_candidate 0.1 beta
run_candidate 1.0 beta
run_candidate 10.0 beta

"${python_bin}" summarize_synthetic_tuning.py
