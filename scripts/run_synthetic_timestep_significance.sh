#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python_bin="${FLOW2PC_PYTHON:-python}"
gpu="${FLOW2PC_GPU:-1}"

run_candidate() {
  local seed="$1"
  local timestep_sampling="$2"
  local time_label="${timestep_sampling}"
  local timestep_args=(--timestep-sampling "${timestep_sampling}")
  if [[ "${timestep_sampling}" == "beta" ]]; then
    time_label="beta-a1-b0p2"
    timestep_args+=(--timestep-beta-alpha 1.0 --timestep-beta-beta 0.2)
  fi
  local label="branching-significance-lambda10-${time_label}-k8-seed${seed}"
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
    --density-loss-weight 10.0 \
    "${timestep_args[@]}" \
    --eval-samples 16384 \
    --certificate-samples 256 \
    --certificate-times 17 \
    --certificate-ode-steps 24 \
    --max-pruned-variants 0 \
    --seed "${seed}" \
    --wandb-project flow2pc \
    --wandb-group synthetic-timestep-significance \
    --wandb-run-name "${label}" \
    --output-dir "${run_dir}"
}

# Seeds 0-2 are reused from the matched confirmation experiment.
for seed in 3 4 5 6 7; do
  run_candidate "${seed}" uniform
  run_candidate "${seed}" beta
done

"${python_bin}" summarize_synthetic_timestep_significance.py
