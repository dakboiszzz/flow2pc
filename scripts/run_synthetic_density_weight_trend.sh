#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python_bin="${FLOW2PC_PYTHON:-python}"
gpu="${FLOW2PC_GPU:-1}"
decision_file="outputs/benchmark/synthetic-timestep-significance/summary.json"
selected="$(${python_bin} -c "import json; print(json.load(open('${decision_file}'))['decision']['selected_timestep_sampling'])")"

if [[ "${selected}" == "beta-a1-b0p2" ]]; then
  timestep_sampling="beta"
  time_label="beta-a1-b0p2"
  timestep_args=(--timestep-sampling beta --timestep-beta-alpha 1.0 --timestep-beta-beta 0.2)
elif [[ "${selected}" == "uniform" ]]; then
  timestep_sampling="uniform"
  time_label="uniform"
  timestep_args=(--timestep-sampling uniform)
else
  echo "Unsupported selected timestep scheme: ${selected}" >&2
  exit 1
fi

run_candidate() {
  local seed="$1"
  local weight="$2"
  if "${python_bin}" summarize_synthetic_density_weight_trend.py \
    --check-run \
    --timestep-sampling "${timestep_sampling}" \
    --density-loss-weight "${weight}" \
    --seed "${seed}"; then
    echo "Reusing completed ${time_label} lambda=${weight} seed=${seed}"
    return
  fi

  local label="branching-weight-trend-lambda${weight}-${time_label}-k8-seed${seed}"
  local run_dir="outputs/runs/synthetic/${label}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" synthetic_experiment.py \
    --device cuda:0 \
    --steps 5000 \
    --capacities 8 \
    --batch-size 2048 \
    --time-order 4 \
    --lr 0.001 \
    --weight-decay 0.01 \
    --gradient-clip 1.0 \
    --density-loss-weight "${weight}" \
    "${timestep_args[@]}" \
    --eval-samples 16384 \
    --certificate-samples 256 \
    --certificate-times 17 \
    --certificate-ode-steps 24 \
    --max-pruned-variants 0 \
    --seed "${seed}" \
    --wandb-project flow2pc \
    --wandb-group "synthetic-density-weight-trend-${time_label}" \
    --wandb-run-name "${label}" \
    --output-dir "${run_dir}"
}

for weight in 0 1 3 10 30 100; do
  for seed in 0 1 2; do
    run_candidate "${seed}" "${weight}"
  done
done

"${python_bin}" summarize_synthetic_density_weight_trend.py
