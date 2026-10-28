#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

python_bin="${FLOW2PC_PYTHON:-python}"
gpu="${FLOW2PC_GPU:-1}"
density_loss_weight="${FLOW2PC_DENSITY_WEIGHT:-0.1}"
density_label="${density_loss_weight//./p}"

CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" synthetic_experiment.py \
  --device cuda:0 \
  --steps 5000 \
  --capacities 2,4,8 \
  --batch-size 2048 \
  --time-order 4 \
  --lr 0.001 \
  --weight-decay 0.01 \
  --gradient-clip 1.0 \
  --density-loss-weight "${density_loss_weight}" \
  --eval-samples 16384 \
  --certificate-samples 512 \
  --certificate-times 17 \
  --certificate-ode-steps 24 \
  --max-pruned-variants 8 \
  --seed 0 \
  --wandb-project flow2pc \
  --wandb-group synthetic-query-faithfulness \
  --wandb-run-name "synthetic-branching-coupled-lambda${density_label}-k2-k4-k8-seed0" \
  --output-dir "outputs/runs/synthetic/branching-coupled-lambda${density_label}-k2-k4-k8-seed0"
