#!/usr/bin/env bash
set -euo pipefail

gpu_index="${FLOW2PC_GPU:-1}"
python_bin="${FLOW2PC_PYTHON:-python}"
run_names=(
  hclt-dense-h64-vanilla-noem-step20000-seed0
  hclt-monarch-h64-vanilla-noem-step20000-seed0
  pd-c8-vanilla-noem-step20000-seed0
  gaussian-k256-vanilla-noem-step20000-seed0
)
checkpoint_names=(step_005000.pt step_010000.pt step_015000.pt)

export CUDA_VISIBLE_DEVICES="${gpu_index}"
teacher_run="outputs/runs/flow/teacher-vanilla-torchcfm-seed0"
if [[ ! -f "${teacher_run}/eval/last/metrics.json" ]]; then
  "${python_bin}" eval.py \
    --run-dir "${teacher_run}" \
    --times 0,0.25,0.5,0.75,1 \
    --num-generated 10000 \
    --batch-size 128 \
    --seed 0 \
    --wandb-project flow2pc \
    --wandb-run-name eval-teacher-vanilla-torchcfm-seed0
fi

for run_name in "${run_names[@]}"; do
  run_dir="outputs/runs/distill/${run_name}"
  for checkpoint_name in "${checkpoint_names[@]}"; do
    checkpoint_stem="${checkpoint_name%.pt}"
    if [[ -f "${run_dir}/eval/${checkpoint_stem}/metrics.json" ]]; then
      echo "Skipping completed evaluation: ${run_name}/${checkpoint_stem}"
      continue
    fi
    "${python_bin}" eval.py \
      --run-dir "${run_dir}" \
      --checkpoint "${checkpoint_name}" \
      --times 0,0.25,0.5,0.75,1 \
      --num-generated 10000 \
      --batch-size 128 \
      --pc-sampling-methods direct \
      --seed 0 \
      --wandb-project flow2pc \
      --wandb-run-name "eval-${run_name}-${checkpoint_stem}"
  done
done

"${python_bin}" summarize_benchmark.py
