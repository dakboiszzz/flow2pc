#!/usr/bin/env bash
set -euo pipefail

requested_gpu="${FLOW2PC_GPU:-auto}"
minimum_free_mb="${FLOW2PC_MIN_FREE_MB:-3500}"
python_bin="${FLOW2PC_PYTHON:-python}"
teacher_checkpoint="outputs/runs/flow/teacher-vanilla-torchcfm-seed0/checkpoints/last.pt"
classifier_checkpoint="outputs/evaluation/mnist-classifier-pytorch-example-seed1/checkpoints/last.pt"
topology_checkpoint="data/MNIST/hclt/topology_x20000_b64_s0p02_c32_v3000_seed0.pt"
benchmark_dir="outputs/benchmark/dense-hclt-expressive-leaf"

for required_file in \
  "${teacher_checkpoint}" \
  "${classifier_checkpoint}" \
  "${topology_checkpoint}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required checkpoint is missing: ${required_file}" >&2
    exit 1
  fi
done

while true; do
  mapfile -t free_memory < <(
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
  )
  selected_gpu=""
  if [[ "${requested_gpu}" == "auto" ]]; then
    for candidate in "${!free_memory[@]}"; do
      if (( free_memory[candidate] >= minimum_free_mb )); then
        selected_gpu="${candidate}"
        break
      fi
    done
  elif [[ "${requested_gpu}" =~ ^[0-9]+$ ]] \
    && (( requested_gpu < ${#free_memory[@]} )); then
    if (( free_memory[requested_gpu] >= minimum_free_mb )); then
      selected_gpu="${requested_gpu}"
    fi
  else
    echo "FLOW2PC_GPU must be 'auto' or a valid non-negative GPU index" >&2
    exit 2
  fi
  if [[ -n "${selected_gpu}" ]]; then
    gpu_index="${selected_gpu}"
    echo "GPU ${gpu_index} is ready with ${free_memory[gpu_index]} MiB free."
    break
  fi
  echo "$(date --iso-8601=seconds) waiting for ${minimum_free_mb} MiB free; GPUs: ${free_memory[*]} MiB"
  sleep 60
done

export CUDA_VISIBLE_DEVICES="${gpu_index}"

train_and_screen() {
  local label="$1"
  shift
  local run_dir="outputs/runs/distill/${label}"
  if [[ -f "${run_dir}/checkpoints/last.pt" ]]; then
    echo "Skipping completed training: ${label}"
  elif [[ -e "${run_dir}" ]]; then
    echo "Incomplete run directory exists; refusing to overwrite: ${run_dir}" >&2
    return 1
  else
    "${python_bin}" train.py distill \
      --teacher-checkpoint "${teacher_checkpoint}" \
      --pc-structure hclt \
      --pc-layer dense \
      --capacity-profile standard \
      --num-latents 32 \
      --time-conditioning fourier \
      --endpoint-pretraining none \
      --steps 20000 \
      --batch-size 64 \
      --num-workers 4 \
      --lr 0.001 \
      --optimizer adamw \
      --scheduler cosine \
      --weight-decay 0.01 \
      --gradient-clip 1.0 \
      --checkpoint-every 5000 \
      --validation-size 3000 \
      --validate-every 1000 \
      --hclt-topology "${topology_checkpoint}" \
      --hclt-samples 20000 \
      --hclt-num-bins 64 \
      --hclt-sigma 0.02 \
      --hclt-chunk-size 32 \
      --seed 0 \
      --wandb-project flow2pc \
      --wandb-group vanilla-noem-dense-hclt-expressive-leaf-20k \
      --wandb-run-name "${label}" \
      --wandb-tags expressive-leaf,dense-hclt,no-endpoint-em,validation-selected,equal-budget-20k \
      --output-dir "${run_dir}" \
      "$@"
  fi

  if [[ ! -f "${run_dir}/eval/last/metrics.json" ]]; then
    "${python_bin}" eval.py \
      --run-dir "${run_dir}" \
      --classifier-checkpoint "${classifier_checkpoint}" \
      --times 0,0.25,0.5,0.75,1 \
      --skip-sample-metrics \
      --batch-size 128 \
      --num-workers 4 \
      --seed 0 \
      --wandb-project flow2pc \
      --wandb-run-name "screen-${label}"
  fi
}

full_evaluate() {
  local label="$1"
  local run_dir="outputs/runs/distill/${label}"
  "${python_bin}" eval.py \
    --run-dir "${run_dir}" \
    --classifier-checkpoint "${classifier_checkpoint}" \
    --times 0,0.25,0.5,0.75,1 \
    --num-generated 10000 \
    --batch-size 128 \
    --num-workers 4 \
    --kid-subsets 100 \
    --kid-subset-size 1000 \
    --precision-recall-k 3 \
    --pc-sampling-methods direct,euler,rk4,dopri5 \
    --pc-ode-steps 20 \
    --ode-rtol 1e-4 \
    --ode-atol 1e-4 \
    --seed 0 \
    --wandb-project flow2pc \
    --wandb-run-name "eval-full-${label}"
}

gmm_label="hclt-dense-h32-fourier16-gmm2-lr0p001-vanilla-noem-leafablation-step20000-seed0"
gaussian_label="hclt-dense-h32-fourier16-gaussian-lr0p001-vanilla-noem-leafablation-step20000-seed0"
matched_label="hclt-dense-h32-fourier32-gaussian-lr0p001-vanilla-noem-leafablation-step20000-seed0"

train_and_screen \
  "${gmm_label}" \
  --time-embedding-dim 16 \
  --hclt-leaf-distribution gaussian-mixture \
  --hclt-leaf-components 2

train_and_screen \
  "${gaussian_label}" \
  --time-embedding-dim 16 \
  --hclt-leaf-distribution gaussian \
  --hclt-leaf-components 1

train_and_screen \
  "${matched_label}" \
  --time-embedding-dim 32 \
  --hclt-leaf-distribution gaussian \
  --hclt-leaf-components 1

full_evaluate "${gmm_label}"
full_evaluate "${gaussian_label}"
full_evaluate "${matched_label}"

"${python_bin}" summarize_leaf_ablation.py --output-dir "${benchmark_dir}"
echo "Expressive-leaf ablation and full evaluations completed."
