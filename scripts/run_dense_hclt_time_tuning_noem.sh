#!/usr/bin/env bash
set -euo pipefail

requested_gpu="${FLOW2PC_GPU:-auto}"
minimum_free_mb="${FLOW2PC_MIN_FREE_MB:-20000}"
python_bin="${FLOW2PC_PYTHON:-python}"
teacher_checkpoint="outputs/runs/flow/teacher-vanilla-torchcfm-seed0/checkpoints/last.pt"
classifier_checkpoint="outputs/evaluation/mnist-classifier-pytorch-example-seed1/checkpoints/last.pt"
topology_checkpoint="data/MNIST/hclt/topology_x20000_b64_s0p02_c32_v3000_seed0.pt"
benchmark_dir="outputs/benchmark/dense-hclt-time-tuning"

for required_file in "${teacher_checkpoint}" "${classifier_checkpoint}"; do
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
  local num_latents="$2"
  shift 2
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
      --num-latents "${num_latents}" \
      --endpoint-pretraining none \
      --steps 20000 \
      --batch-size 64 \
      --num-workers 4 \
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
      --wandb-group vanilla-noem-dense-hclt-time-tuning-20k \
      --wandb-run-name "${label}" \
      --wandb-tags tuning,dense-hclt,no-endpoint-em,validation-selected,equal-budget-20k \
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

train_and_screen \
  hclt-dense-h64-poly4-lr0p001-vanilla-noem-tune-step20000-seed0 \
  64 \
  --time-conditioning polynomial --time-order 4 --lr 0.001

train_and_screen \
  hclt-dense-h32-poly4-lr0p001-vanilla-noem-tune-step20000-seed0 \
  32 \
  --time-conditioning polynomial --time-order 4 --lr 0.001

train_and_screen \
  hclt-dense-h32-fourier8-lr0p001-vanilla-noem-tune-step20000-seed0 \
  32 \
  --time-conditioning fourier --time-embedding-dim 8 --lr 0.001

train_and_screen \
  hclt-dense-h64-fourier8-lr0p001-vanilla-noem-tune-step20000-seed0 \
  64 \
  --time-conditioning fourier --time-embedding-dim 8 --lr 0.001

train_and_screen \
  hclt-dense-h64-fourier16-lr0p001-vanilla-noem-tune-step20000-seed0 \
  64 \
  --time-conditioning fourier --time-embedding-dim 16 --lr 0.001

train_and_screen \
  hclt-dense-h64-fourier16-lr0p0003-vanilla-noem-tune-step20000-seed0 \
  64 \
  --time-conditioning fourier --time-embedding-dim 16 --lr 0.0003

train_and_screen \
  hclt-dense-h64-fourier16-lr0p003-vanilla-noem-tune-step20000-seed0 \
  64 \
  --time-conditioning fourier --time-embedding-dim 16 --lr 0.003

"${python_bin}" summarize_dense_tuning.py --output-dir "${benchmark_dir}"

selection_file="${benchmark_dir}/best_selection.json"
best_run="$(${python_bin} -c "import json; print(json.load(open('${selection_file}'))['run_dir'])")"
best_checkpoint="$(${python_bin} -c "import json; print(json.load(open('${selection_file}'))['checkpoint'])")"
best_label="$(basename "${best_run}")"
best_eval_stem="${best_checkpoint%.pt}"
best_metrics="${best_run}/eval/${best_eval_stem}/metrics.json"
full_sample_count=0
if [[ -f "${best_metrics}" ]]; then
  full_sample_count="$(${python_bin} -c "import json; print(json.load(open('${best_metrics}')).get('protocol/num_generated', 0))")"
fi

if (( full_sample_count != 10000 )); then
  "${python_bin}" eval.py \
    --run-dir "${best_run}" \
    --checkpoint "${best_checkpoint}" \
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
    --wandb-run-name "eval-full-${best_label}-${best_eval_stem}"
fi

"${python_bin}" summarize_dense_tuning.py --output-dir "${benchmark_dir}"
echo "Dense-HCLT time-conditioning tuning and selected full evaluation completed."
