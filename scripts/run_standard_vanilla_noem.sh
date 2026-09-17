#!/usr/bin/env bash
set -euo pipefail

requested_gpu="${FLOW2PC_GPU:-auto}"
minimum_free_mb="${FLOW2PC_MIN_FREE_MB:-12000}"
python_bin="${FLOW2PC_PYTHON:-python}"
teacher_run_dir="outputs/runs/flow/teacher-vanilla-torchcfm-seed0"
teacher_checkpoint="${teacher_run_dir}/checkpoints/last.pt"
classifier_checkpoint="outputs/evaluation/mnist-classifier-pytorch-example-seed1/checkpoints/last.pt"

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

teacher_evaluation="${teacher_run_dir}/eval/last/metrics.json"
teacher_sample_count=0
if [[ -f "${teacher_evaluation}" ]]; then
  teacher_sample_count="$(
    "${python_bin}" -c \
      "import json; print(json.load(open('${teacher_evaluation}')).get('protocol/num_generated', 0))"
  )"
fi
if (( teacher_sample_count != 10000 )); then
  "${python_bin}" eval.py \
    --run-dir "${teacher_run_dir}" \
    --classifier-checkpoint "${classifier_checkpoint}" \
    --times 0,0.25,0.5,0.75,1 \
    --num-generated 10000 \
    --batch-size 128 \
    --num-workers 4 \
    --kid-subsets 100 \
    --kid-subset-size 1000 \
    --precision-recall-k 3 \
    --ode-method dopri5 \
    --ode-rtol 1e-4 \
    --ode-atol 1e-4 \
    --seed 0 \
    --wandb-project flow2pc \
    --wandb-run-name eval-teacher-vanilla-torchcfm-seed0
fi

train_and_evaluate() {
  local label="$1"
  shift
  local run_dir="outputs/runs/distill/${label}"
  if [[ -e "${run_dir}" ]]; then
    echo "Refusing to overwrite existing run directory: ${run_dir}" >&2
    return 1
  fi

  "${python_bin}" train.py distill \
    --teacher-checkpoint "${teacher_checkpoint}" \
    --capacity-profile standard \
    --endpoint-pretraining none \
    --time-order 4 \
    --steps 20000 \
    --batch-size 64 \
    --num-workers 4 \
    --lr 0.001 \
    --optimizer adamw \
    --scheduler cosine \
    --weight-decay 0.01 \
    --gradient-clip 1.0 \
    --checkpoint-every 5000 \
    --hclt-samples 20000 \
    --hclt-num-bins 64 \
    --hclt-sigma 0.02 \
    --hclt-chunk-size 32 \
    --seed 0 \
    --wandb-project flow2pc \
    --wandb-group vanilla-noem-standard-powers2-20k \
    --wandb-run-name "${label}" \
    --wandb-tags baseline,standard-powers2,no-endpoint-em,equal-budget-20k \
    --output-dir "${run_dir}" \
    "$@"

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
    --wandb-run-name "eval-${label}"
}

train_and_evaluate \
  hclt-dense-h64-vanilla-noem-step20000-seed0 \
  --pc-structure hclt --pc-layer dense --num-latents 64

train_and_evaluate \
  hclt-monarch-h64-vanilla-noem-step20000-seed0 \
  --pc-structure hclt --pc-layer monarch --num-latents 64

train_and_evaluate \
  pd-c8-vanilla-noem-step20000-seed0 \
  --pc-structure pd --num-channels 8

train_and_evaluate \
  gaussian-k256-vanilla-noem-step20000-seed0 \
  --pc-structure gaussian --num-components 256

"${python_bin}" summarize_benchmark.py
echo "All standard vanilla/no-EM runs, evaluations, and summary tables completed."


