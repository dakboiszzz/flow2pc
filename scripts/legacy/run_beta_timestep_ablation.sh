#!/usr/bin/env bash
set -euo pipefail

requested_gpu="${FLOW2PC_GPU:-auto}"
minimum_free_mb="${FLOW2PC_MIN_FREE_MB:-4000}"
python_bin="${FLOW2PC_PYTHON:-python}"
teacher_checkpoint="outputs/runs/flow/teacher-vanilla-torchcfm-seed0/checkpoints/last.pt"
classifier_checkpoint="outputs/evaluation/mnist-classifier-pytorch-example-seed1/checkpoints/last.pt"
topology_checkpoint="data/MNIST/hclt/topology_x20000_b64_s0p02_c32_v3000_seed0.pt"
label="hclt-dense-h64-fourier8-lr0p001-beta-a1-b0p8-vanilla-noem-step20000-seed0"
run_dir="outputs/runs/distill/${label}"

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
    && (( requested_gpu < ${#free_memory[@]} )) \
    && (( free_memory[requested_gpu] >= minimum_free_mb )); then
    selected_gpu="${requested_gpu}"
  fi
  if [[ -n "${selected_gpu}" ]]; then
    echo "GPU ${selected_gpu} is ready with ${free_memory[selected_gpu]} MiB free."
    break
  fi
  echo "$(date --iso-8601=seconds) waiting for ${minimum_free_mb} MiB free; GPUs: ${free_memory[*]} MiB"
  sleep 60
done

export CUDA_VISIBLE_DEVICES="${selected_gpu}"

if [[ -f "${run_dir}/checkpoints/last.pt" ]]; then
  echo "Skipping completed training: ${label}"
elif [[ -e "${run_dir}" ]]; then
  echo "Incomplete run directory exists; refusing to overwrite: ${run_dir}" >&2
  exit 1
else
  "${python_bin}" train.py distill \
    --teacher-checkpoint "${teacher_checkpoint}" \
    --pc-structure hclt \
    --pc-layer dense \
    --capacity-profile standard \
    --num-latents 64 \
    --time-conditioning fourier \
    --time-embedding-dim 8 \
    --hclt-leaf-distribution gaussian \
    --hclt-leaf-components 1 \
    --endpoint-pretraining none \
    --timestep-sampling beta \
    --timestep-beta-alpha 1.0 \
    --timestep-beta-beta 0.8 \
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
    --wandb-group vanilla-noem-dense-hclt-timestep-ablation-20k \
    --wandb-run-name "${label}" \
    --wandb-tags timestep-ablation,beta,data-endpoint-biased,bttdm-mirrored,dense-hclt,no-endpoint-em,equal-budget-20k \
    --output-dir "${run_dir}"
fi

best_checkpoint="$(${python_bin} -c "import csv; from pathlib import Path; p=Path('${run_dir}'); rows=list(csv.DictReader((p/'validation_metrics.csv').open())); rows=[r for r in rows if int(r['step']) in (5000,10000,15000,20000)]; r=min(rows,key=lambda x:(float(x['endpoint_bpd']),int(x['step']))); s=int(r['step']); print('last.pt' if s==20000 else f'step_{s:06d}.pt')")"
best_stem="${best_checkpoint%.pt}"

"${python_bin}" eval.py \
  --run-dir "${run_dir}" \
  --checkpoint "${best_checkpoint}" \
  --output-name "${best_stem}-full" \
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
  --wandb-run-name "eval-full-${label}-${best_stem}"

echo "Beta timestep ablation and full evaluation completed: ${run_dir}"
