#!/usr/bin/env bash
set -euo pipefail

requested_gpu="${FLOW2PC_GPU:-0}"
minimum_free_mb="${FLOW2PC_MIN_FREE_MB:-1200}"
endpoint_pretraining="${ENDPOINT_PRETRAINING:-em}"
endpoint_em_epochs="${ENDPOINT_EM_EPOCHS:-20}"
endpoint_em_pseudocount="${ENDPOINT_EM_PSEUDOCOUNT:-0.1}"
endpoint_em_min_std="${ENDPOINT_EM_MIN_STD:-0.01}"
timestep_sampling="${TIMESTEP_SAMPLING:-uniform}"
python_bin="${FLOW2PC_PYTHON:-python}"
teacher_checkpoint="outputs/runs/flow/teacher-vanilla-torchcfm-seed0/checkpoints/last.pt"
classifier_checkpoint="outputs/evaluation/mnist-classifier-pytorch-example-seed1/checkpoints/last.pt"
topology_checkpoint="data/MNIST/hclt/topology_iclr2022_msb3_n60000.pt"

if [[ "${endpoint_pretraining}" != "none" && "${endpoint_pretraining}" != "em" ]]; then
  echo "ENDPOINT_PRETRAINING must be none or em" >&2
  exit 1
fi
if [[ "${timestep_sampling}" != "uniform" ]]; then
  echo "This controlled EM ablation requires TIMESTEP_SAMPLING=uniform" >&2
  exit 1
fi
if [[ "${endpoint_pretraining}" == "em" ]]; then
  endpoint_label="em${endpoint_em_epochs}"
  endpoint_tags="full-batch-em,endpoint-em-${endpoint_em_epochs}"
else
  endpoint_label="noem"
  endpoint_tags="no-endpoint-em"
fi

label="hclt-dense-h64-fourier8-iclr2022tree-lr0p001-${endpoint_label}-vanilla-${timestep_sampling}-step20000-seed0"
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
  free_memory="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
    | sed -n "$((requested_gpu + 1))p" | tr -d ' ')"
  if [[ -n "${free_memory}" ]] && (( free_memory >= minimum_free_mb )); then
    echo "GPU ${requested_gpu} is ready with ${free_memory} MiB free."
    break
  fi
  echo "$(date --iso-8601=seconds) waiting for ${minimum_free_mb} MiB free on GPU ${requested_gpu}; currently ${free_memory:-unknown} MiB"
  sleep 60
done

export CUDA_VISIBLE_DEVICES="${requested_gpu}"

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
    --endpoint-pretraining "${endpoint_pretraining}" \
    --endpoint-em-epochs "${endpoint_em_epochs}" \
    --endpoint-em-pseudocount "${endpoint_em_pseudocount}" \
    --endpoint-em-min-std "${endpoint_em_min_std}" \
    --timestep-sampling "${timestep_sampling}" \
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
    --hclt-topology-protocol iclr2022-categorical \
    --hclt-mi-sample-chunk-size 1024 \
    --hclt-samples 20000 \
    --hclt-num-bins 64 \
    --hclt-sigma 0.02 \
    --hclt-chunk-size 32 \
    --seed 0 \
    --wandb-project flow2pc \
    --wandb-group vanilla-dense-hclt-endpoint-pretraining-20k \
    --wandb-run-name "${label}" \
    --wandb-tags "endpoint-ablation,two-stage,${endpoint_tags},dense-hclt,iclr2022-tree,equal-distillation-budget-20k" \
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

echo "Endpoint-EM two-stage training and full evaluation completed: ${run_dir}"
