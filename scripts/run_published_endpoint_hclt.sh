#!/usr/bin/env bash
set -euo pipefail

requested_gpu="${FLOW2PC_GPU:-2}"
minimum_free_mb="${FLOW2PC_MIN_FREE_MB:-1800}"
python_bin="${FLOW2PC_PYTHON:-python}"
label="hclt-categorical-m16-bs1024-hybrid-em100p20-seed0"
run_dir="outputs/runs/endpoint/${label}"

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

if [[ -f "${run_dir}/checkpoints/final.pt" && -f "${run_dir}/summary.json" ]]; then
  echo "Skipping completed endpoint baseline: ${run_dir}"
  exit 0
fi
if [[ -e "${run_dir}" ]]; then
  echo "Incomplete endpoint run exists; refusing to overwrite: ${run_dir}" >&2
  exit 1
fi

"${python_bin}" train_endpoint_hclt.py \
  --num-latents 16 \
  --batch-size 1024 \
  --mini-batch-epochs 100 \
  --full-batch-epochs 20 \
  --validation-size 0 \
  --evaluate-every 5 \
  --em-step-start 0.15 \
  --em-step-end 0.05 \
  --em-pseudocount 0.1 \
  --topology-source iclr2022 \
  --topology-cache data/MNIST/hclt/topology_iclr2022_msb3_n60000.pt \
  --seed 0 \
  --wandb-project flow2pc \
  --wandb-group endpoint-hclt-iclr2022-table2 \
  --wandb-run-name "${label}" \
  --wandb-tags hclt,iclr-2022,table-2,m16,categorical-endpoint,hybrid-em,no-endpoint-trick \
  --output-dir "${run_dir}"
