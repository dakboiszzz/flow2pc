#!/usr/bin/env bash
# ============================================================
# Knot Distillation Hyperparameter Sweep
# Trains student models on cached ALI-CFM teacher trajectories
# and logs all losses + trajectory image to W&B.
# ============================================================
set -euo pipefail

# ---- Environment config ----
python_bin="${FLOW2PC_PYTHON:-/mnt/disk1/aiotlab/envs/uffm/bin/python}"
gpu="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="${gpu}"

# Redirect tmp to disk2 to avoid root partition filling up
export TMPDIR="/mnt/disk2/tmp"
mkdir -p "${TMPDIR}"

TEACHER_CKPT="checkpoints/teacher.pt"
DATASET="knot"
WANDB_PROJECT="flow2pc"
WANDB_GROUP="knot-distill-sweep-gen"

# ---- Sanity checks ----
for f in "${TEACHER_CKPT}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: Required checkpoint not found: ${f}" >&2
    exit 1
  fi
done

# ============================================================
# Hyperparameter grid
# num_latents  - model capacity
# lr           - learning rate
# time_emb_dim - richness of time conditioning
# ============================================================
NUM_LATENTS_LIST=(64 128 256 384 512)
LR_LIST=(5e-3 1e-3 5e-4 1e-4)
TIME_EMB_DIM_LIST=(32 64 128 256)
STEPS=30000
BATCH_SIZE=128
NUM_MARGINALS=60
TIME_ORDER=4

# Plotting is handled internally by train.py at the end of training.

# ============================================================
# Main sweep loop
# ============================================================
sweep_count=0

for num_latents in "${NUM_LATENTS_LIST[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    for time_emb_dim in "${TIME_EMB_DIM_LIST[@]}"; do

      run_name="knot-distill-gen-h${num_latents}-lr${lr}-td${time_emb_dim}"
      run_dir="outputs/runs/distill/${run_name}"

      echo ""
      echo "========================================================"
      echo "  Run: ${run_name}"
      echo "  num_latents=${num_latents}  lr=${lr}  time_emb_dim=${time_emb_dim}"
      echo "========================================================"

      if [[ -d "${run_dir}" ]]; then
        echo "  Skipping: directory already exists (${run_dir})"
        continue
      fi

      # ---- Train student ----
      "${python_bin}" train.py distill \
        --dataset "${DATASET}" \
        --teacher-checkpoint "${TEACHER_CKPT}" \
        --generator-checkpoint checkpoints/generator.pt \
        --pc-structure hclt \
        --pc-layer dense \
        --num-latents "${num_latents}" \
        --num-marginals "${NUM_MARGINALS}" \
        --time-order "${TIME_ORDER}" \
        --time-conditioning fourier \
        --time-embedding-dim "${time_emb_dim}" \
        --no-fourier-zero-center \
        --no-polynomial-zero-center \
        --hclt-leaf-distribution gaussian \
        --hclt-leaf-components 1 \
        --steps "${STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers 0 \
        --lr "${lr}" \
        --optimizer adamw \
        --scheduler cosine \
        --weight-decay 0.01 \
        --gradient-clip 1.0 \
        --log-every 100 \
        --checkpoint-every "${STEPS}" \
        --output-dir "${run_dir}" \
        --wandb-project "${WANDB_PROJECT}" \
        --wandb-group "${WANDB_GROUP}" \
        --wandb-run-name "${run_name}" \
        --wandb-tags "knot,distill,generator,sweep" \
        --wandb-mode online



      sweep_count=$(( sweep_count + 1 ))
      echo "  Completed run ${sweep_count}: ${run_name}"

    done
  done
done

echo ""
echo "========================================================"
echo "  Sweep complete! Total runs finished: ${sweep_count}"
echo "  All results logged to wandb group: ${WANDB_GROUP}"
echo "========================================================"
