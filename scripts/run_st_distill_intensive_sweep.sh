#!/usr/bin/env bash
# Intensive sweep for both selected ST ALI-CFM folds.
# Each configuration is trained and evaluated before the next one starts.
set -euo pipefail

python_bin="${FLOW2PC_PYTHON:-/mnt/disk1/aiotlab/envs/uffm/bin/python}"
export CUDA_VISIBLE_DEVICES="${FLOW2PC_GPU:-0}"
export TMPDIR="${TMPDIR:-/mnt/disk2/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/mnt/disk2/tmp/flow2pc-matplotlib}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

data_dir="${ST_DATA_DIR:-outputs/ST/distillation_data}"
he_dir="${ST_HE_DIR:-outputs/ST/he_visualization_package/images/he_stains}"
steps="${ST_STEPS:-30000}"
ode_steps="${ST_ODE_STEPS:-100}"
ode_batch_size="${ST_ODE_BATCH_SIZE:-64}"
velocity_batch_size="${ST_VELOCITY_BATCH_SIZE:-64}"
batch_size="${ST_BATCH_SIZE:-32}"
project="${WANDB_PROJECT:-flow2pc}"
group="${WANDB_GROUP:-st-distill-intensive-sweep}"

# Full grid. Set ST_LIMIT_RUNS to stop after a fixed number of completed runs.
NUM_LATENTS_LIST=(64 128 256 512 1024)
LR_LIST=(5e-3 1e-3 5e-4 1e-4)
TIME_EMB_DIM_LIST=(32 64 128 256)
MARGINAL_NLL_WEIGHT_LIST=(0 1e-3 1e-2 1e-1)

folds=(
  "t=1_seed=45|U3|outputs/ST/t=1_seed=45/teacher.pth|outputs/ST/t=1_seed=45/generator.pth|outputs/ST/t=1_seed=45"
  "t=2_seed=42|U4|outputs/ST/t=2_seed=42/teacher.pth|outputs/ST/t=2_seed=42/generator.pth|outputs/ST/t=2_seed=42"
)

completed=0
limit="${ST_LIMIT_RUNS:-0}"

for fold in "${folds[@]}"; do
  IFS='|' read -r label held_out teacher generator reference_dir <<< "${fold}"
  for num_latents in "${NUM_LATENTS_LIST[@]}"; do
    for lr in "${LR_LIST[@]}"; do
      for time_emb_dim in "${TIME_EMB_DIM_LIST[@]}"; do
        for marginal_nll_weight in "${MARGINAL_NLL_WEIGHT_LIST[@]}"; do
          marginal_label="${marginal_nll_weight//./p}"
          run_name="st-${label}-h${num_latents}-lr${lr}-td${time_emb_dim}-mnll${marginal_label}"
          run_dir="outputs/runs/ST/${run_name}"

          if [[ -f "${run_dir}/checkpoints/last.pt" && -f "${run_dir}/eval/st_he_overlay.png" ]]; then
            echo "Skipping completed run: ${run_name}"
            continue
          fi
          if [[ "${limit}" -gt 0 && "${completed}" -ge "${limit}" ]]; then
            echo "Reached ST_LIMIT_RUNS=${limit}"
            echo "Completed ${completed} runs."
            exit 0
          fi

          echo "============================================================"
          echo "Run ${run_name}"
          echo "fold=${label} held_out=${held_out} h=${num_latents} lr=${lr} td=${time_emb_dim} mnll=${marginal_nll_weight}"
          echo "============================================================"

          "${python_bin}" train.py distill \
            --dataset st \
            --data-dir "${data_dir}" \
            --st-held-out-slide "${held_out}" \
            --teacher-arch alicfm \
            --teacher-checkpoint "${teacher}" \
            --generator-checkpoint "${generator}" \
            --pc-structure hclt \
            --pc-layer dense \
            --num-latents "${num_latents}" \
            --num-marginals 4 \
            --time-order 4 \
            --time-conditioning fourier \
            --time-embedding-dim "${time_emb_dim}" \
            --no-fourier-zero-center \
            --marginal-nll-weight "${marginal_nll_weight}" \
            --steps "${steps}" \
            --batch-size "${batch_size}" \
            --num-workers 0 \
            --lr "${lr}" \
            --optimizer adamw \
            --scheduler cosine \
            --weight-decay 0.01 \
            --gradient-clip 1.0 \
            --log-every 100 \
            --checkpoint-every "${steps}" \
            --output-dir "${run_dir}" \
            --wandb-project "${project}" \
            --wandb-group "${group}" \
            --wandb-run-name "${run_name}" \
            --wandb-tags "st,distill,sweep,${label},h${num_latents},mnll${marginal_label}" \
            --wandb-mode "${WANDB_MODE:-online}"

          "${python_bin}" scripts/evaluate_st_student.py \
            --checkpoint "${run_dir}/checkpoints/last.pt" \
            --data-dir "${data_dir}" \
            --held-out-slide "${held_out}" \
            --reference-dir "${reference_dir}" \
            --he-image-dir "${he_dir}" \
            --ode-steps "${ode_steps}" \
            --ode-batch-size "${ode_batch_size}" \
            --velocity-batch-size "${velocity_batch_size}" \
            --output "${run_dir}/eval/st_he_overlay.png" \
            --wandb-project "${project}" \
            --wandb-run-name "${run_name}" \
            --wandb-run-id "$(<"${run_dir}/wandb_run_id.txt")" \
            --wandb-mode "${WANDB_MODE:-online}"

          completed=$((completed + 1))
          echo "Completed ${completed}: ${run_name}"
        done
      done
    done
  done
done

echo "Sweep complete: ${completed} runs completed."
