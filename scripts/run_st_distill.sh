#!/usr/bin/env bash
# Distill the two selected ALI-CFM spatial-transcriptomics folds into Flow2PC.
# The fold-specific teacher/generator bundles are the selected U3 and U4 runs
# under outputs/ST.  Override ST_STEPS, ST_NUM_LATENTS, or FLOW2PC_GPU as needed.
set -euo pipefail

python_bin="${FLOW2PC_PYTHON:-/mnt/disk1/aiotlab/envs/uffm/bin/python}"
export CUDA_VISIBLE_DEVICES="${FLOW2PC_GPU:-0}"
export TMPDIR="${TMPDIR:-/mnt/disk2/tmp}"
mkdir -p "${TMPDIR}"

data_dir="${ST_DATA_DIR:-outputs/ST/distillation_data}"
steps="${ST_STEPS:-30000}"
batch_size="${ST_BATCH_SIZE:-64}"
num_latents="${ST_NUM_LATENTS:-512}"
time_embedding_dim="${ST_TIME_EMB_DIM:-128}"
lr="${ST_LR:-5e-4}"
marginal_nll_weight="${ST_MARGINAL_NLL_WEIGHT:-0.1}"
ode_steps="${ST_ODE_STEPS:-100}"
ode_batch_size="${ST_ODE_BATCH_SIZE:-64}"
velocity_batch_size="${ST_VELOCITY_BATCH_SIZE:-64}"
marginal_nll_label="${marginal_nll_weight//./p}"
project="${WANDB_PROJECT:-flow2pc}"
group="${WANDB_GROUP:-st-distill-selected-alicfm}"

declare -a folds=(
  "t=1_seed=45|U3|outputs/ST/t=1_seed=45/teacher.pth|outputs/ST/t=1_seed=45/generator.pth"
  "t=2_seed=42|U4|outputs/ST/t=2_seed=42/teacher.pth|outputs/ST/t=2_seed=42/generator.pth"
)

for fold in "${folds[@]}"; do
  IFS='|' read -r label held_out teacher generator <<< "${fold}"
  run_name="st-${label}-h${num_latents}-lr${lr}-td${time_embedding_dim}-mnll${marginal_nll_label}"
  run_dir="outputs/runs/ST/${run_name}"

  for required_file in "${teacher}" "${generator}" \
    "${data_dir}/U2_normalized_float32.csv" \
    "${data_dir}/U3_normalized_float32.csv" \
    "${data_dir}/U4_normalized_float32.csv" \
    "${data_dir}/U5_normalized_float32.csv"; do
    if [[ ! -f "${required_file}" ]]; then
      echo "ERROR: required ST input is missing: ${required_file}" >&2
      exit 1
    fi
  done

  run_id=""
  if [[ -f "${run_dir}/wandb_run_id.txt" ]]; then
    run_id="$(<"${run_dir}/wandb_run_id.txt")"
  else
    # Recover runs created before train.py started persisting the W&B ID.
    latest_wandb_dir="$(find "${run_dir}/wandb" -mindepth 1 -maxdepth 1 -type d -name 'run-*' -print 2>/dev/null | sort | tail -1 || true)"
    if [[ -n "${latest_wandb_dir}" ]]; then
      run_id="${latest_wandb_dir##*-}"
    fi
  fi

  evaluate_run() {
    local eval_args=(
      --checkpoint "${run_dir}/checkpoints/last.pt"
      --data-dir "${data_dir}"
      --held-out-slide "${held_out}"
      --reference-dir "outputs/ST/${label}"
      --output "${run_dir}/eval/st_he_overlay.png"
      --he-image-dir "outputs/ST/he_visualization_package/images/he_stains"
      --ode-steps "${ode_steps}"
      --ode-batch-size "${ode_batch_size}"
      --velocity-batch-size "${velocity_batch_size}"
      --seed 0
      --wandb-project "${project}"
      --wandb-run-name "${run_name}"
      --wandb-mode "${WANDB_MODE:-online}"
    )
    if [[ -n "${run_id}" ]]; then
      eval_args+=(--wandb-run-id "${run_id}")
    fi
    "${python_bin}" scripts/evaluate_st_student.py "${eval_args[@]}"
  }

  if [[ -f "${run_dir}/checkpoints/last.pt" ]]; then
    wandb_he_media="$(find "${run_dir}/wandb" -type f -path '*media/images/eval/st_he_overlay*' -print -quit 2>/dev/null || true)"
    if [[ -f "${run_dir}/eval/st_he_overlay.png" && -n "${wandb_he_media}" ]]; then
      echo "Skipping completed and evaluated run: ${run_dir}"
    else
      echo "Checkpoint exists without evaluation; evaluating: ${run_dir}"
      evaluate_run
    fi
    continue
  fi

  echo "Running ${run_name} (held out ${held_out})"
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
    --time-embedding-dim "${time_embedding_dim}" \
    --no-fourier-zero-center \
    --steps "${steps}" \
    --batch-size "${batch_size}" \
    --num-workers 0 \
    --lr "${lr}" \
    --optimizer adamw \
    --scheduler cosine \
    --weight-decay 0.01 \
    --gradient-clip 1.0 \
    --marginal-nll-weight "${marginal_nll_weight}" \
    --log-every 100 \
    --checkpoint-every "${steps}" \
    --output-dir "${run_dir}" \
    --wandb-project "${project}" \
    --wandb-group "${group}" \
    --wandb-run-name "${run_name}" \
    --wandb-tags "st,distill,alicfm,generator,${label}" \
    --wandb-mode "${WANDB_MODE:-online}"

  run_id="$(<"${run_dir}/wandb_run_id.txt")"
  evaluate_run
done

echo "ST distillation complete. Runs are under outputs/runs/ST/."
