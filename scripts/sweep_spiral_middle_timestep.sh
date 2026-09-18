#!/usr/bin/env bash
set -euo pipefail

# Compare the existing uniform transition sampling with distributions that
# concentrate OT transitions around the middle of the spiral trajectory.
# This follows sweep_spiral_mega_scale.sh; override arrays below for a smaller
# smoke test before launching the full sweep.
DATASET="spiral"
TEACHER_CKPT="checkpoints/spiral_alicfm_teacher.pt"
STEPS="${STEPS:-40000}"

mkdir -p logs

run_experiment() {
    local latents="$1"
    local time_dim="$2"
    local num_bins="$3"
    local lr="$4"
    local density_weight="$5"
    local sampling="$6"
    local alpha="$7"
    local beta="$8"

    local label="uniform"
    if [[ "$sampling" == "middle_beta" ]]; then
        label="middle-beta-a${alpha}-b${beta}"
    fi
    local run_name="${DATASET}-mega-h${latents}-d${time_dim}-b${num_bins}-lr${lr}-dw${density_weight}-ts${label}"

    echo "=========================================================="
    echo "Starting run: ${run_name}"
    echo "=========================================================="

    WANDB__SERVICE_WAIT=300 python train.py distill \
        --dataset "$DATASET" \
        --teacher-checkpoint "$TEACHER_CKPT" \
        --pc-structure hclt \
        --pc-layer dense \
        --capacity-profile standard \
        --flow-matcher vanilla \
        --teacher-arch alicfm \
        --num-marginals 60 \
        --multimarginal-time-sampling "$sampling" \
        --multimarginal-beta-alpha "$alpha" \
        --multimarginal-beta-beta "$beta" \
        --steps "$STEPS" \
        --batch-size 128 \
        --num-workers 4 \
        --data-dir data \
        --lr "$lr" \
        --gradient-clip 1.0 \
        --log-every 100 \
        --checkpoint-every "$STEPS" \
        --num-latents "$latents" \
        --time-order 4 \
        --time-conditioning fourier \
        --time-embedding-dim "$time_dim" \
        --hclt-leaf-distribution gaussian \
        --hclt-leaf-components 1 \
        --endpoint-pretraining none \
        --density-weight "$density_weight" \
        --timestep-sampling uniform \
        --hclt-topology-protocol pyjuice-kde \
        --hclt-mi-sample-chunk-size 1024 \
        --hclt-samples 20000 \
        --hclt-num-bins "$num_bins" \
        --hclt-sigma 0.02 \
        --hclt-chunk-size 32 \
        --wandb-project flow2pc \
        --wandb-run-name "$run_name" \
        --wandb-group spiral_middle_timestep \
        --wandb-tags spiral_middle_timestep,"$sampling" \
        --wandb-mode online
}

lrs=("0.005" "0.001" "0.0005")
density_weights=("0.0" "0.1")
capacities=(
    "1024 128 128"
    "2048 256 128"
    "4096 256 256"
    "8192 512 256"
)

# Beta(a, b) with a=b is symmetric around t=0.5.  Larger concentrations
# spend progressively more updates in the middle of the spiral trajectory.
samplings=(
    "uniform_interval 1.0 1.0"
    "middle_beta 2.0 2.0"
    "middle_beta 5.0 5.0"
    "middle_beta 10.0 10.0"
)

for lr in "${lrs[@]}"; do
    for dw in "${density_weights[@]}"; do
        for cap in "${capacities[@]}"; do
            read -r latents time_dim num_bins <<< "$cap"
            for spec in "${samplings[@]}"; do
                read -r sampling alpha beta <<< "$spec"
                run_experiment "$latents" "$time_dim" "$num_bins" "$lr" "$dw" \
                    "$sampling" "$alpha" "$beta"
            done
        done
    done
done

echo "All spiral middle-timestep sweep runs completed successfully!"
