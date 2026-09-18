#!/bin/bash
set -e

# Base configurations
DATASET="spiral"
TEACHER_CKPT="checkpoints/spiral_alicfm_teacher.pt"
STEPS=5000

# Ensure the log directory exists
mkdir -p logs

run_experiment() {
    local latents=$1
    local time_dim=$2
    local lr=$3
    local density_weight=$4

    local run_name="${DATASET}-scale-h${latents}-d${time_dim}-lr${lr}-dw${density_weight}"
    echo "=========================================================="
    echo "Starting run: $run_name"
    echo "=========================================================="

    WANDB__SERVICE_WAIT=300 python train.py distill \
        --dataset $DATASET \
        --teacher-checkpoint $TEACHER_CKPT \
        --pc-structure hclt \
        --pc-layer dense \
        --capacity-profile standard \
        --flow-matcher vanilla \
        --teacher-arch alicfm \
        --num-marginals 60 \
        --multimarginal-time-sampling uniform_interval \
        --steps $STEPS \
        --batch-size 128 \
        --num-workers 4 \
        --data-dir data \
        --lr $lr \
        --gradient-clip 1.0 \
        --log-every 100 \
        --checkpoint-every $STEPS \
        --num-latents $latents \
        --time-order 4 \
        --time-conditioning fourier \
        --time-embedding-dim $time_dim \
        --hclt-leaf-distribution gaussian \
        --hclt-leaf-components 1 \
        --endpoint-pretraining none \
        --density-weight $density_weight \
        --timestep-sampling uniform \
        --hclt-topology-protocol pyjuice-kde \
        --hclt-mi-sample-chunk-size 1024 \
        --hclt-samples 20000 \
        --hclt-num-bins 64 \
        --hclt-sigma 0.02 \
        --hclt-chunk-size 32 \
        --wandb-project flow2pc \
        --wandb-run-name $run_name \
        --wandb-group param_scaling_sweep \
        --wandb-tags param_scaling_sweep \
        --wandb-mode online
}

# The sweeping grid matching the knot experiment
lrs=("0.0005" "0.001" "0.005")
density_weights=("0.0" "0.1")

# The architectural capacity profiles
# Format: "num_latents time_embedding_dim"
capacities=(
    "512 96"
    "512 128"
    "512 32"
    "512 64"
)

for lr in "${lrs[@]}"; do
    for dw in "${density_weights[@]}"; do
        for cap in "${capacities[@]}"; do
            read -r latents time_dim <<< "$cap"
            run_experiment $latents $time_dim $lr $dw
        done
    done
done

echo "All sweep runs completed successfully!"
