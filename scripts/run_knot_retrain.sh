#!/bin/bash
set -e

DATASET="knot"
TEACHER_CKPT="checkpoints/knot_alicfm_teacher.pt"
STEPS=10000 # Using 10000 steps based on the previous sweep, adjust if necessary

mkdir -p logs

run_experiment() {
    local latents=$1
    local time_dim=$2
    local lr=$3
    local density_weight=$4

    # Format the run name exactly as requested
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
        --wandb-group knot_retrain_sweep \
        --wandb-tags knot_retrain \
        --wandb-mode online
}

# 1. knot-scale-h256-d128-lr5e-3-dw0.0
run_experiment 256 128 0.005 0.0

# 2. knot-scale-h384-d32-lr5e-3-dw0.0
run_experiment 384 32 0.005 0.0

# 3. knot-scale-h384-d64-lr5e-3-dw0.0
run_experiment 384 64 0.005 0.0

echo "All retrain runs completed successfully!"
