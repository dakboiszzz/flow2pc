#!/bin/bash
set -e

# Standard knot grid search: architecture × LR × density_weight
# No parameter-scaling experiments (see implementation_plan.md for that).

CHECKPOINT="checkpoints/knot_alicfm_teacher.pt"
WANDB_PROJECT="flow2pc"
WANDB_GROUP="large_model_sweep"
TAG="large_model_sweep"
STEPS=5000
BATCH=128

echo "======================================================="
echo "  Knot Grid Search (arch × LR × density_weight)"
echo "  $(date)"
echo "======================================================="

run_and_plot() {
    local RUN_NAME="$1"
    shift
    echo ""
    echo "-------------------------------------------------------"
    echo "  Run: $RUN_NAME"
    echo "-------------------------------------------------------"

    python train.py distill \
        --dataset knot \
        --teacher-checkpoint $CHECKPOINT \
        --teacher-arch alicfm \
        --pc-structure hclt \
        --pc-layer dense \
        --steps $STEPS \
        --checkpoint-every $STEPS \
        --batch-size $BATCH \
        --wandb-project $WANDB_PROJECT \
        --wandb-group $WANDB_GROUP \
        --wandb-run-name "$RUN_NAME" \
        --wandb-tags "$TAG" \
        "$@"

    LATEST_RUN=$(ls -td outputs/runs/distill/*/ | head -1)
    CKPT="${LATEST_RUN}checkpoints/last.pt"
    IMG_OUT="${LATEST_RUN}student_trajectories.png"
    echo "  Generating ODE path image -> $IMG_OUT"
    python scripts/plot_ode_path.py \
        --model-type student \
        --checkpoint "$CKPT" \
        --output "$IMG_OUT" \
        --num-trajectories 200 \
        --num-steps 100 || echo "  [WARN] ODE path failed for $RUN_NAME"
    echo "  Done: $RUN_NAME"
}

# Configs: num_latents × time_embedding_dim
# H=128 D=32  ->  32,896 params
# H=128 D=64  ->  49,280 params
# H=256 D=32  ->  98,560 params
# H=256 D=64  -> 131,328 params

for LATENTS in 128 256; do
    for DIM in 32 64; do
        for LR in 1e-3 5e-3; do
            for DW in 0.0 0.01 0.1 1.0; do

                RUN_NAME="knot-h${LATENTS}-d${DIM}-lr${LR}-dw${DW}"

                run_and_plot "$RUN_NAME" \
                    --time-conditioning fourier \
                    --time-embedding-dim $DIM \
                    --num-latents $LATENTS \
                    --lr $LR \
                    --density-weight $DW

            done
        done
    done
done

echo ""
echo "======================================================="
echo "  All 64 runs complete. Generating summary CSV..."
echo "======================================================="

python scripts/summarize_knot_tuning.py \
    --output-dir outputs/benchmark/knot-large-model-sweep \
    --tag large_model_sweep \
    --fast

echo ""
echo "Summary written to: outputs/benchmark/knot-large-model-sweep/metrics.csv"
echo "Done. $(date)"
