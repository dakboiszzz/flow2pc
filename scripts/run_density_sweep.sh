#!/bin/bash
set -e

# Intensive hyperparameter tuning sweep for density_weight and architecture on knot dataset
# Usage: ./scripts/run_density_sweep.sh

CHECKPOINT="checkpoints/knot_alicfm_teacher.pt"
WANDB_PROJECT="flow2pc"

echo "Starting Intensive Density Sweep"

for LR in 5e-3; do
    for LATENTS in 64 128 256; do
        for COND in fourier; do
            if [ "$COND" = "fourier" ]; then
                DIM_VALS="32 64"
            else
                DIM_VALS="0"
            fi
            
            for DIM in $DIM_VALS; do
                for DENSITY_WEIGHT in 0.01 0.1 1.0; do
                    
                    if [ "$COND" = "fourier" ]; then
                        RUN_NAME="knot-lr${LR}-h${LATENTS}-f${DIM}-dw${DENSITY_WEIGHT}"
                    else
                        RUN_NAME="knot-lr${LR}-h${LATENTS}-poly-dw${DENSITY_WEIGHT}"
                    fi
                    
                    echo "=========================================================="
                    echo "Starting run: $RUN_NAME"
                    echo "=========================================================="
                    
                    CMD="python train.py distill \
                        --dataset knot \
                        --teacher-checkpoint $CHECKPOINT \
                        --teacher-arch alicfm \
                        --pc-structure hclt \
                        --pc-layer dense \
                        --time-conditioning $COND \
                        --num-latents $LATENTS \
                        --lr $LR \
                        --steps 5000 \
                        --checkpoint-every 5000 \
                        --validate-every 1000 \
                        --density-weight $DENSITY_WEIGHT \
                        --wandb-project $WANDB_PROJECT \
                        --wandb-tags density_sweep,knot \
                        --wandb-group intensive_density_sweep \
                        --wandb-run-name $RUN_NAME"
                    
                    if [ "$COND" = "fourier" ]; then
                        CMD="$CMD --time-embedding-dim $DIM"
                    fi

                    echo "Executing: $CMD"
                    eval $CMD
                    echo "Finished run: $RUN_NAME"
                    echo ""
                done
            done
        done
    done
done

echo "Density sweep completed successfully! Running summarization script..."
python scripts/summarize_knot_tuning.py

echo "Done."
