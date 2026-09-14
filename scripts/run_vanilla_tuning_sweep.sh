#!/bin/bash
set -e

STEPS=5000
WANDB_PROJECT="flow2pc"
CHECKPOINT="checkpoints/knot_alicfm_teacher.pt"

echo "Starting Vanilla Flow2PC Knot Tuning Sweep"

for LR in 1e-4 5e-4 1e-3 5e-3; do
    for LATENTS in 64 128 256; do
        for COND in polynomial fourier; do
            if [ "$COND" = "fourier" ]; then
                DIM_VALS="32 64"
            else
                DIM_VALS="0"
            fi

            for DIM in $DIM_VALS; do
                RUN_NAME="knot-vanilla-lr${LR}-h${LATENTS}-${COND}"
                if [ "$COND" = "fourier" ]; then
                    RUN_NAME="${RUN_NAME}-d${DIM}"
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
                    --steps $STEPS \
                    --checkpoint-every 5000 \
                    --validate-every 1000 \
                    --wandb-project $WANDB_PROJECT \
                    --wandb-tags vanilla_sweep,knot \
                    --wandb-run-name $RUN_NAME"
                
                if [ "$COND" = "fourier" ]; then
                    CMD="$CMD --time-embedding-dim $DIM"
                fi

                $CMD
            done
        done
    done
done

echo "Tuning sweep completed successfully! Running summarization script..."
python scripts/summarize_knot_tuning.py

echo "Done."
