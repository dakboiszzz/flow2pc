#!/bin/bash
set -e

# 1. Retrain the knot teacher with the updated ground truth distribution
echo "=========================================================="
echo "Retraining the Knot Teacher (this will take a minute or two)"
echo "=========================================================="

# We'll use 20000 steps by default as specified in the script
python scripts/train_knot_teacher.py \
    --output checkpoints/knot_alicfm_teacher.pt \
    --steps 20000 \
    --batch-size 256 \
    --lr 0.001

# 2. Evaluate the teacher (generate trajectories and NLL/MSE)
echo "=========================================================="
echo "Evaluating the retrained Knot Teacher"
echo "=========================================================="

python scripts/plot_ode_path.py \
    --dataset knot \
    --model-type teacher \
    --checkpoint checkpoints/knot_alicfm_teacher.pt \
    --output outputs/knot_teacher_trajectories.png

echo "=========================================================="
echo "Done! Check outputs/knot_teacher_trajectories.png for the visual results."
echo "=========================================================="
