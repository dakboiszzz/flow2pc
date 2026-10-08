#!/bin/bash
set -e

TEACHER_CKPT="$1"
if [[ -z "$TEACHER_CKPT" ]]; then
    echo "Usage: bash scripts/evaluate_teacher.sh <path_to_teacher_checkpoint>"
    exit 1
fi

if [[ "$TEACHER_CKPT" =~ cite_([0-9]+D) ]]; then
    DIM="${BASH_REMATCH[1]}"
else
    echo "Error: Could not infer dimension (e.g. cite_50D) from checkpoint path"
    exit 1
fi

if [[ "$TEACHER_CKPT" =~ seed=([0-9]+) ]]; then
    SEED="${BASH_REMATCH[1]}"
else
    SEED="42"
fi

REFERENCE_DIR="/data/lthung/flow2pc/outputs/reference/cite_${DIM}/teacher_seed${SEED}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${FLOW2PC_PYTHON:-python}"

echo "Evaluating Teacher for ${DIM}..."
echo "Checkpoint: $TEACHER_CKPT"
echo "Saving Reference to: $REFERENCE_DIR"

exec "$python_bin" -u scripts/evaluate_single_cell.py \
    --data-dir "/data/lthung/single_cell" \
    --teacher-checkpoint "$TEACHER_CKPT" \
    --output-dir "$REFERENCE_DIR" \
    --wandb-project flow2pc \
    --wandb-mode online
