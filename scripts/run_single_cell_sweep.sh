#!/usr/bin/env bash
set -euo pipefail
TEACHER_CKPT="$1"
if [[ -z "$TEACHER_CKPT" ]]; then
    echo "Usage: bash scripts/run_single_cell_sweep.sh <path_to_teacher_checkpoint>"
    exit 1
fi
shift

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

export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-120}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"

REFERENCE="/data/lthung/flow2pc/outputs/reference/cite_${DIM}/teacher_seed${SEED}/metrics.json"
OUTPUT_DIR="/data/lthung/flow2pc/outputs/single_cell/cite${DIM,,}-intensive-seed${SEED}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${FLOW2PC_PYTHON:-python}"

exec "$python_bin" -u scripts/run_single_cell_sweep.py \
    --grid "configs/single_cell/cite5d_intensive_sweep.json" \
    --data-dir "/data/lthung/single_cell" \
    --output-dir "$OUTPUT_DIR" \
    --teacher-checkpoint "$TEACHER_CKPT" \
    --reference "$REFERENCE" \
    --wandb-mode online \
    --wandb-project flow2pc \
    "$@"
