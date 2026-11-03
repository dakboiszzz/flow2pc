#!/usr/bin/env bash
# Prepared entry point; training starts only when the user invokes this script.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_alicfm.sh pretrain|distill|evaluate-teacher|evaluate-student DATASET [CLI options]" >&2
  echo "Datasets: cite5 cite50 cite100 multi5 multi50 multi100 eb st cell_tracking knot" >&2
  exit 2
fi

stage="$1"
dataset="$2"
shift 2
mode_args=()
case "$stage" in
  pretrain|distill) ;;
  evaluate-teacher) stage=pretrain; mode_args=(--evaluate teacher) ;;
  evaluate-student) stage=distill; mode_args=(--evaluate student) ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
case "$dataset" in
  cite5|cite50|cite100|multi5|multi50|multi100|eb|st|cell_tracking|knot) ;;
  *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${FLOW2PC_PYTHON:-python}" "$repo_root/main.py" \
  --config "$repo_root/configs/$stage/alicfm_$dataset.yaml" "${mode_args[@]}" "$@"
