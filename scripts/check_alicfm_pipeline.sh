#!/usr/bin/env bash
# User-run teacher -> reference figures -> student -> comparison figures.
# Full configuration budgets are retained. This script is not a sweep.
set -euo pipefail

# Parse the full workflow before launching long-running children.
# This protects an active run from on-disk edits shifting Bash input offsets.
main() {
dataset="${1:-cite50}"
if [[ $# -gt 1 ]]; then
  echo "Usage: bash scripts/check_alicfm_pipeline.sh [DATASET]" >&2
  exit 2
fi
case "$dataset" in
  cite5|cite50|cite100|multi5|multi50|multi100|eb|st|cell_tracking|knot) ;;
  *) echo "Unknown dataset: $dataset" >&2; exit 2 ;;
esac
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${FLOW2PC_DATA_ROOT:-/data/lthung/ali-cfm-data}"
run_root="${FLOW2PC_CHECK_ROOT:-/data/lthung/flow2pc/framework/migration-check/$dataset}"
case "$data_root" in /*) ;; *) data_root="$PWD/$data_root" ;; esac
case "$run_root" in /*) ;; *) run_root="$PWD/$run_root" ;; esac
# Share the same lock as the multi-GPU queue, so two launchers cannot write
# teacher/student checkpoints or W&B IDs for this dataset simultaneously.
mkdir -p "$run_root"
exec 9>"$run_root/.pipeline.lock"
if ! flock -n 9; then
  echo "Another launcher is already using $run_root" >&2
  exit 75
fi
teacher_dir="$run_root/teacher"
student_dir="$run_root/student"

teacher_args=(
  --set "data.root=$data_root"
  --set "output.dir=$teacher_dir"
  --set "training.device=${TEACHER_DEVICE:-cuda:0}"
  --set logging.project=flow2pc
  --set logging.mode=online
  --set evaluation.enabled=true
)
student_args=(
  --set "teacher.checkpoint=$teacher_dir/checkpoints/last.pt"
  --set "output.dir=$student_dir"
  --set "distillation.device=${STUDENT_DEVICE:-cuda:0}"
  --set logging.project=flow2pc
  --set logging.mode=online
  --set evaluation.enabled=true
  --set distillation.resume=true
)

teacher_evaluation_complete() {
  "${FLOW2PC_PYTHON:-python}" - "$teacher_dir" <<'PY_CACHE'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
def read(name):
    path = root / name
    return json.loads(path.read_text()) if path.is_file() else {}
try:
    media = read("eval/media.json")
    ready = (root / "checkpoints/last.pt").is_file() and (root / "eval/metrics.json").is_file()
    ready = ready and read("status.json").get("status") == "complete"
    ready = ready and read("eval/status.json").get("status") == "complete"
    ready = ready and bool(media) and all(Path(path).is_file() for path in media.values())
except (OSError, ValueError):
    ready = False
raise SystemExit(0 if ready else 1)
PY_CACHE
}

if teacher_evaluation_complete; then
  echo "Reusing the completed teacher and evaluation: $teacher_dir"
elif [[ -f "$teacher_dir/checkpoints/last.pt" ]]; then
  echo "Evaluating the existing teacher: $teacher_dir"
  bash "$repo_root/scripts/run_alicfm.sh" evaluate-teacher "$dataset" "${teacher_args[@]}"
else
  echo "Training and evaluating the teacher: $teacher_dir"
  bash "$repo_root/scripts/run_alicfm.sh" pretrain "$dataset" "${teacher_args[@]}"
fi

echo "Distilling (or resuming) and evaluating the selected student: $student_dir"
bash "$repo_root/scripts/run_alicfm.sh" distill "$dataset" "${student_args[@]}"
echo "Teacher figures: $teacher_dir/eval/media.json"
echo "Student/comparison figures: $student_dir/eval/media.json"
echo "Metrics: $teacher_dir/eval/metrics.json and $student_dir/eval/metrics.json"
echo "W&B project: flow2pc"
}

{
  main "$@"
  exit
}
