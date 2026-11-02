#!/usr/bin/env bash
# Native controls only: six single-cell datasets and historical ST, seed 42/t=1.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${FLOW2PC_PYTHON:-/data/nvidia/conda_envs/uffm/bin/python}"
native_root="${ALI_CFM_ROOT:-$repo_root/../ali-cfm}"
data_root="${FLOW2PC_DATA_ROOT:-/data/lthung/ali-cfm-data}"
output_root="${ALI_CFM_CONTROL_ROOT:-/data/lthung/flow2pc/alicfm-native-controls}"
dry_run=false
if [[ ${1:-} == --dry-run ]]; then
  dry_run=true
  shift
fi
if (( $# )); then
  echo "Usage: bash $0 [--dry-run]" >&2
  exit 2
fi
IFS=, read -r -a gpu_ids <<< "${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}}"
datasets=(cite5 cite50 cite100 multi5 multi50 multi100 st)
if (( ${#gpu_ids[@]} < ${#datasets[@]} )); then
  echo "Seven independent native controls require at least seven selected GPUs; set GPUS." >&2
  exit 2
fi
declare -A used_gpus=()
for gpu in "${gpu_ids[@]}"; do
  if [[ -z $gpu || -n ${used_gpus[$gpu]:-} ]]; then
    echo "Empty or duplicate GPU in GPUS." >&2
    exit 2
  fi
  used_gpus[$gpu]=1
done

export OMP_NUM_THREADS="${CPU_THREADS_PER_RUN:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export MPLBACKEND=Agg
runner=("$python_bin" "$repo_root/scripts/run_native_alicfm_control.py"
  --native-root "$native_root" --data-root "$data_root" --output-root "$output_root")

# Check every source/data path before starting any training or making log dirs.
for index in "${!datasets[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu_ids[$index]}" "${runner[@]}" --dataset "${datasets[$index]}" --dry-run
done
if [[ $dry_run == true ]]; then
  exit 0
fi

mkdir -p "$output_root/_launcher/logs"
pids=()
stop_workers() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  exit 130
}
trap stop_workers INT TERM
for index in "${!datasets[@]}"; do
  tag="${datasets[$index]}"
  echo "Launching native $tag on GPU ${gpu_ids[$index]}"
  CUDA_VISIBLE_DEVICES="${gpu_ids[$index]}" "${runner[@]}" --dataset "$tag" \
    > "$output_root/_launcher/logs/$tag.log" 2>&1 &
  pids+=("$!")
done
failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Finished: ${datasets[$index]}"
  else
    echo "Failed: ${datasets[$index]}; see $output_root/_launcher/logs/${datasets[$index]}.log" >&2
    failed=1
  fi
done
exit "$failed"
