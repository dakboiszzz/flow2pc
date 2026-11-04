#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/run_mmfm.sh pretrain|distill|evaluate PROFILE [hclt|gaussian|teacher] [--dry-run] [CLI options]' >&2
  exit 2
fi
stage="$1"
profile="$2"
shift 2
case "$profile" in *[!a-zA-Z0-9_]*|'') echo 'Invalid profile' >&2; exit 2;; esac
case "$stage" in
  pretrain) config="$repo_root/configs/pretrain/mmfm/$profile.yaml";;
  distill|evaluate)
    kind=hclt
    if [[ ${1:-} != --* && $# -gt 0 ]]; then kind="$1"; shift; fi
    case "$kind" in hclt|gaussian|teacher) ;; *) echo 'Invalid role/student' >&2; exit 2;; esac
    config="$repo_root/configs/$stage/mmfm/$profile/$kind.yaml";;
  *) echo 'Invalid stage' >&2; exit 2;;
esac
[[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
args=()
for arg in "$@"; do
  if [[ "$arg" == --dry-run ]]; then args+=(--validate-only); else args+=("$arg"); fi
done
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
exec "${FLOW2PC_PYTHON:-python}" main.py --config "$config" "${args[@]}"
