#!/usr/bin/env bash
# Prepare arch snapshots and train MMFM with the repository's full arch config.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${FLOW2PC_PYTHON:-python}"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$repo_root"
"$python_bin" -m flow2pc.data.prepare_mfm arch

FLOW2PC_PYTHON="$python_bin" bash scripts/run_mmfm.sh pretrain arch "$@"
