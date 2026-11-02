#!/usr/bin/env bash
# Config launcher based on the original MFM pretrain and metric distill scripts.
# Training starts only when invoked without --dry-run or --validate-only.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_mfm.sh pretrain|distill DATASET [VARIANT] [CLI options]

Pretrain datasets: arch sphere cite multi eb lidar afhq
                  cite5 cite50 cite100 multi5 multi50 multi100
  cite/multi default to 100 dimensions, eb to 5 (as in the source scripts).
  VARIANT: ot-mfm (default for arch/cite/multi/eb), i-mfm, or mfm
           sphere/lidar/afhq use mfm; only existing configs are accepted.

Distill datasets: arch sphere cite multi eb lidar afhq
  VARIANT: all (default), hclt, gaussian, hclt_t2, gaussian_t2,
           hclt_t3, gaussian_t3 (only where a config exists).
  all runs the dataset's available student configs sequentially.

Options:
  --dry-run         Print commands without loading data or starting jobs.
  --validate-only   Forward to main.py; validate without training.
  --set KEY=VALUE   Forward a config override to main.py.
  -h, --help        Show this help.

Environment: FLOW2PC_PYTHON selects Python (default: python).
             CUDA_VISIBLE_DEVICES can restrict GPUs; no GPU auto-selection.
Data, paired teacher checkpoints, output paths and logging come from configs.
Distill does not automatically select checkpoints from a preceding pretrain run.
EOF
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  usage
  exit 0
fi
if (( $# < 2 )); then
  usage >&2
  exit 2
fi
stage="$1"
dataset="$2"
shift 2
variant=""
if (( $# )) && [[ "$1" != -* ]]; then
  variant="$1"
  shift
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
configs=()
case "$stage" in
  pretrain)
    case "$dataset" in
      arch)
        variant="${variant:-ot-mfm}"
        case "$variant" in
          ot-mfm) config="configs/pretrain/mfm/arch.yaml" ;;
          i-mfm) config="configs/pretrain/mfm/runs/arch/i-mfm.yaml" ;;
          *) echo "Unsupported arch variant: $variant" >&2; exit 2 ;;
        esac
        ;;
      cite|multi|cite5|cite50|cite100|multi5|multi50|multi100|eb)
        variant="${variant:-ot-mfm}"
        case "$variant" in ot-mfm|i-mfm) ;; *) echo "Unknown variant: $variant" >&2; exit 2 ;; esac
        case "$dataset" in
          cite|multi) name="$dataset"; dims=100 ;;
          eb) name=eb; dims=5 ;;
          cite*) name=cite; dims="${dataset#cite}" ;;
          multi*) name=multi; dims="${dataset#multi}" ;;
        esac
        config="configs/pretrain/mfm/runs/single_cell/${dims}dims/${variant}_${name}.yaml"
        ;;
      sphere|lidar|afhq)
        variant="${variant:-mfm}"
        if [[ "$variant" != mfm ]]; then
          echo "$dataset only supports the mfm variant" >&2
          exit 2
        fi
        folder="$dataset"
        [[ "$dataset" != afhq ]] || folder=images
        config="configs/pretrain/mfm/runs/$folder/mfm.yaml"
        ;;
      *) echo "Unknown pretrain dataset: $dataset" >&2; exit 2 ;;
    esac
    configs+=("$repo_root/$config")
    ;;
  distill)
    case "$dataset" in
      arch|sphere|cite|multi|eb|lidar|afhq) ;;
      *) echo "Unknown distill dataset: $dataset; use cite/multi without a dimension suffix" >&2; exit 2 ;;
    esac
    variant="${variant:-all}"
    case "$variant" in
      all)
        for student in hclt gaussian; do
          for suffix in '' _t2 _t3; do
            config="$repo_root/configs/distill/mfm/$dataset/$student$suffix.yaml"
            if [[ -f "$config" ]]; then configs+=("$config"); fi
          done
        done
        ;;
      hclt|gaussian|hclt_t2|gaussian_t2|hclt_t3|gaussian_t3)
        configs+=("$repo_root/configs/distill/mfm/$dataset/$variant.yaml")
        ;;
      *) echo "Unknown distill variant: $variant" >&2; exit 2 ;;
    esac
    ;;
  *) echo "Unknown stage: $stage; expected pretrain or distill" >&2; exit 2 ;;
esac

dry_run=false
cli_options=()
while (( $# )); do
  case "$1" in
    --dry-run) dry_run=true; shift ;;
    --validate-only) cli_options+=("$1"); shift ;;
    --set)
      if (( $# < 2 )); then echo "--set needs KEY=VALUE" >&2; exit 2; fi
      cli_options+=("$1" "$2"); shift 2 ;;
    --set=*) cli_options+=("$1"); shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if (( ${#configs[@]} == 0 )); then
  echo "No configs found for $stage/$dataset" >&2
  exit 2
fi
for config in "${configs[@]}"; do
  if [[ ! -f "$config" ]]; then
    echo "Config not available: $config" >&2
    exit 2
  fi
done

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
for config in "${configs[@]}"; do
  command=("${FLOW2PC_PYTHON:-python}" "$repo_root/main.py" --config "$config" "${cli_options[@]}")
  printf 'Command:'
  printf ' %q' "${command[@]}"
  printf '\n'
  if [[ "$dry_run" == false ]]; then
    "${command[@]}"
  fi
done
