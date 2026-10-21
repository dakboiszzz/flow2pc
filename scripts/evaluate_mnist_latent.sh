#!/usr/bin/env bash
# Compare a latent student to its exact frozen teacher, with full MNIST metrics.
# Invoked explicitly by the user or the complete latent-distillation launcher.
set -euo pipefail
if [[ $# -lt 1 ]]; then
    echo "Usage: bash scripts/evaluate_mnist_latent.sh <student-or-teacher.pt> [evaluation options]" >&2
    exit 2
fi
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
checkpoint="$1"
shift
if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
fi
artifact_root="/data/lthung/flow2pc"
timestamp="$(date -u +%Y%m%dT%H%M%S%NZ)"
output="${LATENT_EVAL_DIR:-${artifact_root}/outputs/mnist-latent/eval-${timestamp}}"
data_dir="${LATENT_DATA_DIR:-${artifact_root}/data/mnist}"
bundle_dir="${LATENT_BUNDLE_DIR:-${artifact_root}/latents/mnist-eqvae-seed0}"
python_bin="${FLOW2PC_PYTHON:-python}"
for artifact_path in "${output}" "${data_dir}" "${bundle_dir}"; do
    case "$(realpath -m -- "${artifact_path}")" in
        "${artifact_root}"/*) ;;
        *) echo "Runtime paths must be under ${artifact_root}: ${artifact_path}" >&2; exit 1 ;;
    esac
done
if [[ -e "${output}" ]]; then
    echo "Use a fresh LATENT_EVAL_DIR: ${output} already exists." >&2
    exit 1
fi
runtime="${output}.runtime"
export CUDA_VISIBLE_DEVICES="${FLOW2PC_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${runtime}/pycache"
export TMPDIR="${runtime}/tmp"
export XDG_CACHE_HOME="${runtime}/cache"
export XDG_CONFIG_HOME="${runtime}/config"
export MPLCONFIGDIR="${runtime}/matplotlib"
export WANDB_DIR="${runtime}/wandb"
export WANDB_DATA_DIR="${runtime}/wandb-data"
export WANDB_CACHE_DIR="${runtime}/wandb-cache"
export WANDB_CONFIG_DIR="${runtime}/wandb-config"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-120}"
export OMP_NUM_THREADS="${FLOW2PC_THREADS:-4}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}" \
    "${WANDB_DIR}" "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"
exec > >(tee -a "${runtime}/evaluate.log") 2>&1

classifier="${LATENT_CLASSIFIER_CHECKPOINT:-}"
registry="${artifact_root}/outputs/evaluation/mnist-latent-classifier.txt"
if [[ -n "${classifier}" && ! -f "${classifier}" ]]; then
    echo "Missing LATENT_CLASSIFIER_CHECKPOINT: ${classifier}" >&2
    exit 1
fi
if [[ -z "${classifier}" && -f "${registry}" ]]; then
    registered="$(<"${registry}")"
    if [[ -f "${registered}" ]]; then
        classifier="${registered}"
    fi
fi
if [[ -z "${classifier}" ]]; then
    for candidate in \
        "${artifact_root}/outputs/evaluation/mnist-classifier-pytorch-example-seed1/checkpoints/last.pt" \
        "${repo_root}/outputs/evaluation/mnist-classifier-pytorch-example-seed1/checkpoints/last.pt"; do
        if [[ -f "${candidate}" ]]; then
            classifier="${candidate}"
            break
        fi
    done
fi
if [[ -z "${classifier}" ]]; then
    classifier_dir="${artifact_root}/outputs/evaluation/mnist-classifier-pytorch-example-seed1"
    if [[ -e "${classifier_dir}" ]]; then
        classifier_dir="${classifier_dir}-${timestamp}"
    fi
    echo "Preparing the fixed MNIST evaluator (14 epochs, training split only; seed 1)."
    "${python_bin}" -B -u train_mnist_classifier.py \
        --data-dir "${data_dir}" --output-dir "${classifier_dir}" \
        --device "${FLOW2PC_DEVICE:-cuda:0}" --seed 1 --epochs 14 \
        --num-workers "${LATENT_NUM_WORKERS:-0}" \
        --wandb-project flow2pc --wandb-mode "${WANDB_MODE:-online}"
    classifier="${classifier_dir}/checkpoints/last.pt"
fi
classifier="$(realpath -- "${classifier}")"
mkdir -p "${artifact_root}/outputs/evaluation"
printf '%s\n' "${classifier}" > "${registry}.${timestamp}.tmp"
mv "${registry}.${timestamp}.tmp" "${registry}"
echo "Using fixed classifier: ${classifier}"

"${python_bin}" -B -u train_mnist_latent.py evaluate \
    --data-dir "${data_dir}" --bundle-dir "${bundle_dir}" \
    --eqvae-root "${EQVAE_ROOT:-${repo_root}/../eqvae}" \
    --device "${FLOW2PC_DEVICE:-cuda:0}" --threads "${FLOW2PC_THREADS:-4}" \
    --seed "${LATENT_SEED:-0}" --split test \
    --num-generated "${LATENT_EVAL_SAMPLES:-10000}" \
    --batch-size "${LATENT_EVAL_BATCH_SIZE:-128}" --ode-steps "${LATENT_ODE_STEPS:-100}" \
    --wandb-project flow2pc --wandb-group mnist-latent-distillation \
    --wandb-mode "${WANDB_MODE:-online}" --wandb-run-name "mnist-latent-eval-${timestamp}" \
    "$@" --checkpoint "${checkpoint}" --classifier-checkpoint "${classifier}" --output-dir "${output}"
echo "Teacher/student fidelity report: ${output}/teacher_student_comparison.csv"
