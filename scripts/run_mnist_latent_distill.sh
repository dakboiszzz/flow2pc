#!/usr/bin/env bash
# Export original MNIST posterior means, train a vanilla latent flow teacher,
# inspect/evaluate it on validation data, distill the PC, then evaluate on test.
# This file does nothing until you invoke it.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${FLOW2PC_PYTHON:-python}"
stage="all"
if [[ "${1:-}" == "--sweep" ]]; then
    shift
    exec "${python_bin}" -B -u scripts/run_mnist_latent_sweep.py "$@"
fi
if [[ "${1:-}" == "--teacher-only" ]]; then
    stage="teacher"
    shift
elif [[ "${1:-}" == "--student-only" ]]; then
    stage="student"
    shift
fi
artifact_root="/data/lthung/flow2pc"
seed="${LATENT_SEED:-0}"
run_label="${LATENT_RUN_NAME:-mnist-latent-seed${seed}-$(date -u +%Y%m%dT%H%M%S%NZ)}"
run_dir="${LATENT_RUN_DIR:-${artifact_root}/outputs/mnist-latent/${run_label}}"
bundle_dir="${LATENT_BUNDLE_DIR:-${artifact_root}/latents/mnist-eqvae-seed0}"
data_dir="${LATENT_DATA_DIR:-${artifact_root}/data/mnist}"
eqvae_root="${EQVAE_ROOT:-${repo_root}/../eqvae}"
vae_checkpoint="${LATENT_VAE_CHECKPOINT:-/data/lthung/eqvae/mnist-eqvae-seed0/best.pt}"
teacher_checkpoint="${LATENT_TEACHER_CHECKPOINT:-}"
classifier_checkpoint="${LATENT_CLASSIFIER_CHECKPOINT:-}"
if [[ "${stage}" == "teacher" ]]; then
    # This explicit mode always trains a new teacher, even if the calling
    # shell inherited the previous representation's teacher path.
    teacher_checkpoint=""
elif [[ "${stage}" == "student" && -z "${teacher_checkpoint}" ]]; then
    echo "--student-only requires LATENT_TEACHER_CHECKPOINT from the matching teacher run." >&2
    exit 1
fi

if [[ ! -f "${vae_checkpoint}" || ! -f "${eqvae_root}/train_eqvae/train_mnist.py" ]]; then
    echo "Missing EQ-VAE checkpoint or source. Set LATENT_VAE_CHECKPOINT / EQVAE_ROOT." >&2
    exit 1
fi
if [[ -n "${teacher_checkpoint}" && ! -f "${teacher_checkpoint}" ]]; then
    echo "Missing LATENT_TEACHER_CHECKPOINT: ${teacher_checkpoint}" >&2
    exit 1
fi
if [[ -n "${classifier_checkpoint}" && ! -f "${classifier_checkpoint}" ]]; then
    echo "Missing LATENT_CLASSIFIER_CHECKPOINT: ${classifier_checkpoint}" >&2
    exit 1
fi
if [[ -e "${run_dir}" ]]; then
    echo "Use a fresh LATENT_RUN_DIR: ${run_dir} already exists." >&2
    exit 1
fi
for artifact_path in "${run_dir}" "${bundle_dir}" "${data_dir}"; do
    case "$(realpath -m -- "${artifact_path}")" in
        "${artifact_root}"/*) ;;
        *) echo "Runtime paths must be under ${artifact_root}: ${artifact_path}" >&2; exit 1 ;;
    esac
done

# Set these before Python imports anything; every runtime file stays external.
export CUDA_VISIBLE_DEVICES="${FLOW2PC_GPU:-0}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${run_dir}/.runtime/pycache"
export TMPDIR="${run_dir}/.runtime/tmp"
export XDG_CACHE_HOME="${run_dir}/.runtime/cache"
export XDG_CONFIG_HOME="${run_dir}/.runtime/config"
export MPLCONFIGDIR="${run_dir}/.runtime/matplotlib"
export WANDB_DIR="${run_dir}/.runtime/wandb"
export WANDB_DATA_DIR="${run_dir}/.runtime/wandb-data"
export WANDB_CACHE_DIR="${run_dir}/.runtime/wandb-cache"
export WANDB_CONFIG_DIR="${run_dir}/.runtime/wandb-config"
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-300}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-300}"
export WANDB_HTTP_TIMEOUT="${WANDB_HTTP_TIMEOUT:-120}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}" \
    "${WANDB_DIR}" "${WANDB_DATA_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}"
exec > >(tee -a "${run_dir}/pipeline.log") 2>&1

common=(
    --eqvae-root "${eqvae_root}" --vae-checkpoint "${vae_checkpoint}"
    --bundle-dir "${bundle_dir}" --data-dir "${data_dir}"
    --device "${FLOW2PC_DEVICE:-cuda}" --threads "${FLOW2PC_THREADS:-4}"
    --seed "${seed}" --num-workers "${LATENT_NUM_WORKERS:-0}"
    --ode-steps "${LATENT_ODE_STEPS:-100}"
    --wandb-project flow2pc --wandb-group mnist-latent-distillation
    --wandb-mode "${WANDB_MODE:-online}"
)

if [[ ! -f "${bundle_dir}/manifest.json" ]]; then
    "${python_bin}" -B -u train_mnist_latent.py export "${common[@]}" \
        --download --batch-size 256 --output-dir "${run_dir}/export"
else
    echo "Reusing latent bundle (checksums verified on load): ${bundle_dir}"
fi

if [[ -z "${teacher_checkpoint}" ]]; then
    "${python_bin}" -B -u train_mnist_latent.py flow "${common[@]}" \
        --steps "${LATENT_TEACHER_STEPS:-30000}" --batch-size "${LATENT_TEACHER_BATCH_SIZE:-256}" \
        --lr "${LATENT_TEACHER_LR:-5e-4}" \
        --weight-decay "${LATENT_TEACHER_WEIGHT_DECAY:-0.01}" \
        --hidden-dim "${LATENT_TEACHER_HIDDEN_DIM:-512}" --num-blocks "${LATENT_TEACHER_BLOCKS:-6}" \
        --validate-every "${LATENT_VALIDATE_EVERY:-1000}" --image-every "${LATENT_IMAGE_EVERY:-0}" \
        --checkpoint-every "${LATENT_CHECKPOINT_EVERY:-5000}" \
        --wandb-run-name "${run_label}-teacher" --output-dir "${run_dir}/teacher"
    teacher_checkpoint="${run_dir}/teacher/checkpoints/best.pt"
fi

# Validate teacher generation before distillation. Teacher/student test metrics
# are computed only in the final evaluation, where both models are scored.
if [[ "${stage}" != "student" ]]; then
    LATENT_EVAL_DIR="${run_dir}/teacher-validation" \
    bash scripts/evaluate_mnist_latent.sh "${teacher_checkpoint}" "${common[@]}" --split validation \
    --num-generated "${LATENT_TEACHER_EVAL_SAMPLES:-5000}" --batch-size "${LATENT_EVAL_BATCH_SIZE:-128}" \
    --wandb-run-name "${run_label}-teacher-validation"
fi

if [[ "${stage}" == "teacher" ]]; then
    echo "Teacher training complete. Checkpoint: ${teacher_checkpoint}"
    exit 0
fi

"${python_bin}" -B -u train_mnist_latent.py distill "${common[@]}" \
    --teacher-checkpoint "${teacher_checkpoint}" \
    --pc-structure "${LATENT_PC_STRUCTURE:-hclt}" --pc-layer "${LATENT_PC_LAYER:-dense}" \
    --num-latents "${LATENT_NUM_LATENTS:-64}" --num-components "${LATENT_NUM_COMPONENTS:-256}" \
    --time-conditioning "${LATENT_TIME_CONDITIONING:-fourier}" \
    --time-embedding-dim "${LATENT_TIME_EMBEDDING_DIM:-16}" --time-order "${LATENT_TIME_ORDER:-4}" \
    --leaf-components "${LATENT_LEAF_COMPONENTS:-1}" --hclt-samples "${LATENT_HCLT_SAMPLES:-20000}" \
    --steps "${LATENT_STUDENT_STEPS:-20000}" --batch-size "${LATENT_STUDENT_BATCH_SIZE:-64}" \
    --lr "${LATENT_STUDENT_LR:-5e-4}" \
    --weight-decay "${LATENT_STUDENT_WEIGHT_DECAY:-0.01}" \
    --gradient-clip "${LATENT_STUDENT_GRADIENT_CLIP:-1.0}" \
    --validate-every "${LATENT_VALIDATE_EVERY:-1000}" --image-every "${LATENT_IMAGE_EVERY:-0}" \
    --checkpoint-every "${LATENT_CHECKPOINT_EVERY:-5000}" \
    --wandb-run-name "${run_label}-student" "$@" \
    --output-dir "${run_dir}/student" --teacher-checkpoint "${teacher_checkpoint}"

LATENT_EVAL_DIR="${run_dir}/test" \
    bash scripts/evaluate_mnist_latent.sh "${run_dir}/student/checkpoints/best.pt" "${common[@]}" --split test \
    --num-generated "${LATENT_EVAL_SAMPLES:-10000}" --batch-size "${LATENT_EVAL_BATCH_SIZE:-128}" \
    --wandb-run-name "${run_label}-test"

echo "Latent MNIST distillation complete. Artifacts: ${run_dir}"
