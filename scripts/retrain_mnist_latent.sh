#!/usr/bin/env bash
# Retrain the latent teacher and PC from scratch, then compare ODE generations.
# Reuse the frozen EQ-VAE, verified latent bundle, and fixed MNIST evaluator.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
seed="${LATENT_SEED:-0}"
timestamp="$(date -u +%Y%m%dT%H%M%S%NZ)"

# Always start a new teacher: an inherited checkpoint must not skip retraining.
export LATENT_TEACHER_CHECKPOINT=""
export LATENT_RUN_DIR="${LATENT_RETRAIN_RUN_DIR:-${LATENT_RUN_DIR:-/data/lthung/flow2pc/outputs/mnist-latent/retrain-seed${seed}-${timestamp}}}"
export LATENT_TEACHER_STEPS="${LATENT_TEACHER_STEPS:-30000}"
export LATENT_STUDENT_STEPS="${LATENT_STUDENT_STEPS:-20000}"
export LATENT_IMAGE_EVERY="${LATENT_IMAGE_EVERY:-0}"
export LATENT_VALIDATE_EVERY="${LATENT_VALIDATE_EVERY:-1000}"
export LATENT_CHECKPOINT_EVERY="${LATENT_CHECKPOINT_EVERY:-5000}"
export LATENT_EVAL_SAMPLES="${LATENT_EVAL_SAMPLES:-10000}"
export LATENT_ODE_STEPS=100
export WANDB_MODE="${WANDB_MODE:-online}"

echo "Retraining teacher (${LATENT_TEACHER_STEPS} steps) and student (${LATENT_STUDENT_STEPS} steps)."
echo "Image interval: ${LATENT_IMAGE_EVERY} (0 = end only); all generations use 100-step RK4."
echo "Final evaluation compares both models using FID/KID/IS-MNIST and additional fidelity metrics."
echo "W&B project: flow2pc; artifacts: ${LATENT_RUN_DIR}"

exec bash "${repo_root}/scripts/run_mnist_latent_distill.sh" "$@" --ode-steps 100
