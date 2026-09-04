#!/usr/bin/env bash
set -euo pipefail

endpoint_run="outputs/runs/endpoint/hclt-categorical-m16-bs1024-hybrid-em100p20-seed0"
endpoint_checkpoint="${endpoint_run}/checkpoints/final.pt"
endpoint_summary="${endpoint_run}/summary.json"
poll_seconds="${FLOW2PC_QUEUE_POLL_SECONDS:-60}"
requested_gpu="${FLOW2PC_GPU:-2}"
minimum_free_mb="${FLOW2PC_MIN_FREE_MB:-3000}"

while [[ ! -f "${endpoint_checkpoint}" || ! -f "${endpoint_summary}" ]]; do
  echo "$(date --iso-8601=seconds) waiting for the ICLR 2022 M=16 endpoint baseline"
  sleep "${poll_seconds}"
done

echo "Endpoint baseline completed; starting H64/Fourier-8 two-stage reconfiguration."
FLOW2PC_GPU="${requested_gpu}" \
FLOW2PC_MIN_FREE_MB="${minimum_free_mb}" \
ENDPOINT_PRETRAINING=em \
ENDPOINT_EM_EPOCHS=20 \
TIMESTEP_SAMPLING=uniform \
  bash scripts/run_endpoint_em_then_distill.sh
