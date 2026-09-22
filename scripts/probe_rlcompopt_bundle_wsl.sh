#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/c/Users/vmargapu/Documents/Research/governed-self-evolution"
RLCOMPOPT_ROOT="${RLCOMPOPT_ROOT:-/home/vmargapu/external/RLCompOpt}"
VENV="${VENV:-/home/vmargapu/tdmpc2-metaworld-official}"
OUTPUT="${OUTPUT:-/home/vmargapu/experiments/rlcompopt_probe_v1/bundle_audit.json}"

source "${VENV}/bin/activate"
python "${PROJECT_ROOT}/scripts/probe_rlcompopt_bundle.py" \
  --rlcompopt-root "${RLCOMPOPT_ROOT}" \
  --output "${OUTPUT}"
