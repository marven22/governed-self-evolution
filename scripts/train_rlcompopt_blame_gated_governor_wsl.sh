#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/mnt/c/Users/vmargapu/Documents/Research/governed-self-evolution"
RUN_ROOT="${RUN_ROOT:-/home/vmargapu/experiments/rlcompopt_autophase_nvp_pilot_v1}"
VENV="${VENV:-/home/vmargapu/rlcompopt-venv}"
source "${VENV}/bin/activate"
python "${PROJECT_ROOT}/scripts/train_rlcompopt_blame_gated_governor.py" \
  --ledger "${RUN_ROOT}/blame_gated_development_ledger_v2.json" \
  --audit "${RUN_ROOT}/blame_gated_development_ledger_v2_audit.json" \
  --model "${RUN_ROOT}/blame_gated_governor_v1.pkl" \
  --report "${RUN_ROOT}/blame_gated_governor_v1_report.json" \
  --bootstrap 8
