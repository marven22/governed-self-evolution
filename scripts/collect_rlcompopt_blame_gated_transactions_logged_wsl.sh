#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/vmargapu/experiments/rlcompopt_autophase_nvp_pilot_v1}"
mkdir -p "${RUN_ROOT}"
exec >>"${RUN_ROOT}/blame_gated_development_collection.log" 2>&1
exec bash "$(dirname "$0")/collect_rlcompopt_blame_gated_transactions_wsl.sh"
