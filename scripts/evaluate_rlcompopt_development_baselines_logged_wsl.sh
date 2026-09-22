#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/vmargapu/experiments/rlcompopt_autophase_nvp_pilot_v1}"
mkdir -p "${RUN_ROOT}"
exec >>"${RUN_ROOT}/development_baselines.log" 2>&1
exec bash "$(dirname "$0")/evaluate_rlcompopt_development_baselines_wsl.sh"
