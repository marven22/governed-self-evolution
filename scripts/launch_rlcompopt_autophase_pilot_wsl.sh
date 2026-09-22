#!/usr/bin/env bash
# Launch the pilot detached, leaving a PID and a log for explicit monitoring.
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/home/vmargapu/experiments/rlcompopt_autophase_nvp_pilot_v1}"
mkdir -p "${RUN_ROOT}"
nohup bash "$(dirname "$0")/run_rlcompopt_autophase_pilot_wsl.sh" \
  >"${RUN_ROOT}/driver.log" 2>&1 &
echo $! >"${RUN_ROOT}/pid"
echo "launched pid $(cat "${RUN_ROOT}/pid")"
