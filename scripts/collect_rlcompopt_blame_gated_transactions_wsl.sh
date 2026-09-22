#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/c/Users/vmargapu/Documents/Research/governed-self-evolution"
RLCOMPOPT_ROOT="${RLCOMPOPT_ROOT:-/home/vmargapu/external/RLCompOpt}"
VENV="${VENV:-/home/vmargapu/rlcompopt-venv}"
RUN_ROOT="${RUN_ROOT:-/home/vmargapu/experiments/rlcompopt_autophase_nvp_pilot_v1}"
source "${VENV}/bin/activate"
export LD_LIBRARY_PATH="/home/vmargapu/compiler-gym-runtime-compat/lib/x86_64-linux-gnu"

python "${PROJECT_ROOT}/scripts/collect_rlcompopt_blame_gated_transactions.py" \
  --model-db "${RUN_ROOT}/model.db" \
  --trajectory-data "${RLCOMPOPT_ROOT}/data/trajdataset_all10k-val-medium-all10k.json" \
  --vocab-db "${RLCOMPOPT_ROOT}/data/all_ssl_vocab.db" \
  --split "${PROJECT_ROOT}/configs/compilergym_program_split_v2.json" \
  --output "${RUN_ROOT}/blame_gated_development_transactions.json" \
  "$@"
