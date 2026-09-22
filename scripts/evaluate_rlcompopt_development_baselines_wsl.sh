#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/c/Users/vmargapu/Documents/Research/governed-self-evolution"
RLCOMPOPT_ROOT="${RLCOMPOPT_ROOT:-/home/vmargapu/external/RLCompOpt}"
VENV="${VENV:-/home/vmargapu/rlcompopt-venv}"
RUN_ROOT="${RUN_ROOT:-/home/vmargapu/experiments/rlcompopt_autophase_nvp_pilot_v1}"
source "${VENV}/bin/activate"
export LD_LIBRARY_PATH="/home/vmargapu/compiler-gym-runtime-compat/lib/x86_64-linux-gnu"
python "${PROJECT_ROOT}/scripts/evaluate_rlcompopt_development_baselines.py" \
  --model-db "${RUN_ROOT}/model.db" \
  --trajectory-data "${RLCOMPOPT_ROOT}/data/trajdataset_all10k-val-medium-all10k.json" \
  --vocab-db "${RLCOMPOPT_ROOT}/data/all_ssl_vocab.db" \
  --split "${PROJECT_ROOT}/configs/compilergym_program_split_v2.json" \
  --report "${RUN_ROOT}/development_baselines.json"
