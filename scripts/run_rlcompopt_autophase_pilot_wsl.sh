#!/usr/bin/env bash
# Train the public RLCompOpt AutoPhase normalized-value controller in isolation.
set -euo pipefail

PROJECT_ROOT="/mnt/c/Users/vmargapu/Documents/Research/governed-self-evolution"
RLCOMPOPT_ROOT="${RLCOMPOPT_ROOT:-/home/vmargapu/external/RLCompOpt}"
VENV="${VENV:-/home/vmargapu/rlcompopt-venv}"
RUN_ROOT="${RUN_ROOT:-/home/vmargapu/experiments/rlcompopt_autophase_nvp_pilot_v1}"
TOTAL_STEPS="${TOTAL_STEPS:-1000}"

source "${VENV}/bin/activate"
export LD_LIBRARY_PATH="/home/vmargapu/compiler-gym-runtime-compat/lib/x86_64-linux-gnu"
mkdir -p "${RUN_ROOT}"

cd "${RLCOMPOPT_ROOT}"
exec python -m torch.distributed.run --standalone --nproc_per_node=1 \
  rlcompopt/train.py --config-name autophase \
  distributed=true device=cuda dataset.num_workers=1 dataset.pre_load=false \
  train_batch_size=256 eval_batch_size=256 total_steps="${TOTAL_STEPS}" \
  warmup_steps=100 eval_frequence=100 save_frequence=100 print_frequence=100 \
  hydra.run.dir="${RUN_ROOT}"
