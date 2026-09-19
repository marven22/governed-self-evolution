#!/usr/bin/env bash
# Add repeated, richer V3 transitions from controllers already burned by V2.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="/home/vmargapu/experiments/m14_v3_development_enrichment_v1"
BANK_ROOT="/home/vmargapu/experiments/m14_controller_bank_v1"
PYTHON="/home/vmargapu/tdmpc2-metaworld-official/bin/python"
export LD_LIBRARY_PATH="/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs/usr/lib/x86_64-linux-gnu:/home/vmargapu/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${REPO_ROOT}/scripts:/home/vmargapu/src/tdmpc2/tdmpc2${PYTHONPATH:+:$PYTHONPATH}"
cd /home/vmargapu/src/tdmpc2/tdmpc2

# Two independent executions per controller-action are a first calibration
# dataset; increase this only through a new, documented protocol revision.
for CONTROLLER_SEED in 117 120 121 122 123; do
  CHECKPOINT="${BANK_ROOT}/seed${CONTROLLER_SEED}/dagger/policy_dagger.pt"
  for REPLICATE in 0 1; do
    EXECUTION_SEED=$((CONTROLLER_SEED * 100 + REPLICATE))
    ROOT="${RUN_ROOT}/seed${CONTROLLER_SEED}/rep${REPLICATE}"
    if [ ! -f "${ROOT}/completion.json" ]; then
      "${PYTHON}" "${REPO_ROOT}/scripts/run_m14_mt2_grammar_sweep.py" \
        --checkpoint "${CHECKPOINT}" --candidates "${REPO_ROOT}/configs/m14_development_candidates_v1.json" \
        --run-dir "${ROOT}" --seed "${EXECUTION_SEED}" --controller-seed "${CONTROLLER_SEED}" --replicate-id "${REPLICATE}" \
        --demo-episodes 50 --eval-episodes 50 --record-plasticity-state \
        --max-reach-drop-from-hold 0.05 --max-pick-place-drop-from-hold 0.05
    fi
  done
done
