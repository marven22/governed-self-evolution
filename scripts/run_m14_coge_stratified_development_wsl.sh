#!/usr/bin/env bash
# Collect paired COGE transitions from development parents only (125--128).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BANK_ROOT="/home/vmargapu/experiments/m14_coge_stratified_bank_v1"
RUN_ROOT="/home/vmargapu/experiments/m14_coge_stratified_development_v1"
PYTHON="/home/vmargapu/tdmpc2-metaworld-official/bin/python"
export LD_LIBRARY_PATH="/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs/usr/lib/x86_64-linux-gnu:/home/vmargapu/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${REPO_ROOT}/scripts:/home/vmargapu/src/tdmpc2/tdmpc2${PYTHONPATH:+:$PYTHONPATH}"
cd /home/vmargapu/src/tdmpc2/tdmpc2

for PARENT_SEED in 125 126 127 128; do
  PARENT_ID="p${PARENT_SEED}"
  for STAGE in bc r1 r3; do
    case "${STAGE}" in
      bc) CHECKPOINT="${BANK_ROOT}/parent${PARENT_SEED}/bc/policy_bc.pt" ;;
      r1) CHECKPOINT="${BANK_ROOT}/parent${PARENT_SEED}/dagger/policy_dagger_round1.pt" ;;
      r3) CHECKPOINT="${BANK_ROOT}/parent${PARENT_SEED}/dagger/policy_dagger_round3.pt" ;;
    esac
    CONTROLLER_ID="${PARENT_ID}-${STAGE}"
    for REPLICATE in 0 1; do
      EXECUTION_SEED=$((PARENT_SEED * 100 + 10 * ${#STAGE} + REPLICATE))
      ROOT="${RUN_ROOT}/${CONTROLLER_ID}/rep${REPLICATE}"
      if [ ! -f "${ROOT}/completion.json" ]; then
        "${PYTHON}" "${REPO_ROOT}/scripts/run_m14_mt2_grammar_sweep.py" \
          --checkpoint "${CHECKPOINT}" --candidates "${REPO_ROOT}/configs/m14_coge_micro_candidates_v31.json" \
          --run-dir "${ROOT}" --seed "${EXECUTION_SEED}" --controller-seed "${PARENT_SEED}" \
          --controller-id "${CONTROLLER_ID}" --parent-controller-id "${PARENT_ID}" --replicate-id "${REPLICATE}" \
          --demo-episodes 50 --eval-episodes 50 --record-plasticity-state \
          --max-reach-drop-from-hold .05 --max-pick-place-drop-from-hold .05
      fi
    done
  done
done
