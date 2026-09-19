#!/usr/bin/env bash
# Development-only paired transactional calibration; no held-out parent appears here.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BANK_ROOT="/home/vmargapu/experiments/m14_coge_stratified_bank_v1"
RUN_ROOT="/home/vmargapu/experiments/m14_transactional_v5_calibration_v1"
PYTHON="/home/vmargapu/tdmpc2-metaworld-official/bin/python"
export PYTHONPATH="${REPO_ROOT}/scripts:/home/vmargapu/src/tdmpc2/tdmpc2"
export LD_LIBRARY_PATH="/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs/usr/lib/x86_64-linux-gnu:/home/vmargapu/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd /home/vmargapu/src/tdmpc2/tdmpc2
for PARENT in 125 126 127 128; do
  for STAGE in bc r1 r3; do
    case "${STAGE}" in
      bc) CKPT="${BANK_ROOT}/parent${PARENT}/bc/policy_bc.pt" ;;
      r1) CKPT="${BANK_ROOT}/parent${PARENT}/dagger/policy_dagger_round1.pt" ;;
      r3) CKPT="${BANK_ROOT}/parent${PARENT}/dagger/policy_dagger_round3.pt" ;;
    esac
    for LABEL in MICRO_25_25R MICRO_25_50R MICRO_25_75R; do
      ROOT="${RUN_ROOT}/p${PARENT}-${STAGE}/${LABEL}"
      if [ ! -f "${ROOT}/result.json" ]; then
        "${PYTHON}" "${REPO_ROOT}/scripts/run_m14_transactional_v5.py" --checkpoint "${CKPT}" --candidates "${REPO_ROOT}/configs/m14_coge_micro_candidates_v31.json" --candidate-label "${LABEL}" --run-dir "${ROOT}" --controller-seed "${PARENT}" --execution-seed $((PARENT*1000+${#STAGE})) --demo-episodes 50 --verify-episodes 50 --alpha .05 --epsilon .05
      fi
    done
  done
done
