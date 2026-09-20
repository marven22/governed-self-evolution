#!/usr/bin/env bash
# Fresh V5 sequential-development trajectories; these parent identities are development-only.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BANK_ROOT="/home/vmargapu/experiments/m14_coge_stratified_bank_v1"
RUN_ROOT="/home/vmargapu/experiments/m14_transactional_sequence_development_v1"
PYTHON="/home/vmargapu/tdmpc2-metaworld-official/bin/python"
export PYTHONPATH="${REPO_ROOT}/scripts:/home/vmargapu/src/tdmpc2/tdmpc2"
export LD_LIBRARY_PATH="/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs/usr/lib/x86_64-linux-gnu:/home/vmargapu/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd /home/vmargapu/src/tdmpc2/tdmpc2
for PARENT in 131 132 133 134; do
 for STAGE in bc r1 r3; do
  case "${STAGE}" in
   bc) CKPT="${BANK_ROOT}/parent${PARENT}/bc/policy_bc.pt";;
   r1) CKPT="${BANK_ROOT}/parent${PARENT}/dagger/policy_dagger_round1.pt";;
   r3) CKPT="${BANK_ROOT}/parent${PARENT}/dagger/policy_dagger_round3.pt";;
  esac
  ROOT="${RUN_ROOT}/p${PARENT}-${STAGE}"
  if [ ! -f "${ROOT}/trajectory.json" ]; then
   "${PYTHON}" "${REPO_ROOT}/scripts/run_m14_transactional_sequence_v5.py" --checkpoint "${CKPT}" --candidates "${REPO_ROOT}/configs/m14_coge_micro_candidates_v31.json" --proposal-order MICRO_25_75R MICRO_25_25R MICRO_25_50R --run-dir "${ROOT}" --controller-seed "${PARENT}" --execution-seed $((PARENT*1000+${#STAGE})) --demo-episodes 50 --verify-episodes 50 --alpha .05 --epsilon .05
  fi
 done
done
