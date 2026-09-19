#!/usr/bin/env bash
# Refit frozen V4 selection on development parents, then evaluate untouched 129/130.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BANK_ROOT="/home/vmargapu/experiments/m14_coge_stratified_bank_v1"
DEV_ROOT="/home/vmargapu/experiments/m14_coge_stratified_development_v1"
RUN_ROOT="/home/vmargapu/experiments/m14_coge_v4_heldout_v1"
PYTHON="/home/vmargapu/tdmpc2-metaworld-official/bin/python"
MODEL="${RUN_ROOT}/governor_v4_runtime.pkl"; REPORT="${RUN_ROOT}/governor_v4_runtime.json"
export LD_LIBRARY_PATH="/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs/usr/lib/x86_64-linux-gnu:/home/vmargapu/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${REPO_ROOT}/scripts:/home/vmargapu/src/tdmpc2/tdmpc2${PYTHONPATH:+:$PYTHONPATH}"
cd /home/vmargapu/src/tdmpc2/tdmpc2
if [ ! -f "${MODEL}" ]; then
  "${PYTHON}" "${REPO_ROOT}/scripts/train_m14_coge_stratified_governor_v4.py" --data-glob "${DEV_ROOT}/*/rep*/transitions.json" --report "${REPORT}" --model "${MODEL}"
fi
for PARENT_SEED in 129 130; do
  for STAGE in bc r1 r3; do
    case "${STAGE}" in
      bc) CHECKPOINT="${BANK_ROOT}/parent${PARENT_SEED}/bc/policy_bc.pt" ;;
      r1) CHECKPOINT="${BANK_ROOT}/parent${PARENT_SEED}/dagger/policy_dagger_round1.pt" ;;
      r3) CHECKPOINT="${BANK_ROOT}/parent${PARENT_SEED}/dagger/policy_dagger_round3.pt" ;;
    esac
    ID="p${PARENT_SEED}-${STAGE}"; ROOT="${RUN_ROOT}/${ID}"
    if [ ! -f "${ROOT}/pre/completion.json" ]; then
      "${PYTHON}" "${REPO_ROOT}/scripts/run_m14_mt2_grammar_sweep.py" --checkpoint "${CHECKPOINT}" --candidates "${REPO_ROOT}/configs/m14_coge_micro_candidates_v31.json" --candidate-label HOLD --run-dir "${ROOT}/pre" --seed $((PARENT_SEED*100+${#STAGE})) --controller-seed "${PARENT_SEED}" --controller-id "${ID}" --parent-controller-id "p${PARENT_SEED}" --replicate-id 0 --demo-episodes 50 --eval-episodes 50 --record-plasticity-state --max-reach-drop-from-hold .05 --max-pick-place-drop-from-hold .05
    fi
    if [ ! -f "${ROOT}/plan.json" ]; then
      "${PYTHON}" "${REPO_ROOT}/scripts/plan_m14_coge_v31.py" --model "${MODEL}" --candidates "${REPO_ROOT}/configs/m14_coge_micro_candidates_v31.json" --pre-transition "${ROOT}/pre/transitions.json" --output "${ROOT}/plan.json"
    fi
    if [ ! -f "${ROOT}/full/completion.json" ]; then
      "${PYTHON}" "${REPO_ROOT}/scripts/run_m14_mt2_grammar_sweep.py" --checkpoint "${CHECKPOINT}" --candidates "${REPO_ROOT}/configs/m14_coge_micro_candidates_v31.json" --run-dir "${ROOT}/full" --seed $((PARENT_SEED*100+${#STAGE})) --controller-seed "${PARENT_SEED}" --controller-id "${ID}" --parent-controller-id "p${PARENT_SEED}" --replicate-id 0 --demo-episodes 50 --eval-episodes 50 --record-plasticity-state --max-reach-drop-from-hold .05 --max-pick-place-drop-from-hold .05
    fi
    "${PYTHON}" "${REPO_ROOT}/scripts/score_m14_coge_heldout.py" --plan "${ROOT}/plan.json" --transitions "${ROOT}/full/transitions.json" --output "${ROOT}/score.json"
  done
done
