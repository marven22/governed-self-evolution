#!/usr/bin/env bash
# Train V3 on frozen development data, then precommit and score COGE on fresh controllers.
set -euo pipefail
if [ "$#" -eq 0 ]; then echo "usage: $0 SEED:CHECKPOINT [...]" >&2; exit 2; fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="/home/vmargapu/experiments/m14_coge_heldout_v31"
DATA_ROOT="/home/vmargapu/experiments/m14_v3_development_enrichment_v1"
PYTHON="/home/vmargapu/tdmpc2-metaworld-official/bin/python"
MODEL="${RUN_ROOT}/m14_rich_governor_runtime_v3.pkl"; REPORT="${RUN_ROOT}/m14_rich_governor_runtime_v3.json"
export LD_LIBRARY_PATH="/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs/usr/lib/x86_64-linux-gnu:/home/vmargapu/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${REPO_ROOT}/scripts:/home/vmargapu/src/tdmpc2/tdmpc2${PYTHONPATH:+:$PYTHONPATH}"
cd /home/vmargapu/src/tdmpc2/tdmpc2
if [ ! -f "${MODEL}" ]; then
  "${PYTHON}" "${REPO_ROOT}/scripts/train_m14_rich_governor_v3.py" --data-glob "${DATA_ROOT}/seed*/rep*/transitions.json" --report "${REPORT}" --model "${MODEL}"
fi
for RECORD in "$@"; do
  SEED="${RECORD%%:*}"; CHECKPOINT="${RECORD#*:}"; ROOT="${RUN_ROOT}/seed${SEED}"
  if [ "${SEED}" = "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then echo "invalid controller: ${RECORD}" >&2; exit 2; fi
  if [ ! -f "${ROOT}/pre/completion.json" ]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/run_m14_mt2_grammar_sweep.py" --checkpoint "${CHECKPOINT}" --candidates "${REPO_ROOT}/configs/m14_coge_micro_candidates_v31.json" --candidate-label HOLD --run-dir "${ROOT}/pre" --seed "${SEED}" --controller-seed "${SEED}" --replicate-id 0 --demo-episodes 50 --eval-episodes 50 --record-plasticity-state --max-reach-drop-from-hold .05 --max-pick-place-drop-from-hold .05
  fi
  if [ ! -f "${ROOT}/plan.json" ]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/plan_m14_coge_v31.py" --model "${MODEL}" --candidates "${REPO_ROOT}/configs/m14_coge_micro_candidates_v31.json" --pre-transition "${ROOT}/pre/transitions.json" --output "${ROOT}/plan.json"
  fi
  if [ ! -f "${ROOT}/full/completion.json" ]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/run_m14_mt2_grammar_sweep.py" --checkpoint "${CHECKPOINT}" --candidates "${REPO_ROOT}/configs/m14_coge_micro_candidates_v31.json" --run-dir "${ROOT}/full" --seed "${SEED}" --controller-seed "${SEED}" --replicate-id 0 --demo-episodes 50 --eval-episodes 50 --record-plasticity-state --max-reach-drop-from-hold .05 --max-pick-place-drop-from-hold .05
  fi
  "${PYTHON}" "${REPO_ROOT}/scripts/score_m14_coge_heldout.py" --plan "${ROOT}/plan.json" --transitions "${ROOT}/full/transitions.json" --output "${ROOT}/score.json"
done
