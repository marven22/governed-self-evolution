#!/usr/bin/env bash
# Evaluate V2 only on fresh, qualified controllers supplied as SEED:CHECKPOINT.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 SEED:CHECKPOINT [SEED:CHECKPOINT ...]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="/home/vmargapu/experiments/m14_baseline_anchored_heldout_v2"
PYTHON="/home/vmargapu/tdmpc2-metaworld-official/bin/python"
MODEL="${RUN_ROOT}/m14_baseline_anchored_governor_runtime_v2.pkl"
REPORT="${RUN_ROOT}/m14_baseline_anchored_governor_runtime_v2.json"
export LD_LIBRARY_PATH="/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs/usr/lib/x86_64-linux-gnu:/home/vmargapu/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${REPO_ROOT}/scripts:/home/vmargapu/src/tdmpc2/tdmpc2${PYTHONPATH:+:$PYTHONPATH}"
cd /home/vmargapu/src/tdmpc2/tdmpc2

# Refit in the execution environment so serialized NumPy internals are never
# shared across Windows/WSL. This uses only committed development transitions.
if [ ! -f "${MODEL}" ]; then
  "${PYTHON}" "${REPO_ROOT}/scripts/train_m14_baseline_anchored_governor.py" \
    --data-glob "${REPO_ROOT}/data/m14_development_*_v1.json" \
    --split "${REPO_ROOT}/configs/m14_governor_selection_split_v1.json" \
    --report "${REPORT}" --model "${MODEL}"
fi

for RECORD in "$@"; do
  SEED="${RECORD%%:*}"
  CHECKPOINT="${RECORD#*:}"
  if [ "${SEED}" = "${CHECKPOINT}" ] || [ ! -f "${CHECKPOINT}" ]; then
    echo "invalid controller record: ${RECORD}" >&2
    exit 2
  fi
  ROOT="${RUN_ROOT}/seed${SEED}"
  if [ ! -f "${ROOT}/pre/completion.json" ]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/run_m14_mt2_grammar_sweep.py" \
      --checkpoint "${CHECKPOINT}" --candidates "${REPO_ROOT}/configs/m14_development_candidates_v1.json" \
      --candidate-label HOLD --run-dir "${ROOT}/pre" --seed "${SEED}" --demo-episodes 50 --eval-episodes 20
  fi
  if [ ! -f "${ROOT}/plan.json" ]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/plan_m14_baseline_anchored_governor.py" \
      --model "${MODEL}" --candidates "${REPO_ROOT}/configs/m14_development_candidates_v1.json" \
      --pre-transition "${ROOT}/pre/transitions.json" --output "${ROOT}/plan.json"
  fi
  if [ ! -f "${ROOT}/full/completion.json" ]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/run_m14_mt2_grammar_sweep.py" \
      --checkpoint "${CHECKPOINT}" --candidates "${REPO_ROOT}/configs/m14_development_candidates_v1.json" \
      --run-dir "${ROOT}/full" --seed "${SEED}" --demo-episodes 50 --eval-episodes 20
  fi
  "${PYTHON}" "${REPO_ROOT}/scripts/score_m14_grammar_governor_heldout.py" \
    --plan "${ROOT}/plan.json" --transitions "${ROOT}/full/transitions.json" --output "${ROOT}/score.json"
done
