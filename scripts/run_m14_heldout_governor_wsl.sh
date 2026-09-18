#!/usr/bin/env bash
# Precommit and score the frozen grammar governor on untouched controllers.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="/home/vmargapu/experiments/m14_grammar_heldout_v1"
PYTHON="/home/vmargapu/tdmpc2-metaworld-official/bin/python"
RUNTIME_MODEL="/home/vmargapu/experiments/m14_grammar_heldout_v1/m14_grammar_governor_runtime_v1.pkl"
RUNTIME_REPORT="/home/vmargapu/experiments/m14_grammar_heldout_v1/m14_grammar_governor_runtime_v1.json"
export LD_LIBRARY_PATH="/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs/usr/lib/x86_64-linux-gnu:/home/vmargapu/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${REPO_ROOT}/scripts:/home/vmargapu/src/tdmpc2/tdmpc2${PYTHONPATH:+:$PYTHONPATH}"
cd /home/vmargapu/src/tdmpc2/tdmpc2

# Pickle is NumPy-build-specific. Refit the already selected architecture and
# hyperparameter protocol from committed development data in this runtime.
if [ ! -f "${RUNTIME_MODEL}" ]; then
  "${PYTHON}" "${REPO_ROOT}/scripts/train_m14_grammar_governor.py" \
    --data-glob "${REPO_ROOT}/data/m14_development_*_v1.json" \
    --split "${REPO_ROOT}/configs/m14_governor_selection_split_v1.json" \
    --report "${RUNTIME_REPORT}" --model "${RUNTIME_MODEL}"
fi

CONTROLLERS=(
  "seed111|111|/home/vmargapu/experiments/m14_controller_bank_v1/seed111/dagger/policy_dagger.pt"
  "seed114|114|/home/vmargapu/experiments/m14_controller_bank_v1/seed114/dagger/policy_dagger.pt"
)

for RECORD in "${CONTROLLERS[@]}"; do
  IFS='|' read -r ID SEED CHECKPOINT <<< "${RECORD}"
  ROOT="${RUN_ROOT}/${ID}"
  if [ ! -f "${ROOT}/pre/completion.json" ]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/run_m14_mt2_grammar_sweep.py" \
      --checkpoint "${CHECKPOINT}" --candidates "${REPO_ROOT}/configs/m14_development_candidates_v1.json" \
      --candidate-label HOLD --run-dir "${ROOT}/pre" --seed "${SEED}" --demo-episodes 50 --eval-episodes 20
  fi
  if [ ! -f "${ROOT}/plan.json" ]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/plan_m14_grammar_governor.py" \
      --model "${RUNTIME_MODEL}" \
      --candidates "${REPO_ROOT}/configs/m14_development_candidates_v1.json" \
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
