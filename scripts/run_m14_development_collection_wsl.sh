#!/usr/bin/env bash
# Run the preregistered grammar design on development controllers only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="/home/vmargapu/experiments/m14_grammar_development_v1"
PYTHON="/home/vmargapu/tdmpc2-metaworld-official/bin/python"
export LD_LIBRARY_PATH="/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs/usr/lib/x86_64-linux-gnu:/home/vmargapu/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${REPO_ROOT}/scripts:/home/vmargapu/src/tdmpc2/tdmpc2${PYTHONPATH:+:$PYTHONPATH}"
cd /home/vmargapu/src/tdmpc2/tdmpc2

CONTROLLERS=(
  "seed101|101|/home/vmargapu/experiments/m14_mt2_dagger_v1/policy_dagger.pt"
  "seed102|102|/home/vmargapu/experiments/m14_mt2_dagger_v2/policy_dagger.pt"
  "seed103|103|/home/vmargapu/experiments/m14_controller_bank_v1/seed103/dagger/policy_dagger.pt"
  "seed105|105|/home/vmargapu/experiments/m14_controller_bank_v1/seed105/dagger/policy_dagger.pt"
  "seed106|106|/home/vmargapu/experiments/m14_controller_bank_v1/seed106/dagger/policy_dagger.pt"
  "seed112|112|/home/vmargapu/experiments/m14_controller_bank_v1/seed112/dagger/policy_dagger.pt"
)

for RECORD in "${CONTROLLERS[@]}"; do
  IFS='|' read -r ID SEED CHECKPOINT <<< "${RECORD}"
  RUN_DIR="${RUN_ROOT}/${ID}"
  if [ -f "${RUN_DIR}/completion.json" ]; then
    continue
  fi
  "${PYTHON}" "${REPO_ROOT}/scripts/run_m14_mt2_grammar_sweep.py" \
    --checkpoint "${CHECKPOINT}" \
    --candidates "${REPO_ROOT}/configs/m14_development_candidates_v1.json" \
    --run-dir "${RUN_DIR}" --seed "${SEED}" \
    --demo-episodes 50 --eval-episodes 20
done
