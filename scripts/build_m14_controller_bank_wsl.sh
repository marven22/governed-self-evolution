#!/usr/bin/env bash
# Build independent MT2 DAgger controllers sequentially and resume safely.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 SEED [SEED ...]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BANK_ROOT="/home/vmargapu/experiments/m14_controller_bank_v1"
PYTHON="/home/vmargapu/tdmpc2-metaworld-official/bin/python"
export LD_LIBRARY_PATH="/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs/usr/lib/x86_64-linux-gnu:/home/vmargapu/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="${REPO_ROOT}/scripts:/home/vmargapu/src/tdmpc2/tdmpc2${PYTHONPATH:+:$PYTHONPATH}"
cd /home/vmargapu/src/tdmpc2/tdmpc2

for SEED in "$@"; do
  ROOT="${BANK_ROOT}/seed${SEED}"
  if [ ! -f "${ROOT}/bc/completion.json" ]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/run_m14_mt2_behavior_cloning.py" \
      --run-dir "${ROOT}/bc" --demo-episodes 100 --updates 5000 \
      --eval-episodes 50 --seed "${SEED}"
  fi
  if [ ! -f "${ROOT}/dagger/completion.json" ]; then
    "${PYTHON}" "${REPO_ROOT}/scripts/run_m14_mt2_dagger.py" \
      --run-dir "${ROOT}/dagger" --checkpoint "${ROOT}/bc/policy_bc.pt" \
      --expert-data "${ROOT}/bc/expert_mt2.pt" --rounds 3 \
      --rollout-episodes 10 --updates-per-round 2000 --eval-episodes 50 \
      --seed "${SEED}"
  fi
  "${PYTHON}" "${REPO_ROOT}/scripts/summarize_m14_controller_bank.py" --root "${BANK_ROOT}"
done
