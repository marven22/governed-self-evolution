#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; BANK="/home/vmargapu/experiments/m14_coge_stratified_bank_v1"; ROOT="/home/vmargapu/experiments/m14_cpe_balanced_development_v1"; PY="/home/vmargapu/tdmpc2-metaworld-official/bin/python"
export PYTHONPATH="${REPO_ROOT}/scripts:/home/vmargapu/src/tdmpc2/tdmpc2"; export LD_LIBRARY_PATH="/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs/usr/lib/x86_64-linux-gnu:/home/vmargapu/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd /home/vmargapu/src/tdmpc2/tdmpc2
for P in 143 144 145 146 147 148 149 150; do for S in bc r1 r3; do
 case "$S" in bc) C="$BANK/parent$P/bc/policy_bc.pt";;r1) C="$BANK/parent$P/dagger/policy_dagger_round1.pt";;r3) C="$BANK/parent$P/dagger/policy_dagger_round3.pt";;esac
 D="$ROOT/p$P-$S"; [ -f "$D/completion.json" ] || "$PY" "$REPO_ROOT/scripts/run_m14_cpe_balanced.py" --checkpoint "$C" --candidates "$REPO_ROOT/configs/m14_coge_micro_candidates_v31.json" --run-dir "$D" --controller-id "p$P-$S" --parent-id "p$P" --controller-seed "$P" --seed $((P*1000+${#S})) --demo-episodes 50 --episodes 50 --replicates 2 --alpha .05 --epsilon .05
done; done
