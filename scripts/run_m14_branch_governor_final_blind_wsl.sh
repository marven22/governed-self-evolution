#!/usr/bin/env bash
# Final support-constrained governor evaluation on untouched parents 139--142.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; BANK="/home/vmargapu/experiments/m14_coge_stratified_bank_v1"; ROOT="/home/vmargapu/experiments/m14_branch_governor_final_blind_v1"; DEV="/home/vmargapu/experiments/m14_branch_certify_replay_v1"; PY="/home/vmargapu/tdmpc2-metaworld-official/bin/python"; MODEL="$ROOT/governor.json"; LABELS=(MICRO_25_25R MICRO_25_50R MICRO_25_75R)
export PYTHONPATH="${REPO_ROOT}/scripts:/home/vmargapu/src/tdmpc2/tdmpc2"; export LD_LIBRARY_PATH="/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs/usr/lib/x86_64-linux-gnu:/home/vmargapu/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd /home/vmargapu/src/tdmpc2/tdmpc2; mkdir -p "$ROOT"
[ -f "$MODEL" ] || "$PY" "$REPO_ROOT/scripts/train_m14_branch_governor_v1.py" --archive "$DEV/archive.json" --report "$ROOT/governor_development_report.json" --model "$MODEL"
for P in 139 140 141 142; do for S in bc r1 r3; do
 case "$S" in bc) C="$BANK/parent$P/bc/policy_bc.pt";;r1) C="$BANK/parent$P/dagger/policy_dagger_round1.pt";;r3) C="$BANK/parent$P/dagger/policy_dagger_round3.pt";;esac
 D="$ROOT/p$P-$S"; PLAN="$D/plan.json"; mkdir -p "$D"
 [ -f "$PLAN" ] || "$PY" "$REPO_ROOT/scripts/plan_m14_branch_governor_v1.py" --checkpoint "$C" --model "$MODEL" --candidates "$REPO_ROOT/configs/m14_coge_micro_candidates_v31.json" --labels "${LABELS[@]}" --controller-id "p$P-$S" --controller-seed "$P" --seed $((P*1000+${#S})) --output "$PLAN"
 LABEL=$("$PY" -c "import json; print(json.load(open('$PLAN'))['chosen'])")
 [ "$LABEL" = HOLD ] || [ -f "$D/trajectory.json" ] || "$PY" "$REPO_ROOT/scripts/run_m14_branch_certify_v5.py" --checkpoint "$C" --candidates "$REPO_ROOT/configs/m14_coge_micro_candidates_v31.json" --labels "$LABEL" --run-dir "$D" --controller-id "p$P-$S" --controller-seed "$P" --seed $((P*1000+${#S})) --max-commits 1 --screen-episodes 50 --confirm-episodes 200 --alpha .05 --epsilon .05
done; done
