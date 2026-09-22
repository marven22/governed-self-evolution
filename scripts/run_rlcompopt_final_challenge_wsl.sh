#!/usr/bin/env bash
# One-time locked final promotion evaluation. Do not add tuning or fitting here.
set -euo pipefail
root=/mnt/c/Users/vmargapu/Documents/Research/governed-self-evolution
venv=/home/vmargapu/rlcompopt-venv
run=/home/vmargapu/experiments/rlcompopt_autophase_nvp_pilot_v1
export PYTHONPATH="$root"
export LD_LIBRARY_PATH=/home/vmargapu/compiler-gym-runtime-compat/lib/x86_64-linux-gnu
common=(
  "$venv/bin/python" "$root/scripts/run_rlcompopt_controlled_repair_benchmark.py"
  --model-db "$run/model.db"
  --trajectory-data /home/vmargapu/external/RLCompOpt/data/trajdataset_all10k-val-medium-all10k.json
  --vocab-db /home/vmargapu/external/RLCompOpt/data/all_ssl_vocab.db
  --split "$root/configs/rlcompopt_governor_final_challenge_v1.json"
  --model "$run/blame_gated_pairwise_governor_v2.pkl"
  --cohort final_evaluation
  --candidate-budget 24
  --donor-limit 10
  --damage-templates early,middle,quarter3,late
)
"${common[@]}" --output "$run/final_challenge_incumbent_v2.json" --strategy-name frozen_v2
"${common[@]}" --output "$run/final_challenge_challenger_v5.json" --strategy-name strategy_grammar_champion_v5 --risk-weight 1.75 --threshold 0.2
"$venv/bin/python" "$root/scripts/audit_rlcompopt_governor_final_promotion.py" \
  --incumbent "$run/final_challenge_incumbent_v2.json" \
  --challenger "$run/final_challenge_challenger_v5.json" \
  --protocol "$root/configs/rlcompopt_governor_final_challenge_v1.json" \
  --report "$run/final_challenge_promotion_audit_v1.json"
