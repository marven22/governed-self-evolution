#!/usr/bin/env bash
set -euo pipefail
root=/mnt/c/Users/vmargapu/Documents/Research/governed-self-evolution; venv=/home/vmargapu/rlcompopt-venv; run=/home/vmargapu/experiments/rlcompopt_autophase_nvp_pilot_v1
export PYTHONPATH="$root:$root/scripts"; export LD_LIBRARY_PATH=/home/vmargapu/compiler-gym-runtime-compat/lib/x86_64-linux-gnu
base=("$venv/bin/python" "$root/scripts/run_rlcompopt_csmith_repair_benchmark.py" --model-db "$run/model.db" --trajectory-data /home/vmargapu/external/RLCompOpt/data/trajdataset_all10k-val-medium-all10k.json --vocab-db /home/vmargapu/external/RLCompOpt/data/all_ssl_vocab.db --split "$root/configs/rlcompopt_csmith_selection_governor_v1.json" --cohort selection --candidate-budget 24 --donor-limit 10 --damage-templates early,middle,quarter3,late)
"${base[@]}" --model "$run/blame_gated_pairwise_governor_v2.pkl" --output "$run/csmith_selection_incumbent_v2.json" --strategy-name frozen_v2
"${base[@]}" --model "$run/csmith_pairwise_governor_v1.pkl" --output "$run/csmith_selection_challenger_v1.json" --strategy-name csmith_adapted_v1
"$venv/bin/python" "$root/scripts/audit_rlcompopt_governor_final_promotion.py" --incumbent "$run/csmith_selection_incumbent_v2.json" --challenger "$run/csmith_selection_challenger_v1.json" --protocol "$root/configs/rlcompopt_csmith_selection_governor_v1.json" --report "$run/csmith_selection_promotion_audit_v1.json"
