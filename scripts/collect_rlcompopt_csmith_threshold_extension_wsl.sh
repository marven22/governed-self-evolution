#!/usr/bin/env bash
set -euo pipefail
root=/mnt/c/Users/vmargapu/Documents/Research/governed-self-evolution
venv=/home/vmargapu/rlcompopt-venv
run=/home/vmargapu/experiments/rlcompopt_autophase_nvp_pilot_v1
export PYTHONPATH="$root:$root/scripts"
export LD_LIBRARY_PATH=/home/vmargapu/compiler-gym-runtime-compat/lib/x86_64-linux-gnu
exec "$venv/bin/python" "$root/scripts/collect_rlcompopt_blame_gated_transactions.py" --model-db "$run/model.db" --trajectory-data /home/vmargapu/external/RLCompOpt/data/trajdataset_all10k-val-medium-all10k.json --vocab-db /home/vmargapu/external/RLCompOpt/data/all_ssl_vocab.db --split "$root/configs/rlcompopt_csmith_adaptation_threshold_extension_v1.json" --cohort development --certificate-profile csmith --seed-ledger "$run/csmith_adaptation_development_ledger_v3.json" --output "$run/csmith_adaptation_development_ledger_v4.json" --candidate-budget 12 --parent-ranks 0,1 --donor-limit 10
