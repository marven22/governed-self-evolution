#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${CG_VENV:-/home/vmargapu/compiler-gym-venv}"
compat="${CG_RUNTIME_COMPAT:-/home/vmargapu/compiler-gym-runtime-compat}"
export LD_LIBRARY_PATH="$compat/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bank="$root/configs/compilergym_bootstrap_controller_bank_v1.json"
"$venv/bin/python" "$root/scripts/build_compilergym_controller_bank.py" --output "$bank"
exec "$venv/bin/python" "$root/scripts/evaluate_compilergym_controller_bank.py" --bank "$bank" --split "$root/configs/compilergym_program_split_v2.json" --report /home/vmargapu/experiments/compilergym_bootstrap_bank_v1/report.json
