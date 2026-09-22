#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; venv="${CG_VENV:-/home/vmargapu/compiler-gym-venv}"; compat="${CG_RUNTIME_COMPAT:-/home/vmargapu/compiler-gym-runtime-compat}"
export LD_LIBRARY_PATH="$compat/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
grammar="$root/configs/compilergym_edit_grammar_v1.json"
"$venv/bin/python" "$root/scripts/generate_compilergym_edit_grammar.py" --output "$grammar"
exec "$venv/bin/python" "$root/scripts/collect_compilergym_evolution.py" --bank "$root/configs/compilergym_bootstrap_controller_bank_v1.json" --grammar "$grammar" --split "$root/configs/compilergym_program_split_v2.json" --output /home/vmargapu/experiments/compilergym_evolution_transitions_v1/transitions.json
