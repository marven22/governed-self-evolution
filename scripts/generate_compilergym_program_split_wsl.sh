#!/usr/bin/env bash
# Freeze the cBench development/selection/held-out partition.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cg_venv="${CG_VENV:-/home/vmargapu/compiler-gym-venv}"
runtime_root="${CG_RUNTIME_COMPAT:-/home/vmargapu/compiler-gym-runtime-compat}"
output="${COMPILERGYM_SPLIT_OUTPUT:-$repo_root/configs/compilergym_program_split_v2.json}"

export LD_LIBRARY_PATH="$runtime_root/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$cg_venv/bin/python" "$repo_root/scripts/generate_compilergym_program_split.py" --output "$output"
