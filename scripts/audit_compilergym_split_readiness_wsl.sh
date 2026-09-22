#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cg_venv="${CG_VENV:-/home/vmargapu/compiler-gym-venv}"
runtime_root="${CG_RUNTIME_COMPAT:-/home/vmargapu/compiler-gym-runtime-compat}"
report="${COMPILERGYM_READINESS_REPORT:-/home/vmargapu/experiments/compilergym_split_readiness_v1/report.json}"

export LD_LIBRARY_PATH="$runtime_root/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$cg_venv/bin/python" "$repo_root/scripts/audit_compilergym_split_readiness.py" \
  --split "$repo_root/configs/compilergym_program_split_v2.json" \
  --report "$report"
