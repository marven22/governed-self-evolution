#!/usr/bin/env bash
# Run the fail-closed CompilerGym feasibility gate in its isolated environment.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cg_venv="${CG_VENV:-/home/vmargapu/compiler-gym-venv}"
runtime_root="${CG_RUNTIME_COMPAT:-/home/vmargapu/compiler-gym-runtime-compat}"
report="${COMPILERGYM_FEASIBILITY_REPORT:-/home/vmargapu/experiments/compilergym_feasibility_v1/report.json}"

if [[ ! -x "$cg_venv/bin/python" ]]; then
  echo "CompilerGym virtual environment not found: $cg_venv" >&2
  exit 2
fi
if [[ ! -f "$runtime_root/lib/x86_64-linux-gnu/libtinfo.so.5" ]]; then
  echo "CompilerGym compatibility runtime not found: $runtime_root" >&2
  exit 2
fi

export LD_LIBRARY_PATH="$runtime_root/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$cg_venv/bin/python" "$repo_root/scripts/run_compilergym_feasibility.py" --report "$report"
