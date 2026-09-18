#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT="/home/vmargapu/src/tdmpc2/tdmpc2"
MESA_ROOT="/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs"
MUJOCO_ROOT="/home/vmargapu/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$MESA_ROOT/usr/lib/x86_64-linux-gnu:$MUJOCO_ROOT/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$SOURCE_ROOT"
exec /home/vmargapu/tdmpc2-metaworld-official/bin/python run_m14_mt3_feasibility.py "$@"
