#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH="/home/vmargapu/.local/tdmpc2-legacy-mesa/rootfs/usr/lib/x86_64-linux-gnu:/home/vmargapu/.mujoco/mujoco210/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd /home/vmargapu/src/tdmpc2/tdmpc2
exec /home/vmargapu/tdmpc2-metaworld-official/bin/python run_m14_mt2_behavior_cloning.py "$@"
