#!/usr/bin/env python3
"""Build and validate AgentDojo transition rows from completed HOLD and update runs.

Example:

    python scripts/build_agentdojo_transitions.py \
      --hold runs/agentdojo/workspace_m3_dev_v1__claude_haiku_4_5_cached__HOLD__rep0 \
      --updates runs/agentdojo/workspace_m3_dev_v1__claude_haiku_4_5_cached__*__rep0 \
      --output data/agentdojo_transitions_dev_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentdojo.task_suite.load_suites import get_suite

from gse_agentdojo.transitions import CAPABILITIES, TRANSITION_PROTOCOL, build_transitions, validate_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hold", type=Path, required=True)
    parser.add_argument("--updates", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    updates = [d for d in args.updates if d.resolve() != args.hold.resolve()]  # tolerate a glob that matches HOLD
    hold_profile = json.loads((args.hold / "result.json").read_text())["profile"]["content"]
    suite = get_suite(hold_profile["benchmark_version"], hold_profile["suite"])
    rows = build_transitions(args.hold, updates, suite)
    for row in rows:
        validate_row(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"protocol": TRANSITION_PROTOCOL, "rows": rows}, indent=2) + "\n")

    pre = rows[0]["pre_capability"] if rows else {}
    print(f"{'update':30}" + "".join(f"{name[:12]:>14}" for name in CAPABILITIES) + f"{'utility':>9}{'feasible':>10}")
    print(f"{'HOLD':30}" + "".join(f"{_fmt(pre.get(n)):>14}" for n in CAPABILITIES)
          + f"{_fmt(rows[0]['utility']['pre'] if rows else None):>9}")
    for row in rows:
        post = row["post_capability"]
        print(f"{row['update']['name']:30}" + "".join(f"{_fmt(post[n]):>14}" for n in CAPABILITIES)
              + f"{_fmt(row['utility']['post']):>9}{str(row['feasible']):>10}")
    print(f"wrote {len(rows)} validated rows to {args.output}")


def _fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


if __name__ == "__main__":
    main()
