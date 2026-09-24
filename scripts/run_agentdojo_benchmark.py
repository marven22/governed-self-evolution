#!/usr/bin/env python3
"""Run one AgentDojo profile with one agent configuration.

Example (offline, no model calls):

    python scripts/run_agentdojo_benchmark.py \
      --profile configs/agentdojo_workspace_m1_smoke_v1.json \
      --agent configs/agentdojo_agent_ground_truth_oracle_v1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from gse_agentdojo.runner import load_json, run_benchmark


def print_progress(index: int, total: int, row: dict) -> None:
    outcome = row["error"] or (
        f"utility={row['utility']}"
        + (
            ""
            if row["kind"] == "benign"
            else f" exposed={row['injection_exposed']} injection_succeeded={row['injection_task_success']}"
        )
    )
    print(
        f"[{index}/{total}] {row['case_id']}: {row['status']} {outcome} "
        f"({row['usage']['calls']} calls, {row['elapsed_seconds']}s)",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, help="default: runs/agentdojo/<profile>__<agent>__rep<k>")
    args = parser.parse_args()

    output_dir = args.output_dir or (
        REPO_ROOT
        / "runs"
        / "agentdojo"
        / f"{load_json(args.profile)['name']}__{load_json(args.agent)['name']}__rep{args.repetition}"
    )
    print(f"Running {args.profile.name} with {args.agent.name} -> {output_dir}", file=sys.stderr, flush=True)
    result = run_benchmark(
        profile_path=args.profile,
        agent_path=args.agent,
        output_dir=output_dir,
        repo_root=REPO_ROOT,
        repetition=args.repetition,
        on_case=print_progress,
    )
    print(json.dumps({"output_dir": str(output_dir), "metrics": result["metrics"]}, indent=2))
    if not result["metrics"]["complete"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
