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
            f" attacker_contacted={row['attacker_contacted']}"
        )
    )
    print(
        f"[{index}/{total}] {row['case_id']}: {row['status']} {outcome}"
        + (f" blocked={row['policy_blocked_calls']}" if row["policy_blocked_calls"] else "")
        + f" ({row['usage']['calls']} calls, {row['elapsed_seconds']}s"
        + (f", spent ${row['spent_usd']:.3f}" if row.get("cost_usd") else "")
        + ")",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--update", default="HOLD", help="named update from configs/agentdojo_update_grammar_v1.json")
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--max-usd", type=float, help="stop starting new cases once estimated spend would pass this")
    parser.add_argument("--output-dir", type=Path, help="default: runs/agentdojo/<profile>__<agent>__<update>__rep<k>")
    args = parser.parse_args()

    output_dir = args.output_dir or (
        REPO_ROOT
        / "runs"
        / "agentdojo"
        / f"{load_json(args.profile)['name']}__{load_json(args.agent)['name']}__{args.update}__rep{args.repetition}"
    )
    print(f"Running {args.profile.name} with {args.agent.name}, update {args.update} -> {output_dir}", file=sys.stderr, flush=True)
    try:
        result = run_benchmark(
            profile_path=args.profile,
            agent_path=args.agent,
            output_dir=output_dir,
            repo_root=REPO_ROOT,
            repetition=args.repetition,
            update_name=args.update,
            max_usd=args.max_usd,
            on_case=print_progress,
        )
    except (RuntimeError, FileExistsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(2)
    if result["budget"]["stop"] is not None:
        print(f"Spend cap reached: {result['budget']['stop']}", file=sys.stderr)
    print(json.dumps({"output_dir": str(output_dir), "metrics": result["metrics"]}, indent=2))
    if not result["metrics"]["complete"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
