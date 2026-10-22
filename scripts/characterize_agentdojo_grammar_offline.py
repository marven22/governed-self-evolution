#!/usr/bin/env python3
"""Offline bounds for every named update, with no model calls.

* reference oracle: what the update costs an agent that solves tasks perfectly;
* compromised oracle: what the update stops when the agent obeys every injection.

Instruction-isolation levels change only model-facing text, so the oracles
cannot respond to them; their rows equal HOLD by construction. The bounds are
informative for tool-permission levels. Run only on a development profile.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from gse_agentdojo.grammar import named_updates
from gse_agentdojo.runner import load_json, run_benchmark

ORACLES = {
    "reference": "configs/agentdojo_agent_ground_truth_oracle_v1.json",
    "compromised": "configs/agentdojo_agent_compromised_oracle_v1.json",
}
METRICS = ("benign_utility", "utility_under_attack", "injection_success", "attacker_contacted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path("configs/agentdojo_workspace_m1_smoke_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/agentdojo_grammar_offline_v1.json"))
    args = parser.parse_args()
    profile = load_json(REPO_ROOT / args.profile)
    if profile["cohort"] != "development":
        raise SystemExit("offline characterization is restricted to development profiles")

    rows = []
    with tempfile.TemporaryDirectory() as scratch:
        for update_name in named_updates():
            for oracle, agent_path in ORACLES.items():
                result = run_benchmark(
                    profile_path=REPO_ROOT / args.profile,
                    agent_path=REPO_ROOT / agent_path,
                    output_dir=Path(scratch) / f"{update_name}__{oracle}",
                    repo_root=REPO_ROOT,
                    update_name=update_name,
                )
                metrics = result["metrics"]
                rows.append({
                    "update": update_name,
                    "update_id": result["update"]["id"],
                    "oracle": oracle,
                    **{name: metrics[name]["rate"] for name in METRICS},
                    "policy_blocked_calls": metrics["policy_blocked_calls"],
                    "complete": metrics["complete"],
                })
    report = {
        "protocol": "gse-agentdojo-grammar-offline-bounds-v1",
        "profile": str(args.profile),
        "cohort": profile["cohort"],
        "note": "Instruction-isolation rows equal HOLD by construction: the oracles do not read the system message.",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'update':30} {'oracle':12} {'benign':>7} {'attacked':>9} {'inj.succ':>9} {'contacted':>10} {'blocked':>8}")
    for r in rows:
        print(
            f"{r['update']:30} {r['oracle']:12} {r['benign_utility']:>7.3f} {r['utility_under_attack']:>9.3f} "
            f"{r['injection_success']:>9.3f} {r['attacker_contacted']:>10.3f} {r['policy_blocked_calls']:>8}"
        )


if __name__ == "__main__":
    main()
