#!/usr/bin/env python3
"""Certified held-out execution of a frozen RLCompOpt controller.

The controller ranks RLCompOpt's released coreset from an initial AutoPhase
observation. This evaluator never uses a held-out outcome to select a coreset
member: it executes the highest-ranked member once and certifies it with the
existing cBench reference-output contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_compilergym_feasibility import evaluate_fixed_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-db", type=Path, required=True)
    parser.add_argument("--trajectory-data", type=Path, required=True)
    parser.add_argument("--vocab-db", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    # Imported lazily so this repository stays usable without RLCompOpt.
    from rlcompopt.model_testing import Environment

    split = json.loads(args.split.read_text(encoding="utf-8"))
    benchmarks: list[str] = split["held_out"]
    runner = Environment(
        str(args.model_db),
        None,  # latest saved state; no selection on the held-out cohort
        0,
        str(args.vocab_db),
        max_step=100,
        benchmarks=[],
        train_dataset_path=str(args.trajectory_data),
        sampling=False,
    )

    records: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        observation = runner.reset(benchmark)
        ranked_indices = runner.get_model_action(observation, return_prob=False)
        coreset_index = int(ranked_indices[0])
        actions = [int(action) for action in runner.actionseqs[coreset_index]]
        certificate = evaluate_fixed_policy(benchmark, actions)
        records.append(
            {
                "benchmark": benchmark,
                "coreset_index": coreset_index,
                "action_count": len(actions),
                "actions": actions,
                "certificate": certificate,
                "certified": bool(
                    certificate["validation_available"]
                    and certificate["validation"]["okay"]
                    and certificate["validation"]["benchmark_semantics_validated"]
                ),
                "improvement_over_oz": float(certificate["total_reward"]),
            }
        )

    certified = [record for record in records if record["certified"]]
    report = {
        "protocol": "rlcompopt-frozen-heldout-cbench-v1",
        "model_db": str(args.model_db),
        "selection_rule": "highest RLCompOpt coreset score from initial AutoPhase observation",
        "held_out_touched": True,
        "records": records,
        "summary": {
            "programs": len(records),
            "certified": len(certified),
            "mean_improvement_over_oz": (
                sum(record["improvement_over_oz"] for record in certified) / len(certified)
                if certified
                else None
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
