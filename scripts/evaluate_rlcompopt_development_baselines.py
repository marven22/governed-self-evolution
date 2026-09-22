#!/usr/bin/env python3
"""Certificate-first development baselines for a frozen RLCompOpt controller."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from run_compilergym_feasibility import evaluate_fixed_policy


def random_index(benchmark: str, count: int) -> int:
    digest = hashlib.sha256(f"rlcompopt-random-coreset-v1|{benchmark}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % count


def certified(record: dict[str, Any]) -> bool:
    return bool(
        record["validation_available"]
        and record["validation"]["okay"]
        and record["validation"]["benchmark_semantics_validated"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-db", type=Path, required=True)
    parser.add_argument("--trajectory-data", type=Path, required=True)
    parser.add_argument("--vocab-db", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    from rlcompopt.model_testing import Environment

    split = json.loads(args.split.read_text(encoding="utf-8"))
    benchmarks: list[str] = split["development"]
    runner = Environment(
        str(args.model_db), None, 0, str(args.vocab_db), max_step=100,
        benchmarks=[], train_dataset_path=str(args.trajectory_data), sampling=False,
    )
    coreset = [[int(a) for a in sequence] for sequence in runner.actionseqs]
    table: list[dict[str, Any]] = []

    try:
        for benchmark in benchmarks:
            observation = runner.reset(benchmark)
            ranked = [int(index) for index in runner.get_model_action(observation)]
            top_index = ranked[0]
            sampled_index = random_index(benchmark, len(coreset))
            cache: dict[tuple[int, ...], dict[str, Any]] = {}

            def measure(actions: list[int]) -> dict[str, Any]:
                key = tuple(actions)
                if key not in cache:
                    cache[key] = evaluate_fixed_policy(benchmark, actions)
                return cache[key]

            hold = measure([])
            random_record = measure(coreset[sampled_index])
            top_record = measure(coreset[top_index])
            candidates = [
                (index, measure(actions)) for index, actions in enumerate(coreset)
            ]
            valid_candidates = [item for item in candidates if certified(item[1])]
            oracle_index, oracle_record = max(
                valid_candidates, key=lambda item: item[1]["total_reward"]
            )
            table.append(
                {
                    "benchmark": benchmark,
                    "hold": {"certificate": hold, "reward": hold["total_reward"], "certified": certified(hold)},
                    "random": {"coreset_index": sampled_index, "certificate": random_record, "reward": random_record["total_reward"], "certified": certified(random_record)},
                    "rlcompopt_top1": {"coreset_index": top_index, "certificate": top_record, "reward": top_record["total_reward"], "certified": certified(top_record)},
                    "oracle_coreset": {"coreset_index": oracle_index, "certificate": oracle_record, "reward": oracle_record["total_reward"], "certified": certified(oracle_record)},
                }
            )
    finally:
        runner.env.close()
        runner.model.connection.close()

    methods = ("hold", "random", "rlcompopt_top1", "oracle_coreset")
    summary = {
        method: {
            "mean_reward": sum(row[method]["reward"] for row in table) / len(table),
            "certified": sum(bool(row[method]["certified"]) for row in table),
        }
        for method in methods
    }
    report = {
        "protocol": "rlcompopt-development-baselines-v1",
        "cohort": "development only",
        "held_out_touched": False,
        "random_rule": "SHA256(rlcompopt-random-coreset-v1|benchmark) modulo 50",
        "oracle_note": "development-only upper bound over the released 50-sequence coreset; not deployable",
        "rows": table,
        "summary": summary,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
