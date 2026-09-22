#!/usr/bin/env python3
"""Collect development-only, certificate-backed parent-child edit outcomes.

This is the foundation ledger for a later governor.  It does not train a
governor, tune on selection/held-out cohorts, or promote an uncertified child.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from rlcompopt_edit_grammar import apply_edit, generate_candidates
from run_compilergym_feasibility import evaluate_fixed_policy


def certified(record: dict[str, Any]) -> bool:
    validation = record.get("validation", {})
    return bool(
        record.get("validation_available")
        and validation.get("okay")
        and validation.get("benchmark_semantics_validated")
    )


def stable_reward(first: dict[str, Any], second: dict[str, Any], tolerance: float) -> bool:
    return abs(float(first["total_reward"]) - float(second["total_reward"])) <= tolerance


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize(transactions: list[dict[str, Any]], blame_penalty: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in transactions:
        grouped[row["edit"]["edit_program_id"]].append(row)
    programs: list[dict[str, Any]] = []
    for program_id, rows in sorted(grouped.items()):
        certified_rows = [row for row in rows if row["outcome"]["certified_child"]]
        deltas = [float(row["outcome"]["delta_reward"]) for row in certified_rows]
        blame_rate = sum(row["outcome"]["blame"] for row in rows) / len(rows)
        invalid_rate = sum(not row["outcome"]["certified_child"] for row in rows) / len(rows)
        mean_delta = statistics.fmean(deltas) if deltas else None
        programs.append({
            "edit_program_id": program_id,
            "trials": len(rows),
            "certified_trials": len(certified_rows),
            "mean_certified_delta": mean_delta,
            "blame_rate": blame_rate,
            "invalid_rate": invalid_rate,
            "utility": (mean_delta if mean_delta is not None else -1.0) - blame_penalty * blame_rate,
        })
    return {"edit_program_archive": programs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-db", type=Path, required=True)
    parser.add_argument("--trajectory-data", type=Path, required=True)
    parser.add_argument("--vocab-db", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-budget", type=int, default=12)
    parser.add_argument(
        "--cohort",
        choices=("development", "selection"),
        default="development",
        help="Cohort to collect. Selection is evaluation-only and must never seed training.",
    )
    parser.add_argument(
        "--parent-ranks",
        default="0",
        help="Comma-separated ranks from the frozen controller's ordered coreset."
    )
    parser.add_argument("--donor-limit", type=int, default=5)
    parser.add_argument(
        "--seed-ledger",
        type=Path,
        default=None,
        help="Existing ledger from the same cohort to retain when writing an expanded ledger.",
    )
    parser.add_argument("--max-actions", type=int, default=64)
    parser.add_argument("--improvement-epsilon", type=float, default=0.01)
    parser.add_argument("--blame-margin", type=float, default=0.01)
    parser.add_argument("--repeat-tolerance", type=float, default=1e-9)
    parser.add_argument("--blame-penalty", type=float, default=1.0)
    parser.add_argument("--max-programs", type=int, default=None, help="Optional smoke-test limit.")
    args = parser.parse_args()
    parent_ranks = [int(value) for value in args.parent_ranks.split(",") if value.strip()]
    if not parent_ranks or min(parent_ranks) < 0:
        raise ValueError("parent-ranks must contain non-negative ranks")

    from rlcompopt.model_testing import Environment

    split = json.loads(args.split.read_text(encoding="utf-8"))
    benchmarks: list[str] = split[args.cohort]
    if args.max_programs is not None:
        benchmarks = benchmarks[: args.max_programs]
    runner = Environment(
        str(args.model_db), None, 0, str(args.vocab_db), max_step=100,
        benchmarks=[], train_dataset_path=str(args.trajectory_data), sampling=False,
    )
    coreset = [[int(action) for action in sequence] for sequence in runner.actionseqs]
    transactions: list[dict[str, Any]] = []
    if args.seed_ledger is not None:
        seed = json.loads(args.seed_ledger.read_text(encoding="utf-8"))
        expected_cohort = f"{args.cohort} only"
        if seed.get("cohort") != expected_cohort or seed.get("held_out_touched"):
            raise ValueError("seed-ledger must be from the same non-held-out cohort")
        transactions = list(seed["transactions"])
    seeded_transaction_count = len(transactions)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        for benchmark in benchmarks:
            observation = runner.reset(benchmark)
            ranked = [int(index) for index in runner.get_model_action(observation)]
            donor_indices = ranked[: min(args.donor_limit, len(ranked))]
            donors = [coreset[index] for index in donor_indices]
            seen_parent_actions: set[tuple[int, ...]] = set()
            for parent_rank in parent_ranks:
                if parent_rank >= len(ranked):
                    raise ValueError(f"Parent rank {parent_rank} is unavailable for {benchmark}")
                parent_index = ranked[parent_rank]
                parent_actions = coreset[parent_index]
                if tuple(parent_actions) in seen_parent_actions:
                    continue
                seen_parent_actions.add(tuple(parent_actions))
                parent_first = evaluate_fixed_policy(benchmark, parent_actions)
                parent_second = evaluate_fixed_policy(benchmark, parent_actions)
                parent_certified = certified(parent_first) and certified(parent_second)
                parent_stable = stable_reward(parent_first, parent_second, args.repeat_tolerance)
                # A parent with a repeatable, certified non-negative reward is
                # trusted for blame attribution. A weak parent can still yield
                # exploratory data, but disagreement is not charged as blame.
                parent_trusted = bool(parent_certified and parent_stable and parent_first["total_reward"] >= 0.0)
                candidates = generate_candidates(
                    benchmark=benchmark,
                    parent=parent_actions,
                    donors=donors,
                    candidate_budget=args.candidate_budget,
                    max_actions=args.max_actions,
                    donor_limit=args.donor_limit,
                )
                for edit in candidates:
                    child_actions = apply_edit(parent_actions, donors, edit)
                    child = evaluate_fixed_policy(benchmark, child_actions)
                    child_certified = certified(child)
                    delta = float(child["total_reward"] - parent_first["total_reward"])
                    blame = bool(parent_trusted and child_certified and delta < -args.blame_margin)
                    if not child_certified:
                        label = "unsafe_rejected"
                    elif delta > args.improvement_epsilon:
                        label = "certified_improvement"
                    elif blame:
                        label = "certified_regression_blamed"
                    elif delta < -args.blame_margin:
                        label = "certified_regression_untrusted_parent"
                    else:
                        label = "certified_neutral"
                    transactions.append({
                    "protocol": "rlcompopt-blame-gated-transition-v1",
                    "cohort": args.cohort,
                    "benchmark": benchmark,
                    "controller": {
                        "parent_coreset_index": parent_index,
                        "parent_rank": parent_rank,
                        "parent_actions": parent_actions,
                        "ranked_donor_indices": donor_indices,
                        "autophase": [int(value) for value in observation],
                    },
                    "edit": {key: value for key, value in edit.items() if key != "child_actions"},
                    "child_actions": child_actions,
                    "parent": {
                        "first_certificate": parent_first,
                        "repeat_certificate": parent_second,
                        "certified": parent_certified,
                        "repeat_stable": parent_stable,
                        "trusted_for_blame": parent_trusted,
                    },
                    "child_certificate": child,
                    "outcome": {
                        "certified_child": child_certified,
                        "delta_reward": delta,
                        "label": label,
                        "blame": blame,
                        "hard_rejected": not child_certified,
                    },
                })
                report = {
                    "protocol": "rlcompopt-blame-gated-transition-ledger-v1",
                    "cohort": f"{args.cohort} only",
                    "held_out_touched": False,
                    "selection_touched": args.cohort == "selection",
                    "parameters": {
                        "candidate_budget": args.candidate_budget,
                        "parent_ranks": parent_ranks,
                        "donor_limit": args.donor_limit,
                        "max_actions": args.max_actions,
                        "improvement_epsilon": args.improvement_epsilon,
                        "blame_margin": args.blame_margin,
                        "repeat_tolerance": args.repeat_tolerance,
                        "blame_penalty": args.blame_penalty,
                    },
                    "transactions": transactions,
                    "seeded_transaction_count": seeded_transaction_count,
                    "new_transaction_count": len(transactions) - seeded_transaction_count,
                    **summarize(transactions, args.blame_penalty),
                }
                atomic_json(args.output, report)
    finally:
        runner.env.close()
        runner.model.connection.close()

    labels: dict[str, int] = defaultdict(int)
    for row in transactions:
        labels[row["outcome"]["label"]] += 1
    print(json.dumps({"transactions": len(transactions), "new_transactions": len(transactions) - seeded_transaction_count, "labels": dict(sorted(labels.items())), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
