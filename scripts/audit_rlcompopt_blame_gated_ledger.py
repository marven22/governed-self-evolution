#!/usr/bin/env python3
"""Fail-closed quality audit for the blame-gated transaction ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EPSILON = 0.01
REQUIRED_TOP_LEVEL = {"protocol", "cohort", "benchmark", "controller", "edit", "child_actions", "parent", "child_certificate", "outcome"}


def certificate_is_valid(certificate: dict[str, Any]) -> bool:
    validation = certificate.get("validation", {})
    return bool(
        certificate.get("validation_available")
        and validation.get("okay")
        and validation.get("benchmark_semantics_validated")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-programs", type=int, default=8)
    parser.add_argument("--expected-transactions", type=int, default=672)
    parser.add_argument("--expected-cohort", choices=("development", "selection"), default="development")
    args = parser.parse_args()

    raw = args.ledger.read_bytes()
    ledger: dict[str, Any] = json.loads(raw)
    rows: list[dict[str, Any]] = ledger["transactions"]
    failures: list[str] = []
    warnings: list[str] = []

    if ledger.get("cohort") != f"{args.expected_cohort} only":
        failures.append(f"Ledger cohort is not {args.expected_cohort} only")
    if ledger.get("held_out_touched"):
        failures.append("Ledger claims the protected held-out cohort was touched")
    if args.expected_cohort == "development" and ledger.get("selection_touched"):
        failures.append("Development ledger claims the selection cohort was touched")
    if len(rows) != args.expected_transactions:
        failures.append(f"Expected {args.expected_transactions} rows; found {len(rows)}")

    missing_fields = [index for index, row in enumerate(rows) if not REQUIRED_TOP_LEVEL.issubset(row)]
    if missing_fields:
        failures.append(f"Missing required top-level fields in {len(missing_fields)} rows")

    programs = Counter(row["benchmark"] for row in rows)
    if len(programs) != args.expected_programs:
        failures.append(f"Expected {args.expected_programs} programs; found {len(programs)}")
    expected_per_program = args.expected_transactions // max(1, args.expected_programs)
    if any(count != expected_per_program for count in programs.values()):
        failures.append(f"Each {args.expected_cohort} program must contribute exactly {expected_per_program} rows")

    ranks = Counter((row["benchmark"], row["controller"].get("parent_rank", 0)) for row in rows)
    expected_rank_counts = {0: 12, 1: 24, 2: 24, 3: 24}
    for benchmark in programs:
        for rank, expected in expected_rank_counts.items():
            if ranks[(benchmark, rank)] != expected:
                failures.append(f"{benchmark} rank {rank}: expected {expected}, found {ranks[(benchmark, rank)]}")

    duplicate_keys: dict[tuple[Any, ...], int] = defaultdict(int)
    child_keys: dict[tuple[str, tuple[int, ...]], set[tuple[int, ...]]] = defaultdict(set)
    label_counts: Counter[str] = Counter()
    op_counts: Counter[str] = Counter()
    invalid_label_checks = 0
    all_parent_trusted = 0
    all_child_certified = 0
    leaked_feature_fields: set[str] = set()

    forbidden_feature_names = {"child_certificate", "outcome", "delta_reward", "label", "blame", "hard_rejected"}
    for row in rows:
        controller = row["controller"]
        edit = row["edit"]
        parent = row["parent"]
        outcome = row["outcome"]
        key = (
            row["benchmark"],
            controller.get("parent_rank", 0),
            controller["parent_coreset_index"],
            tuple(row["child_actions"]),
        )
        duplicate_keys[key] += 1
        child_keys[(row["benchmark"], tuple(controller["parent_actions"]))].add(tuple(row["child_actions"]))
        label = outcome["label"]
        label_counts[label] += 1
        op_counts[edit["op"]] += 1
        child_valid = certificate_is_valid(row["child_certificate"])
        if child_valid != bool(outcome["certified_child"]):
            invalid_label_checks += 1
        delta = float(outcome["delta_reward"])
        trusted = bool(parent["trusted_for_blame"])
        if trusted:
            all_parent_trusted += 1
        if child_valid:
            all_child_certified += 1
        if label == "unsafe_rejected":
            valid = (not child_valid) and bool(outcome["hard_rejected"]) and not bool(outcome["blame"])
        elif label == "certified_improvement":
            valid = child_valid and delta > EPSILON and not bool(outcome["hard_rejected"])
        elif label == "certified_regression_blamed":
            valid = child_valid and trusted and delta < -EPSILON and bool(outcome["blame"])
        elif label == "certified_neutral":
            valid = child_valid and not bool(outcome["hard_rejected"])
        elif label == "certified_regression_untrusted_parent":
            valid = child_valid and not trusted and delta < -EPSILON
        else:
            valid = False
        if not valid:
            invalid_label_checks += 1
        for feature_container in (controller, edit):
            leaked_feature_fields.update(set(feature_container) & forbidden_feature_names)
        if tuple(row["child_actions"]) == tuple(controller["parent_actions"]):
            failures.append("A child recipe duplicates its parent recipe")

    duplicate_count = sum(count - 1 for count in duplicate_keys.values() if count > 1)
    if duplicate_count:
        failures.append(f"Found {duplicate_count} duplicate transition keys")
    if invalid_label_checks:
        failures.append(f"Found {invalid_label_checks} certificate/label consistency violations")
    if leaked_feature_fields:
        failures.append(f"Post-outcome fields found in prospective feature containers: {sorted(leaked_feature_fields)}")
    if label_counts["certified_improvement"] < 20:
        failures.append("Too few certified improvements for an initial gain model")
    if label_counts["certified_regression_blamed"] < 20:
        failures.append("Too few blame examples for an initial blame model")
    if label_counts["unsafe_rejected"] < 20:
        warnings.append("Only two unsafe outcomes: do not train a learned semantic-safety head; keep certification hard-gated")
    if len(op_counts) != 4:
        failures.append("One or more grammar operator families are absent")

    report = {
        "protocol": "rlcompopt-blame-gated-ledger-audit-v1",
        "ledger": str(args.ledger),
        "ledger_sha256": hashlib.sha256(raw).hexdigest(),
        "passed_for_gain_and_blame_governor": not failures,
        "not_approved_for_learned_semantic_safety_model": label_counts["unsafe_rejected"] < 20,
        "failures": failures,
        "warnings": warnings,
        "metrics": {
            "transactions": len(rows),
            "programs": len(programs),
            "program_row_counts": dict(sorted(programs.items())),
            "label_counts": dict(sorted(label_counts.items())),
            "operator_counts": dict(sorted(op_counts.items())),
            "trusted_parents": all_parent_trusted,
            "certified_children": all_child_certified,
            "duplicate_transition_rows": duplicate_count,
            "distinct_children_per_parent_context_min": min(map(len, child_keys.values())),
            "distinct_children_per_parent_context_max": max(map(len, child_keys.values())),
        },
        "approved_feature_schema": {
            "allowed": ["controller.autophase", "controller.parent_rank", "controller.parent_actions", "controller.parent_coreset_index", "controller.ranked_donor_indices", "edit.*"],
            "forbidden": ["child_certificate", "outcome.*", "child_actions"],
            "note": "Child actions are the label-side realization of the edit and must not enter a predictor except through the edit AST.",
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed_for_gain_and_blame_governor"], "failures": failures, "warnings": warnings, "metrics": report["metrics"]}, sort_keys=True))
    if failures:
        raise SystemExit("Ledger quality audit failed")


if __name__ == "__main__":
    main()
