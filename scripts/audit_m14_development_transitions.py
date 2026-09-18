"""Audit the exact-transition development collection before model fitting."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

from m14_update_grammar import UpdateSpec


TASKS = ("mw-reach", "mw-pick-place")


def add_error(errors: list[str], message: str) -> None:
    errors.append(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    split = json.loads(args.split.read_text())
    candidate_payload = json.loads(args.candidates.read_text())
    expected_controller_ids = [record["id"] for record in split["development"]]
    held_out_ids = [record["id"] for record in split["held_out"]]
    expected_candidate_ids = {record["id"] for record in candidate_payload["candidates"]}
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    pre_states: dict[str, dict[str, float]] = {}

    for controller_id in expected_controller_ids:
        run_dir = args.root / controller_id
        completion = run_dir / "completion.json"
        transition_path = run_dir / "transitions.json"
        if not completion.exists() or not transition_path.exists():
            add_error(errors, f"{controller_id}: missing completion or transitions file")
            continue
        completion_payload = json.loads(completion.read_text())
        controller_rows = json.loads(transition_path.read_text())
        if not completion_payload.get("complete"):
            add_error(errors, f"{controller_id}: completion flag is false")
        if completion_payload.get("transitions") != len(expected_candidate_ids):
            add_error(errors, f"{controller_id}: completion count differs from candidate design")
        if len(controller_rows) != len(expected_candidate_ids):
            add_error(errors, f"{controller_id}: {len(controller_rows)} rows; expected {len(expected_candidate_ids)}")
        ids = []
        controller_pre = set()
        for row in controller_rows:
            context = row.get("context", {})
            candidate_id = context.get("candidate_id")
            ids.append(candidate_id)
            if context.get("seed") is None:
                add_error(errors, f"{controller_id}/{candidate_id}: missing execution seed")
            if row.get("domain") != "metaworld-mt2":
                add_error(errors, f"{controller_id}/{candidate_id}: unexpected domain")
            if row.get("pre_capability_is_exact") is not True:
                add_error(errors, f"{controller_id}/{candidate_id}: pre-capability is not exact")
            try:
                spec = UpdateSpec.from_dict(row["update"])
                if candidate_id != spec.identifier:
                    add_error(errors, f"{controller_id}/{candidate_id}: candidate id disagrees with canonical update")
            except (KeyError, TypeError, ValueError) as exc:
                add_error(errors, f"{controller_id}/{candidate_id}: invalid update specification ({exc})")
            demand = row.get("demand", {})
            pre, post = row.get("pre_capability", {}), row.get("post_capability", {})
            for task in TASKS:
                for label, capability in (("pre", pre), ("post", post)):
                    value = capability.get(task)
                    if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                        add_error(errors, f"{controller_id}/{candidate_id}: invalid {label} {task}")
                if not isinstance(demand.get(task), (int, float)):
                    add_error(errors, f"{controller_id}/{candidate_id}: invalid demand {task}")
            if abs(sum(demand.get(task, 0.0) for task in TASKS) - 1.0) > 1e-9:
                add_error(errors, f"{controller_id}/{candidate_id}: demand does not sum to one")
            expected_utility = sum(float(demand.get(task, 0.0)) * float(post.get(task, 0.0)) for task in TASKS)
            if abs(float(row.get("utility", float("nan"))) - expected_utility) > 1e-9:
                add_error(errors, f"{controller_id}/{candidate_id}: utility arithmetic mismatch")
            minimum = row.get("constraints", {}).get("min_reach_retention")
            if not isinstance(minimum, (int, float)):
                add_error(errors, f"{controller_id}/{candidate_id}: missing reach-retention constraint")
            elif bool(row.get("feasible")) != (post.get("mw-reach") >= minimum):
                add_error(errors, f"{controller_id}/{candidate_id}: feasibility label mismatch")
            controller_pre.add(tuple((task, pre.get(task)) for task in TASKS))
        if set(ids) != expected_candidate_ids:
            add_error(errors, f"{controller_id}: candidate set does not match frozen design")
        if len(ids) != len(set(ids)):
            add_error(errors, f"{controller_id}: duplicate candidate rows")
        if len(controller_pre) != 1:
            add_error(errors, f"{controller_id}: inconsistent pre-capability state across candidates")
        elif controller_pre:
            pre_states[controller_id] = dict(next(iter(controller_pre)))
        rows.extend(controller_rows)

    keys = [(row.get("context", {}).get("checkpoint"), row.get("context", {}).get("candidate_id")) for row in rows]
    if len(keys) != len(set(keys)):
        add_error(errors, "duplicate checkpoint/candidate transition keys across development collection")
    for held_out_id in held_out_ids:
        if (args.root / held_out_id).exists():
            add_error(errors, f"held-out controller {held_out_id} has development-run artifacts")

    target_counts = Counter(row["update"]["target"] for row in rows)
    mixture_counts = Counter(row["update"]["task_data"]["reach_fraction"] for row in rows)
    action_counts = Counter(row["context"]["candidate_id"] for row in rows)
    summary = {
        "audit_version": "m14-development-audit-v1",
        "grain": "one exact pre/post capability transition for one base controller, one canonical update action, one execution seed, and one fixed demand vector",
        "development_controllers_expected": expected_controller_ids,
        "held_out_controllers_expected": held_out_ids,
        "expected_candidates_per_controller": len(expected_candidate_ids),
        "rows": len(rows),
        "exact_pre_rows": sum(row.get("pre_capability_is_exact") is True for row in rows),
        "controller_pre_capabilities": pre_states,
        "target_family_counts": dict(sorted(target_counts.items())),
        "reach_fraction_counts": {str(key): value for key, value in sorted(mixture_counts.items())},
        "candidate_replication_counts": dict(sorted(action_counts.items())),
        "errors": errors,
        "passed": not errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"passed": summary["passed"], "rows": len(rows), "errors": len(errors)}))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
