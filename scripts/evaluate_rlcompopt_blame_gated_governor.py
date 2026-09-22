#!/usr/bin/env python3
"""Evaluate a frozen blame-gated governor on an untouched selection ledger.

The evaluator never fits or updates a model. It scores each candidate from its
pre-edit controller context and edit AST, then uses the sealed outcome ledger
only to measure the decision that was made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from train_rlcompopt_blame_gated_governor import features, score


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def choose(
    indices: list[int],
    deltas: np.ndarray,
    gain: np.ndarray,
    risk: np.ndarray,
    uncertainty: np.ndarray,
    use_blame: bool,
) -> tuple[int | None, float, float]:
    """Return selected row (or HOLD), predicted score, and observed gain."""
    utility = gain[indices] - uncertainty[indices]
    if use_blame:
        utility = utility - risk[indices]
    position = int(np.argmax(utility))
    chosen = indices[position]
    if utility[position] <= 0.0:
        return None, float(utility[position]), 0.0
    return chosen, float(utility[position]), max(0.0, float(deltas[chosen]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--selection-ledger", type=Path, required=True)
    parser.add_argument("--selection-audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.selection_audit.read_text(encoding="utf-8"))
    if not audit.get("passed_for_gain_and_blame_governor"):
        raise SystemExit("Refusing to evaluate a failed selection-ledger audit")
    raw_ledger = args.selection_ledger.read_bytes()
    if hashlib.sha256(raw_ledger).hexdigest() != audit.get("ledger_sha256"):
        raise SystemExit("Selection ledger changed after audit; re-run the audit")
    ledger = json.loads(raw_ledger)
    if ledger.get("cohort") != "selection only" or ledger.get("held_out_touched"):
        raise SystemExit("Evaluation requires an untouched selection-only ledger")

    model_bytes = args.model.read_bytes()
    payload: dict[str, Any] = pickle.loads(model_bytes)
    if payload.get("protocol") != "rlcompopt-blame-gated-governor-v1":
        raise SystemExit("Unexpected governor protocol")
    rows = [row for row in ledger["transactions"] if row["outcome"]["certified_child"]]
    if len(rows) != len(ledger["transactions"]):
        raise SystemExit("Selection evaluation only supports fully certificate-backed candidate groups")
    x = np.vstack([features(row) for row in rows])
    if x.shape[1] != payload["feature_dim"]:
        raise SystemExit("Feature contract does not match frozen governor")
    gain, risk, uncertainty = score(
        payload["gain_model"], payload["blame_model"], payload["bootstrap"], x
    )
    delta = np.asarray([row["outcome"]["delta_reward"] for row in rows], dtype=float)

    contexts: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        controller = row["controller"]
        contexts[(row["benchmark"], controller["parent_rank"], controller["parent_coreset_index"])].append(index)

    governor, ablation, random_expected, oracle, hold = [], [], [], [], []
    governor_rows: list[dict[str, Any]] = []
    for context, indices in sorted(contexts.items()):
        selected, predicted_utility, observed = choose(indices, delta, gain, risk, uncertainty, use_blame=True)
        ablated, ablated_utility, ablated_observed = choose(indices, delta, gain, risk, uncertainty, use_blame=False)
        governor.append(observed)
        ablation.append(ablated_observed)
        hold.append(0.0)
        random_expected.append(mean([max(0.0, float(delta[index])) for index in indices]))
        oracle.append(max(0.0, float(np.max(delta[indices]))))
        row = {
            "benchmark": context[0], "parent_rank": context[1], "parent_coreset_index": context[2],
            "candidate_count": len(indices), "governor_action": "hold" if selected is None else "edit",
            "governor_predicted_utility": predicted_utility, "governor_observed_gain": observed,
            "no_blame_action": "hold" if ablated is None else "edit",
            "no_blame_predicted_utility": ablated_utility, "no_blame_observed_gain": ablated_observed,
            "random_expected_gain": random_expected[-1], "oracle_gain": oracle[-1],
        }
        if selected is not None:
            row["governor_edit_program_id"] = rows[selected]["edit"]["edit_program_id"]
            row["governor_outcome_label"] = rows[selected]["outcome"]["label"]
        governor_rows.append(row)

    report = {
        "protocol": "rlcompopt-blame-gated-governor-selection-evaluation-v1",
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "selection_ledger_sha256": hashlib.sha256(raw_ledger).hexdigest(),
        "evaluation_rule": "argmax(predicted_gain - bootstrap_uncertainty - predicted_blame); HOLD when best score <= 0",
        "contexts": len(contexts),
        "mean_certified_gain": {
            "governor": mean(governor),
            "hold": mean(hold),
            "random_expected": mean(random_expected),
            "no_blame_ablation": mean(ablation),
            "oracle": mean(oracle),
        },
        "governor_over_hold": mean(governor) - mean(hold),
        "governor_over_random_expected": mean(governor) - mean(random_expected),
        "governor_regret_to_oracle": mean(oracle) - mean(governor),
        "governor_edit_rate": sum(row["governor_action"] == "edit" for row in governor_rows) / len(governor_rows),
        "context_decisions": governor_rows,
        "limitations": [
            "Selection is used once for threshold and ablation evidence, not model fitting.",
            "This measures one-step recipe improvement, not training a new neural controller checkpoint.",
            "A fresh final cohort is still required for a generalization claim.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("contexts", "mean_certified_gain", "governor_over_hold", "governor_over_random_expected", "governor_regret_to_oracle", "governor_edit_rate")}, sort_keys=True))


if __name__ == "__main__":
    main()
