#!/usr/bin/env python3
"""Apply the predeclared, paired final-promotion rule to two repair reports."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def selected_certificate_failed(row: dict) -> bool:
    selected = row.get("selected_outcome")
    return bool(row.get("eligible") and row.get("v2_action") == "edit" and not (selected or {}).get("certified"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--incumbent", type=Path, required=True)
    p.add_argument("--challenger", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    incumbent, challenger, protocol = load(a.incumbent), load(a.challenger), load(a.protocol)
    rule = protocol["promotion_rule"]
    if incumbent.get("cohort") != challenger.get("cohort") or incumbent.get("cohort") not in {"final_evaluation", "selection"}:
        raise ValueError("Both reports must use the same locked selection or final_evaluation cohort")
    if incumbent.get("strategy", {}).get("name") != protocol["incumbent"]:
        raise ValueError("Incumbent report does not identify the predeclared incumbent")
    expected_templates = protocol.get("damage_templates")
    if expected_templates and (incumbent.get("damage_templates") != expected_templates or challenger.get("damage_templates") != expected_templates):
        raise ValueError("Reports do not use the predeclared damage templates")
    def index(report: dict) -> dict:
        return {(r["benchmark"], r["damage"]["name"]): r for r in report["contexts"]}
    left, right = index(incumbent), index(challenger)
    if left.keys() != right.keys():
        raise ValueError("Reports do not contain identical paired contexts")
    paired = []
    for key in sorted(left):
        x, y = left[key], right[key]
        if bool(x["eligible"]) != bool(y["eligible"]):
            raise ValueError(f"Eligibility mismatch: {key}")
        if x["eligible"]:
            paired.append({"benchmark": key[0], "damage": key[1], "incumbent_gain": x["v2_repair_gain"], "challenger_gain": y["v2_repair_gain"], "difference": y["v2_repair_gain"] - x["v2_repair_gain"]})
    d = np.asarray([r["difference"] for r in paired], dtype=float)
    rng = np.random.default_rng(20260922)
    boots = np.asarray([rng.choice(d, size=len(d), replace=True).mean() for _ in range(rule["bootstrap_replicates"])]) if len(d) else np.asarray([0.])
    alpha = 1 - rule["confidence_level"]
    lower, upper = float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
    wins, losses, ties = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    cert_failures = sum(selected_certificate_failed(r) for r in challenger["contexts"])
    passes = {
        "enough_paired_contexts": len(paired) >= rule["minimum_paired_eligible_contexts"],
        "positive_bootstrap_lower_bound": lower > 0,
        "strict_majority_paired_wins": wins > losses,
        "no_selected_certificate_failures": cert_failures == 0,
    }
    report = {"protocol": protocol["protocol"], "scope": "one-time locked final promotion evaluation", "incumbent_strategy": incumbent["strategy"], "challenger_strategy": challenger["strategy"], "paired_eligible_contexts": len(paired), "mean_challenger_minus_incumbent_gain": float(d.mean()) if len(d) else 0., "bootstrap_95_interval": [lower, upper], "wins": wins, "losses": losses, "ties": ties, "challenger_selected_certificate_failures": cert_failures, "passes": passes, "promotion_decision": "promote_challenger" if all(passes.values()) else "retain_incumbent", "paired_results": paired}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("paired_eligible_contexts", "mean_challenger_minus_incumbent_gain", "bootstrap_95_interval", "wins", "losses", "ties", "promotion_decision")}, sort_keys=True))


if __name__ == "__main__":
    main()
