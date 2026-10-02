#!/usr/bin/env python3
"""Milestone-3 analysis: every update against both HOLD repetitions, with a noise floor.

Paired comparisons use exact two-sided sign tests on the cases whose task
success changed. A tool-permission run in which the policy blocked no call saw
model inputs identical to HOLD, so it is also reported as a HOLD-equivalent
run: its spread is an empirical estimate of run-to-run noise.
"""
from __future__ import annotations

import argparse
from math import comb
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentdojo.task_suite.load_suites import get_suite

from gse_agentdojo.runner import wilson_interval
from gse_agentdojo.transitions import diagnostics, load_run, paired_changes

UPDATES = (
    "ISOLATION_FOLLOW_EMBEDDED",
    "ISOLATION_UNTRUSTED_DATA",
    "ISOLATION_DELIMITED",
    "PERMISSION_NO_DESTRUCTIVE",
    "PERMISSION_KNOWN_RECIPIENTS",
)


def sign_test(better: int, worse: int) -> float:
    """Exact two-sided binomial sign test p-value."""
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(better, worse) + 1)) / 2**n
    return min(1.0, 2 * tail)


def counts(run) -> dict:
    cases = list(run["cases"].values())
    benign = [c for c in cases if c["kind"] == "benign"]
    security = [c for c in cases if c["kind"] == "security"]
    exposed = [c for c in security if c["injection_exposed"]]
    return {
        "benign_success": [sum(c["utility"] for c in benign), len(benign)],
        "attacked_success": [sum(c["utility"] for c in security), len(security)],
        "attacked_success_lenient": [sum(bool(c["utility_lenient"]) for c in security), len(security)],
        "exposed_injection_success": [sum(c["injection_task_success"] for c in exposed), len(exposed)],
        "attacker_contacted": [sum(bool(c["attacker_contacted"]) for c in security), len(security)],
        "policy_violations": sum(c["policy_violation"] for c in cases),
        "policy_blocked_calls": diagnostics(run)["policy_blocked_calls"],
        "cost_usd": run["result"]["metrics"]["estimated_cost_usd"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/agentdojo"))
    parser.add_argument("--prefix", default="workspace_m3_dev_v1__claude_haiku_4_5_cached__")
    parser.add_argument("--output", type=Path, default=Path("reports/agentdojo_m3_dev_v1.json"))
    args = parser.parse_args()

    suite = get_suite("v1.2.2", "workspace")
    run = {name: load_run(args.runs / f"{args.prefix}{name}", suite) for name in
           ("HOLD__rep0", "HOLD__rep1", *(f"{u}__rep0" for u in UPDATES))}
    holds = ("HOLD__rep0", "HOLD__rep1")
    report = {"protocol": "gse-agentdojo-m3-analysis-v1", "runs": {}, "comparisons": {}}
    for name, data in run.items():
        report["runs"][name] = counts(data)
    noise = paired_changes(run["HOLD__rep0"], run["HOLD__rep1"])["utility"]
    report["noise_floor"] = {"HOLD__rep0_vs_HOLD__rep1": {**noise, "sign_test_p": sign_test(**noise)}}
    for update in UPDATES:
        name = f"{update}__rep0"
        report["comparisons"][update] = {
            hold: {**(c := paired_changes(run[hold], run[name])["utility"]), "sign_test_p": round(sign_test(**c), 4)}
            for hold in holds
        }
        if update.startswith("PERMISSION") and report["runs"][name]["policy_blocked_calls"] == 0:
            report["noise_floor"][f"{update}_vs_HOLD__rep0 (inert policy)"] = report["comparisons"][update]["HOLD__rep0"]
    exposed = [report["runs"][n]["exposed_injection_success"] for n in report["runs"]]
    k, n = sum(e[0] for e in exposed), sum(e[1] for e in exposed)
    report["pooled_exposed_injection_success"] = {"successes": k, "n": n, "wilson95": wilson_interval(k, n)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"{'run':40}{'benign':>9}{'attacked':>10}{'lenient':>9}{'inj':>8}{'contact':>9}{'blocked':>9}{'cost':>8}")
    for name, c in report["runs"].items():
        print(f"{name:40}{c['benign_success'][0]:>6}/{c['benign_success'][1]:<2}{c['attacked_success'][0]:>7}/{c['attacked_success'][1]:<2}"
              f"{c['attacked_success_lenient'][0]:>6}/{c['attacked_success_lenient'][1]:<2}{c['exposed_injection_success'][0]:>5}/{c['exposed_injection_success'][1]:<2}"
              f"{c['attacker_contacted'][0]:>6}/{c['attacker_contacted'][1]:<2}{c['policy_blocked_calls']:>9}{c['cost_usd']:>8.3f}")
    print("\npaired task-success changes (better/worse, sign-test p):")
    for key, value in report["noise_floor"].items():
        print(f"  noise  {key:55} {value['better']}/{value['worse']}  p={value['sign_test_p']:.3f}")
    for update, by_hold in report["comparisons"].items():
        print(f"  update {update:28}" + "".join(f"  vs {h}: {v['better']}/{v['worse']} p={v['sign_test_p']:.3f}" for h, v in by_hold.items()))
    print(f"\npooled exposed injection success: {k}/{n}, Wilson 95% {report['pooled_exposed_injection_success']['wilson95']}")


if __name__ == "__main__":
    main()
