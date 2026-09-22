"""Fail closed if a non-held-out CompilerGym program lacks a certificate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_compilergym_feasibility import evaluate_fixed_policy


def valid_certificate(record: dict) -> bool:
    validation = record["validation"]
    return bool(
        record["validation_available"]
        and validation["okay"]
        and validation["benchmark_semantics_validated"]
        and not validation["benchmark_semantics_validation_failed"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--cohorts",
        default="development,selection",
        help="Comma-separated non-held-out cohorts to audit.",
    )
    args = parser.parse_args()
    manifest = json.loads(args.split.read_text())
    cohorts = [name.strip() for name in args.cohorts.split(",") if name.strip()]
    if "held_out" in cohorts:
        raise ValueError("held_out programs must remain untouched before final evaluation")
    if not cohorts or any(name not in manifest for name in cohorts):
        raise ValueError(f"unknown or empty cohorts: {cohorts}")

    records = []
    for cohort in cohorts:
        for benchmark in manifest[cohort]:
            result = evaluate_fixed_policy(benchmark, actions=[])
            records.append(
                {
                    "cohort": cohort,
                    "benchmark": benchmark,
                    "certificate_ready": valid_certificate(result),
                    "validation": result["validation"],
                }
            )
    failures = [record for record in records if not record["certificate_ready"]]
    report = {
        "protocol": "compilergym-split-readiness-v1",
        "split": str(args.split),
        "cohorts": cohorts,
        "programs_audited": len(records),
        "held_out_touched": False,
        "passed": not failures,
        "failures": failures,
        "records": records,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("programs_audited", "held_out_touched", "passed")}))
    if failures:
        raise SystemExit("Split contains programs without a valid certificate; controller bank blocked.")


if __name__ == "__main__":
    main()
