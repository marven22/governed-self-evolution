#!/usr/bin/env python3
"""Audit the public RLCompOpt release bundle without mutating it.

RLCompOpt is intentionally kept outside this repository: it is a frozen inner
controller baseline. This probe establishes that the released offline evidence
is complete and records the runtime capability needed before training it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path


REQUIRED = (
    "all10k-train-medium-all10k-autophase.db",
    "all10k-val-medium-all10k-autophase.db",
    "all_ssl_vocab.db",
    "trajdataset_all10k-train-medium-all10k.json",
    "trajdataset_all10k-val-medium-all10k.json",
    "benchmarkdataset_all-test.json",
)


def inspect_database(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(path)
    try:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        counts = {
            table: connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            for table in tables
        }
    finally:
        connection.close()
    return {"bytes": path.stat().st_size, "tables": tables, "row_counts": counts}


def inspect_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        sample = value[0] if value else None
        count = len(value)
    elif isinstance(value, dict):
        sample = next(iter(value.items()), None)
        count = len(value)
    else:
        sample, count = None, None
    return {
        "bytes": path.stat().st_size,
        "type": type(value).__name__,
        "records": count,
        "sample_type": type(sample).__name__ if sample is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlcompopt-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.rlcompopt_root.resolve()
    data = root / "data"
    missing = [name for name in REQUIRED if not (data / name).is_file()]

    report: dict[str, object] = {
        "rlcompopt_root": str(root),
        "data_root": str(data),
        "bundle_complete": not missing,
        "missing_required_files": missing,
        "python_packages": {
            name: importlib.util.find_spec(name) is not None
            for name in ("torch", "torch_geometric", "rlcompopt")
        },
    }
    if not missing:
        report["databases"] = {
            name: inspect_database(data / name)
            for name in REQUIRED
            if name.endswith(".db") and name != "all_ssl_vocab.db"
        }
        report["json"] = {
            name: inspect_json(data / name) for name in REQUIRED if name.endswith(".json")
        }
        try:
            import torch

            report["torch"] = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
            }
        except ImportError:
            report["torch"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if missing:
        raise SystemExit("RLCompOpt bundle is incomplete")


if __name__ == "__main__":
    main()
