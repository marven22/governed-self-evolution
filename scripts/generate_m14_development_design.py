"""Generate the preregistered, stratified grammar-action design for development."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from m14_update_grammar import UpdateSpec, baseline_specs, canonical_record


def design_specs() -> list[UpdateSpec]:
    # Covers each target family and low/balanced/high retained-task mixtures.
    return [
        UpdateSpec("policy", 0.25, 500, 1e-4, True, 0.0, False),
        UpdateSpec("policy", 0.25, 1_000, 3e-4, True, 0.1, False),
        UpdateSpec("policy", 0.50, 500, 3e-4, True, 0.01, False),
        UpdateSpec("policy", 0.50, 1_500, 1e-4, False, 1.0, False),
        UpdateSpec("policy", 0.75, 500, 1e-4, True, 0.1, False),
        UpdateSpec("policy", 0.75, 1_000, 3e-4, False, 0.1, True),
        UpdateSpec("world_model", 0.25, 500, 3e-4, False, 0.0, False),
        UpdateSpec("world_model", 0.75, 1_000, 3e-4, True, 0.0, True),
        UpdateSpec("policy_and_world_model", 0.25, 500, 1e-4, True, 0.01, True),
        UpdateSpec("policy_and_world_model", 0.50, 500, 1e-4, True, 0.1, True),
        UpdateSpec("policy_and_world_model", 0.75, 1_000, 1e-4, True, 1.0, True),
        UpdateSpec("policy_and_world_model", 0.50, 1_000, 3e-4, False, 0.0, False),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [canonical_record(spec, label=name) for name, spec in baseline_specs().items()]
    seen = {record["id"] for record in records}
    for spec in design_specs():
        if spec.identifier in seen:
            raise ValueError(f"duplicate action in development design: {spec.identifier}")
        records.append(canonical_record(spec))
        seen.add(spec.identifier)
    payload = {"grammar_version": "m14-update-grammar-v1", "sampling": {"method": "preregistered stratified coverage design", "version": 1}, "candidates": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"candidates": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
