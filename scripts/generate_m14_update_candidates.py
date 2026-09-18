"""Generate a reproducible, coverage-oriented candidate set from the M14 grammar."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from m14_update_grammar import UpdateSpec, baseline_specs, canonical_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-random", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_random < 0:
        raise ValueError("--num-random must be non-negative")

    rng = random.Random(args.seed)
    records = [canonical_record(spec, label=name) for name, spec in baseline_specs().items()]
    seen = {record["id"] for record in records}
    targets = ("policy", "world_model", "policy_and_world_model")
    reach_fractions = (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9)
    steps = (250, 500, 1_000, 1_500)
    learning_rates = (1e-4, 3e-4, 1e-3)
    penalties = (0.0, 0.01, 0.1, 1.0)

    while len(records) < len(baseline_specs()) + args.num_random:
        target = rng.choice(targets)
        spec = UpdateSpec(
            target=target,
            reach_fraction=rng.choice(reach_fractions),
            gradient_steps=rng.choice(steps),
            learning_rate=rng.choice(learning_rates),
            freeze_encoder=rng.choice((False, True)),
            prior_policy_l2=0.0 if target == "world_model" else rng.choice(penalties),
            protect_policy_during_model_update=rng.choice((False, True)),
        )
        if spec.identifier not in seen:
            records.append(canonical_record(spec))
            seen.add(spec.identifier)

    payload = {
        "grammar_version": "m14-update-grammar-v1",
        "sampling": {"method": "seeded categorical coverage sample", "seed": args.seed},
        "candidates": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"candidates": len(records), "output": str(args.output)}))


if __name__ == "__main__":
    main()
