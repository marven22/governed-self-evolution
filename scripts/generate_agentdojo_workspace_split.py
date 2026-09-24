#!/usr/bin/env python3
"""Freeze the family-blocked Workspace split and the Milestone-1 smoke profile.

Run once, before any model evaluation, and commit both outputs. The profile
records the split's SHA-256 so the runner refuses a split edited afterwards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agentdojo.task_suite.load_suites import get_suite

from gse_agentdojo import PINNED_AGENTDOJO_VERSION
from gse_agentdojo.attacks import SUPPORTED_ATTACKS
from gse_agentdojo.common import sha256_file
from gse_agentdojo.runner import PROFILE_PROTOCOL
from gse_agentdojo.split import build_workspace_split, validate_split


def select_smoke_cases(split: dict, *, seed: int, benign: int, security: int) -> tuple[list[str], list[list[str]]]:
    """Seeded development-only subset; security pairs rotate over every injection task."""
    rng = random.Random(seed)
    users = list(split["user_tasks"]["cohorts"]["development"]["tasks"])
    injections = list(split["injection_tasks"]["cohorts"]["development"]["tasks"])
    benign_tasks = sorted(rng.sample(users, benign), key=lambda t: int(t.rsplit("_", 1)[1]))
    per_injection = {inj: rng.sample(users, len(users)) for inj in injections}
    pairs: list[list[str]] = []
    round_index = 0
    while len(pairs) < security:
        for inj in injections:
            if len(pairs) < security:
                pairs.append([per_injection[inj][round_index], inj])
        round_index += 1
    return benign_tasks, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--split-output", type=Path, default=Path("configs/agentdojo_workspace_split_v1.json"))
    parser.add_argument("--profile-output", type=Path, default=Path("configs/agentdojo_workspace_m1_smoke_v1.json"))
    parser.add_argument("--benign", type=int, default=8)
    parser.add_argument("--security", type=int, default=16)
    args = parser.parse_args()

    suite = get_suite(args.benchmark_version, "workspace")
    split = build_workspace_split(benchmark_version=args.benchmark_version, seed=args.seed)
    validate_split(split, list(suite.user_tasks), list(suite.injection_tasks))
    args.split_output.write_text(json.dumps(split, indent=2) + "\n")

    benign, pairs = select_smoke_cases(split, seed=args.seed, benign=args.benign, security=args.security)
    profile = {
        "protocol": PROFILE_PROTOCOL,
        "name": "workspace_m1_smoke_v1",
        "purpose": "Milestone-1 adapter smoke run on a fixed development-only subset. Not for model selection.",
        "agentdojo_version": PINNED_AGENTDOJO_VERSION,
        "benchmark_version": args.benchmark_version,
        "suite": "workspace",
        "cohort": "development",
        "split": {"path": str(args.split_output), "sha256": sha256_file(args.split_output)},
        "attack": SUPPORTED_ATTACKS[0],
        "selection_seed": args.seed,
        "benign_user_tasks": benign,
        "security_pairs": pairs,
    }
    args.profile_output.write_text(json.dumps(profile, indent=2) + "\n")
    sizes = {
        kind: {c: len(split[kind]["cohorts"][c]["tasks"]) for c in split[kind]["cohorts"]}
        for kind in ("user_tasks", "injection_tasks")
    }
    print(json.dumps({"split": str(args.split_output), "profile": str(args.profile_output), "sizes": sizes}))


if __name__ == "__main__":
    main()
