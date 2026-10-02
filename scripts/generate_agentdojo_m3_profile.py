#!/usr/bin/env python3
"""Freeze the Milestone-3 development profile from the committed split.

Every development user task runs once benign. Security pairs follow a balanced
design: three rounds, each a fresh seeded permutation of the 16 development
user tasks split between two development injection tasks. Every user task is
attacked 3 times by 3 different injection tasks; every injection task attacks
8 user tasks. No selection or held-out task can enter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gse_agentdojo import PINNED_AGENTDOJO_VERSION
from gse_agentdojo.attacks import SUPPORTED_ATTACKS
from gse_agentdojo.common import sha256_file
from gse_agentdojo.runner import PROFILE_PROTOCOL


def task_number(task_id: str) -> int:
    return int(task_id.rsplit("_", 1)[1])


def balanced_pairs(users: list[str], injections: list[str], seed: int) -> list[list[str]]:
    if len(injections) % 2 or len(users) % 2:
        raise ValueError("balanced design needs an even number of user and injection tasks")
    rng, half, pairs = random.Random(seed), len(users) // 2, []
    for first, second in zip(injections[::2], injections[1::2]):
        order = rng.sample(users, len(users))
        pairs += [[user, first] for user in order[:half]] + [[user, second] for user in order[half:]]
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=Path("configs/agentdojo_workspace_split_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("configs/agentdojo_workspace_m3_dev_v1.json"))
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()

    split = json.loads(args.split.read_text())
    users = sorted(split["user_tasks"]["cohorts"]["development"]["tasks"], key=task_number)
    injections = sorted(split["injection_tasks"]["cohorts"]["development"]["tasks"], key=task_number)
    profile = {
        "protocol": PROFILE_PROTOCOL,
        "name": "workspace_m3_dev_v1",
        "purpose": "Milestone-3 baseline matrix on the full development user cohort with a balanced attack design.",
        "agentdojo_version": PINNED_AGENTDOJO_VERSION,
        "benchmark_version": split["benchmark_version"],
        "suite": split["suite"],
        "cohort": "development",
        "split": {"path": str(args.split), "sha256": sha256_file(args.split)},
        "attack": SUPPORTED_ATTACKS[0],
        "selection_seed": args.seed,
        "benign_user_tasks": users,
        "security_pairs": balanced_pairs(users, injections, args.seed),
    }
    args.output.write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps({"profile": str(args.output), "benign": len(users), "security_pairs": len(profile["security_pairs"])}))


if __name__ == "__main__":
    main()
