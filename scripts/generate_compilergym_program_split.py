"""Freeze disjoint CompilerGym program splits before controller evolution."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--development", type=int, default=8)
    parser.add_argument("--selection", type=int, default=5)
    parser.add_argument("--held-out", type=int, default=5)
    args = parser.parse_args()

    import compiler_gym
    from compiler_gym.envs.llvm.datasets import cbench

    # Explicit spaces avoid Gym 0.26's passive checker treating the archived
    # environment's default ``None`` observation space as an API error.
    env = compiler_gym.make(
        "llvm-v0",
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
    )
    try:
        # Never enumerate ``env.datasets`` globally: CompilerGym registers
        # datasets with extremely large or unbounded benchmark collections.
        # The protocol is explicitly limited to cBench.
        uris = sorted(str(uri) for uri in env.datasets["benchmark://cbench-v1"].benchmark_uris())
    finally:
        env.close()
    # Eligibility is derived from CompilerGym's static validator registry; it
    # does not execute any program, including held-out programs.
    eligible = sorted(uri for uri in uris if uri in cbench.VALIDATORS)
    required = args.development + args.selection + args.held_out
    if len(eligible) != required:
        raise ValueError(
            f"cbench-v1 yielded {len(eligible)} certificate-eligible programs but split requires exactly {required}; "
            "change the protocol explicitly rather than silently dropping programs"
        )
    shuffled = eligible.copy()
    random.Random(args.seed).shuffle(shuffled)
    development = sorted(shuffled[: args.development])
    selection = sorted(shuffled[args.development : args.development + args.selection])
    held_out = sorted(shuffled[args.development + args.selection :])
    all_assigned = development + selection + held_out
    if len(set(all_assigned)) != len(all_assigned) or set(all_assigned) != set(eligible):
        raise AssertionError("split is not an exact disjoint partition")
    manifest = {
        "protocol": "compilergym-program-split-v1",
        "dataset": "benchmark://cbench-v1",
        "eligibility": "CompilerGym static cBench validator registry",
        "seed": args.seed,
        "counts": {
            "development": len(development),
            "selection": len(selection),
            "held_out": len(held_out),
        },
        "development": development,
        "selection": selection,
        "held_out": held_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"counts": manifest["counts"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
