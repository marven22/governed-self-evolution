#!/usr/bin/env python3
"""Locked Csmith repair benchmark with reference-output semantic certificates.

Csmith programs take no input and print a final checksum. A child is certified
only when it terminates and emits exactly the checksum obtained from two
independent executions of the unedited program for that seed.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from rlcompopt_edit_grammar import apply_edit, generate_candidates
from run_compilergym_feasibility import unwrap_reset, unwrap_step
from train_rlcompopt_pairwise_governor_v2 import candidate_features

EPS = 0.01
TIMEOUT_SECONDS = 3


def write(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run_bitcode(env, path: Path) -> dict:
    from compiler_gym.third_party import llvm

    env.write_bitcode(path)
    try:
        result = subprocess.run(
            [str(llvm.lli_path()), str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
        return {"returncode": result.returncode, "output": result.stdout.decode("utf-8", "replace")}
    except subprocess.TimeoutExpired:
        return {"returncode": None, "output": "", "timeout": True}


def reference_output(benchmark: str, cache: dict[str, str]) -> str:
    if benchmark in cache:
        return cache[benchmark]
    import compiler_gym

    outputs = []
    with tempfile.TemporaryDirectory() as directory:
        for repeat in range(2):
            env = compiler_gym.make("llvm-v0", benchmark=benchmark, observation_space="Autophase", reward_space="IrInstructionCountOz")
            try:
                unwrap_reset(env.reset(benchmark=benchmark))
                result = run_bitcode(env, Path(directory) / f"reference-{repeat}.bc")
                if result.get("returncode") != 0 or not result.get("output"):
                    raise RuntimeError(f"Reference execution failed for {benchmark}: {result}")
                outputs.append(result["output"])
            finally:
                env.close()
    if outputs[0] != outputs[1]:
        raise RuntimeError(f"Reference output is not deterministic for {benchmark}")
    cache[benchmark] = outputs[0]
    return outputs[0]


def evaluate(benchmark: str, actions: list[int], references: dict[str, str]) -> dict:
    import compiler_gym

    expected = reference_output(benchmark, references)
    env = compiler_gym.make("llvm-v0", benchmark=benchmark, observation_space="Autophase", reward_space="IrInstructionCountOz")
    try:
        initial = unwrap_reset(env.reset(benchmark=benchmark))
        reward = 0.0
        for action in actions:
            _, step_reward, done, _ = unwrap_step(env.step(action))
            reward += step_reward
            if done:
                break
        with tempfile.TemporaryDirectory() as directory:
            result = run_bitcode(env, Path(directory) / "child.bc")
        certified = bool(result.get("returncode") == 0 and result.get("output") == expected)
        return {
            "benchmark": benchmark,
            "initial_observation": [int(x) for x in initial],
            "total_reward": reward,
            "validation_available": True,
            "validation": {
                "okay": certified,
                "benchmark_semantics_validated": certified,
                "profile": "csmith-deterministic-checksum-equivalence-v1",
                "expected_output": expected,
                "observed_output": result.get("output", ""),
                "returncode": result.get("returncode"),
            },
        }
    finally:
        env.close()


def certified(result: dict) -> bool:
    return bool(result["validation_available"] and result["validation"]["okay"] and result["validation"]["benchmark_semantics_validated"])


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("model_db", "trajectory_data", "vocab_db", "split", "model", "output"):
        p.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    p.add_argument("--cohort", default="final_evaluation")
    p.add_argument("--max-programs", type=int, default=None)
    p.add_argument("--candidate-budget", type=int, default=24)
    p.add_argument("--donor-limit", type=int, default=10)
    p.add_argument("--damage-templates", default="early,middle,quarter3,late")
    p.add_argument("--risk-weight", type=float, default=1.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--strategy-name", default="frozen_v2")
    a = p.parse_args()
    split = json.loads(a.split.read_text())
    benchmarks = split.get(a.cohort, [])
    if not benchmarks:
        raise ValueError(f"Unknown or empty cohort: {a.cohort}")
    if a.max_programs:
        benchmarks = benchmarks[: a.max_programs]
    damage_positions = {"early": lambda n: n // 4, "middle": lambda n: n // 2, "quarter3": lambda n: (3 * n) // 4, "late": lambda n: n - 1}
    requested = [x.strip() for x in a.damage_templates.split(",") if x.strip()]
    if not requested or any(x not in damage_positions for x in requested):
        raise ValueError("damage-templates must use early,middle,quarter3,late")
    model = pickle.loads(a.model.read_bytes())
    from rlcompopt.model_testing import Environment

    runner = Environment(str(a.model_db), None, 0, str(a.vocab_db), max_step=100, benchmarks=[], train_dataset_path=str(a.trajectory_data), sampling=False)
    coreset = [[int(x) for x in sequence] for sequence in runner.actionseqs]
    references: dict[str, str] = {}
    contexts = []
    a.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for benchmark in benchmarks:
            observation = runner.reset(benchmark)
            ordered = [int(x) for x in runner.get_model_action(observation)]
            donors = [coreset[index] for index in ordered[: a.donor_limit]]
            original = coreset[ordered[0]]
            original_result = evaluate(benchmark, original, references)
            seen_positions = set()
            for template in requested:
                position = damage_positions[template](len(original))
                if position in seen_positions:
                    continue
                seen_positions.add(position)
                damaged = original[:position] + original[position + 1 :]
                damaged_result = evaluate(benchmark, damaged, references)
                damage_delta = float(damaged_result["total_reward"] - original_result["total_reward"])
                record = {"benchmark": benchmark, "damage": {"name": f"delete_{template}", "position": position}, "original_reward": original_result["total_reward"], "damaged_reward": damaged_result["total_reward"], "damage_delta": damage_delta, "eligible": bool(certified(original_result) and certified(damaged_result) and damage_delta < -EPS)}
                if record["eligible"]:
                    edits = generate_candidates(benchmark=benchmark, parent=damaged, donors=donors, candidate_budget=a.candidate_budget, max_actions=64, donor_limit=a.donor_limit)
                    rows = [{"benchmark": benchmark, "controller": {"parent_coreset_index": ordered[0], "parent_rank": 0, "parent_actions": damaged, "ranked_donor_indices": ordered[: a.donor_limit], "autophase": [int(x) for x in observation]}, "edit": {k: v for k, v in edit.items() if k != "child_actions"}, "child_actions": apply_edit(damaged, donors, edit)} for edit in edits]
                    x = np.vstack([candidate_features(row) for row in rows])
                    pos = model["positive_model"].predict_proba(x)[:, 1]
                    risk = model["blame_model"].predict_proba(x)[:, 1]
                    rank = [model["pair_model"].predict_proba(np.vstack([x[i] - x[j] for j in range(len(rows)) if j != i]))[:, 1].mean() for i in range(len(rows))]
                    utility = pos * np.asarray(rank) - a.risk_weight * risk
                    selected = int(np.argmax(utility))
                    hold = not bool(utility[selected] > 0 and pos[selected] >= a.threshold)
                    outcomes = []
                    for row in rows:
                        value = evaluate(benchmark, row["child_actions"], references)
                        outcomes.append({"certified": certified(value), "reward": value["total_reward"], "repair_gain": float(value["total_reward"] - damaged_result["total_reward"])})
                    valid = [outcome["repair_gain"] for outcome in outcomes if outcome["certified"]]
                    chosen = 0.0 if hold else max(0.0, outcomes[selected]["repair_gain"])
                    headroom = original_result["total_reward"] - damaged_result["total_reward"]
                    record.update({"candidate_count": len(rows), "v2_action": "hold" if hold else "edit", "v2_repair_gain": chosen, "random_expected_repair_gain": float(np.mean([max(0.0, value) for value in valid])), "oracle_repair_gain": max(0.0, max(valid)), "recovery_fraction": float(chosen / headroom), "selected_outcome": None if hold else outcomes[selected], "selected_edit": None if hold else rows[selected]["edit"]})
                contexts.append(record)
                write(a.output, {"protocol": "rlcompopt-csmith-controlled-repair-v1", "scope": "locked external Csmith repair evaluation; no model fitting", "cohort": a.cohort, "damage_templates": requested, "certificate_profile": "csmith-deterministic-checksum-equivalence-v1", "strategy": {"name": a.strategy_name, "risk_weight": a.risk_weight, "threshold": a.threshold}, "contexts": contexts})
    finally:
        runner.env.close()
        runner.model.connection.close()
    eligible = [row for row in contexts if row["eligible"]]
    print(json.dumps({"contexts": len(contexts), "eligible_damages": len(eligible), "mean_v2_repair_gain": float(np.mean([row["v2_repair_gain"] for row in eligible])) if eligible else 0.0}, sort_keys=True))


if __name__ == "__main__":
    main()
