"""Fail-closed CompilerGym reproducibility and validation probe.

This is deliberately *not* a governor experiment.  It establishes the
measurement contract required before collecting any self-evolution
transitions: a fixed compiler policy evaluated twice from the same benchmark
must produce identical evidence, and the transformed program must validate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def json_value(value: Any) -> Any:
    """Convert Gym/NumPy values to a stable JSON representation."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def unwrap_reset(result: Any) -> Any:
    # Gym <=0.21 returns observation; newer Gym returns (observation, info).
    return result[0] if isinstance(result, tuple) and len(result) == 2 else result


def unwrap_step(result: Any) -> tuple[Any, float, bool, dict[str, Any]]:
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward or 0.0), bool(done), dict(info or {})
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward or 0.0), bool(terminated or truncated), dict(info or {})
    raise RuntimeError(f"Unexpected env.step() result with {len(result)} entries")


def configure_cbench_runtime_compatibility() -> None:
    """Pass locally supplied legacy runtime libraries to cBench validators.

    CompilerGym's archived LLVM runtime needs ``libtinfo.so.5``. Its cBench
    validator intentionally constructs a minimal subprocess environment and
    otherwise drops ``LD_LIBRARY_PATH``.  This small, runtime-only wrapper
    preserves that isolation while carrying the explicitly supplied library
    path into the validator. It is not an experimental intervention.
    """
    from compiler_gym.envs.llvm.datasets import cbench

    if getattr(cbench, "_governed_runtime_compatibility", False):
        return
    inherited_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not inherited_library_path:
        return
    original = cbench._compile_and_run_bitcode_file

    def compile_and_run_with_runtime(*args: Any, **kwargs: Any) -> Any:
        validator_env = dict(kwargs.get("env", {}))
        validator_env["LD_LIBRARY_PATH"] = inherited_library_path
        kwargs["env"] = validator_env
        return original(*args, **kwargs)

    original_popen = cbench.Popen

    def popen_with_runtime(*args: Any, **kwargs: Any) -> Any:
        # The sanitizer compilation branch constructs a new minimal
        # environment of its own, so preserve the same explicitly supplied
        # library path there as well.
        process_env = dict(kwargs.get("env", {}))
        process_env["LD_LIBRARY_PATH"] = inherited_library_path
        kwargs["env"] = process_env
        return original_popen(*args, **kwargs)

    cbench._compile_and_run_bitcode_file = compile_and_run_with_runtime
    cbench.Popen = popen_with_runtime
    # CompilerGym 0.2.5 ships LLVM 10 sanitizer runtimes that are not
    # portable to this contemporary WSL image (TSAN/MSAN crash independently
    # of the optimized program). Keep the reference-output validator, which
    # compiles and compares observable program behavior, and report this
    # limitation explicitly. Sanitizer validation is a later portability gate.
    for name, validators in cbench.VALIDATORS.items():
        cbench.VALIDATORS[name] = validators[:1]
    cbench._governed_runtime_compatibility = True


def evaluate_fixed_policy(benchmark: str, actions: list[int]) -> dict[str, Any]:
    import compiler_gym

    configure_cbench_runtime_compatibility()

    env = compiler_gym.make(
        "llvm-v0",
        benchmark=benchmark,
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
    )
    try:
        initial_observation = json_value(unwrap_reset(env.reset(benchmark=benchmark)))
        trace: list[dict[str, Any]] = []
        final_observation: Any = initial_observation
        for action in actions:
            final_observation, reward, done, info = unwrap_step(env.step(action))
            trace.append(
                {
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "info": json_value(info),
                }
            )
            if done:
                break

        try:
            raw_validation = env.validate()
            # ValidationResult includes elapsed wall time, which is not part
            # of the semantic certificate and must not affect exact replay.
            validation = {
                "okay": bool(raw_validation.okay()),
                "reward_validated": bool(raw_validation.reward_validated),
                "actions_replay_failed": bool(raw_validation.actions_replay_failed),
                "reward_validation_failed": bool(raw_validation.reward_validation_failed),
                "benchmark_semantics_validated": bool(
                    raw_validation.benchmark_semantics_validated
                ),
                "benchmark_semantics_validation_failed": bool(
                    raw_validation.benchmark_semantics_validation_failed
                ),
                "error_details": str(raw_validation.error_details),
                "errors": [
                    {
                        "type": str(error.type),
                        "data": json_value(error.data),
                    }
                    for error in raw_validation.errors
                ],
            }
            validation_available = True
        except Exception as exc:  # A missing verifier is a failed certificate.
            validation = {"error": f"{type(exc).__name__}: {exc}"}
            validation_available = False

        return {
            "benchmark": benchmark,
            "validation_profile": "cbench-reference-output-equivalence",
            "initial_observation": initial_observation,
            "final_observation": json_value(final_observation),
            "trace": trace,
            "total_reward": sum(step["reward"] for step in trace),
            "validation_available": validation_available,
            "validation": validation,
        }
    finally:
        env.close()


def fingerprint(record: dict[str, Any]) -> str:
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="cbench-v1/qsort")
    parser.add_argument(
        "--actions",
        default="0,1,2",
        help="Comma-separated fixed LLVM action indices used in both repeats.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    actions = [int(item) for item in args.actions.split(",") if item.strip()]
    if not actions:
        raise ValueError("At least one fixed action is required")

    first = evaluate_fixed_policy(args.benchmark, actions)
    second = evaluate_fixed_policy(args.benchmark, actions)
    first_hash, second_hash = fingerprint(first), fingerprint(second)
    passed = (
        first_hash == second_hash
        and first["validation_available"]
        and second["validation_available"]
        and first["validation"]["okay"]
        and second["validation"]["okay"]
        and first["validation"]["benchmark_semantics_validated"]
        and second["validation"]["benchmark_semantics_validated"]
    )
    report = {
        "protocol": "compilergym-feasibility-v1",
        "benchmark": args.benchmark,
        "actions": actions,
        "first": first,
        "second": second,
        "first_fingerprint": first_hash,
        "second_fingerprint": second_hash,
        "exact_repeat_match": first_hash == second_hash,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("benchmark", "exact_repeat_match", "passed")}))
    if not passed:
        raise SystemExit("CompilerGym feasibility contract failed; no evolution data may be collected.")


if __name__ == "__main__":
    main()
