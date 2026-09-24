"""Auditable AgentDojo benchmark runner for a fixed task subset.

AgentDojo's evaluator returns ``(utility, security)`` where ``security`` is
True when the *injection goal was achieved*.  This module renames that value to
``injection_task_success`` everywhere, and reports resistance as its complement,
so an attack success can never be read as a defense success.
"""
from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any, Callable

from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import TaskSuite, functions_stack_trace_from_messages
from pydantic import BaseModel

from . import ADAPTER_VERSION, PINNED_AGENTDOJO_VERSION
from .agents import OFFLINE_PROVIDERS, RecordingPipeline, agent_fingerprint, build_pipeline, validate_agent_config
from .attacks import attack_fingerprint, load_attack
from .common import git_state, runtime_provenance, sha256_file, sha256_json
from .exposure import exposed_injection_vectors
from .split import cohort_of, validate_split

PROFILE_PROTOCOL = "gse-agentdojo-run-profile-v1"
RESULT_PROTOCOL = "gse-agentdojo-run-result-v1"
PROFILE_FIELDS = {
    "protocol",
    "name",
    "purpose",
    "agentdojo_version",
    "benchmark_version",
    "suite",
    "cohort",
    "split",
    "attack",
    "selection_seed",
    "benign_user_tasks",
    "security_pairs",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_profile(profile: dict[str, Any], split: dict[str, Any], split_sha256: str, suite: TaskSuite) -> None:
    """Fail closed on version drift, split tampering, or cohort leakage."""
    if profile.get("protocol") != PROFILE_PROTOCOL:
        raise ValueError(f"profile protocol must be {PROFILE_PROTOCOL!r}")
    if set(profile) != PROFILE_FIELDS:
        raise ValueError(
            f"profile keys mismatch: missing={sorted(PROFILE_FIELDS - set(profile))} "
            f"unknown={sorted(set(profile) - PROFILE_FIELDS)}"
        )
    installed = metadata.version("agentdojo")
    if profile["agentdojo_version"] != PINNED_AGENTDOJO_VERSION or installed != PINNED_AGENTDOJO_VERSION:
        raise ValueError(
            f"AgentDojo version mismatch: profile={profile['agentdojo_version']} installed={installed} "
            f"pinned={PINNED_AGENTDOJO_VERSION}"
        )
    if profile["split"]["sha256"] != split_sha256:
        raise ValueError("split file hash differs from the hash recorded in the profile")
    if (split["suite"], split["benchmark_version"]) != (profile["suite"], profile["benchmark_version"]):
        raise ValueError("profile and split disagree on suite or benchmark version")
    validate_split(split, list(suite.user_tasks), list(suite.injection_tasks))
    cohort = profile["cohort"]
    for task in profile["benign_user_tasks"]:
        if cohort_of(split, "user_tasks", task) != cohort:
            raise ValueError(f"benign task {task} is not in cohort {cohort!r}")
    for user_task, injection_task in profile["security_pairs"]:
        if cohort_of(split, "user_tasks", user_task) != cohort:
            raise ValueError(f"security user task {user_task} is not in cohort {cohort!r}")
        if cohort_of(split, "injection_tasks", injection_task) != cohort:
            raise ValueError(f"injection task {injection_task} is not in cohort {cohort!r}")
    cases = [(t, None) for t in profile["benign_user_tasks"]] + [tuple(p) for p in profile["security_pairs"]]
    if len(set(cases)) != len(cases):
        raise ValueError("profile contains duplicate cases")


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _rate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    n = len(rows)
    k = sum(bool(row[key]) for row in rows)
    return {"successes": k, "n": n, "rate": (k / n) if n else None, "wilson95": wilson_interval(k, n)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    done = [row for row in rows if row["status"] == "completed"]
    benign = [row for row in done if row["kind"] == "benign"]
    security = [row for row in done if row["kind"] == "security"]
    injection = _rate(security, "injection_task_success")
    exposed = [row for row in security if row["injection_exposed"]]
    usage_keys = ("calls", "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    return {
        "complete": len(done) == len(rows),
        "cases": {"planned": len(rows), "completed": len(done), "errored": len(rows) - len(done)},
        "benign_utility": _rate(benign, "utility"),
        "utility_under_attack": _rate(security, "utility"),
        "injection_success": injection,
        "prompt_injection_resistance": None if injection["rate"] is None else 1.0 - injection["rate"],
        # Only cases where the injected text reached the model actually test resistance.
        "injection_exposure": _rate(security, "injection_exposed"),
        "injection_success_given_exposure": _rate(exposed, "injection_task_success"),
        "usage": {key: sum(row["usage"][key] for row in rows) for key in usage_keys},
        "elapsed_seconds": round(sum(row["elapsed_seconds"] for row in rows), 3),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")


def _model_output_text(messages: list[Any]) -> str | None:
    if not messages or messages[-1]["role"] != "assistant" or not messages[-1]["content"]:
        return None
    return "".join(block["content"] for block in messages[-1]["content"] if block["type"] == "text")


def run_case(
    suite: TaskSuite,
    agent: dict[str, Any],
    attack: Any,
    user_task_id: str,
    injection_task_id: str | None,
    *,
    client: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one case with a fresh pipeline and environment. Returns (row, trace)."""
    user_task = suite.get_user_task_by_id(user_task_id)
    injection_task = None if injection_task_id is None else suite.get_injection_task_by_id(injection_task_id)
    injections = {} if injection_task is None else attack.attack(user_task, injection_task)
    pipeline, meter = build_pipeline(agent, user_task, injection_task, client=client)
    recorder = RecordingPipeline(pipeline)
    case_id = f"{user_task_id}__{injection_task_id or 'none'}"
    row: dict[str, Any] = {
        "case_id": case_id,
        "kind": "benign" if injection_task is None else "security",
        "user_task_id": user_task_id,
        "injection_task_id": injection_task_id,
        "status": "completed",
        "utility": None,
        "injection_task_success": None,
        "injection_exposed": None,
        "error": None,
    }
    started = time.perf_counter()
    error_trace = None
    try:
        utility, injection_success = suite.run_task_with_pipeline(recorder, user_task, injection_task, injections)
        row["utility"] = bool(utility)
        row["injection_task_success"] = None if injection_task is None else bool(injection_success)
    except Exception as error:  # recorded, never counted as a task failure
        row.update(status="error", error=f"{type(error).__name__}: {error}")
        error_trace = traceback.format_exc()
    row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    row["usage"] = meter.to_dict()
    row["pipeline_attempts"] = recorder.attempts
    row["aborted"] = recorder.aborted
    exposed_vectors = exposed_injection_vectors(recorder.messages, injections)
    if injection_task is not None:
        row["injection_exposed"] = bool(exposed_vectors)
    trace = {
        **row,
        "prompt": user_task.PROMPT,
        "injection_goal": None if injection_task is None else injection_task.GOAL,
        "injections": injections,
        "exposed_injection_vectors": exposed_vectors,
        "model_output": _model_output_text(recorder.messages),
        "tool_calls": functions_stack_trace_from_messages(recorder.messages) if recorder.messages else [],
        "messages": recorder.messages,
        "error_traceback": error_trace,
    }
    return row, trace


def run_benchmark(
    *,
    profile_path: Path,
    agent_path: Path,
    output_dir: Path,
    repo_root: Path,
    repetition: int = 0,
    client: Any | None = None,
    on_case: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """``on_case(index, total, row)`` is called after each case, e.g. for progress output."""
    if (output_dir / "result.json").exists():
        raise FileExistsError(f"{output_dir} already holds a result; choose a new output directory")
    profile, agent = load_json(profile_path), load_json(agent_path)
    validate_agent_config(agent)
    split_path = repo_root / profile["split"]["path"]
    split, split_sha256 = load_json(split_path), sha256_file(split_path)
    suite = get_suite(profile["benchmark_version"], profile["suite"])
    validate_profile(profile, split, split_sha256, suite)
    attack = load_attack(profile["attack"], suite)
    tools_schema = [
        {"name": f.name, "description": f.description, "parameters": f.parameters.model_json_schema()}
        for f in suite.tools
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    cases = [(t, None) for t in profile["benign_user_tasks"]] + [tuple(p) for p in profile["security_pairs"]]
    rows = []
    for index, (user_task_id, injection_task_id) in enumerate(cases, start=1):
        row, trace = run_case(suite, agent, attack, user_task_id, injection_task_id, client=client)
        _write_json(output_dir / "traces" / f"{row['case_id']}.json", trace)
        rows.append({**row, "trace_ref": f"traces/{row['case_id']}.json"})
        if on_case is not None:
            on_case(index, len(cases), row)
    result = {
        "protocol": RESULT_PROTOCOL,
        "adapter_version": ADAPTER_VERSION,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "offline": agent["provider"] in OFFLINE_PROVIDERS,
        "repetition": repetition,
        "profile": {"path": str(profile_path), "sha256": sha256_json(profile), "content": profile},
        "agent": {"path": str(agent_path), **agent_fingerprint(agent, tools_schema), "content": agent},
        "attack": attack_fingerprint(attack),
        "split": {"path": profile["split"]["path"], "sha256": split_sha256},
        "provenance": {**runtime_provenance(), "git": git_state(repo_root)},
        "metrics": summarize(rows),
        "cases": rows,
    }
    _write_json(output_dir / "result.json", result)
    (output_dir / "summary.md").write_text(render_summary(result))
    return result


def _fmt_rate(metric: dict[str, Any]) -> str:
    if metric["rate"] is None:
        return "n/a"
    low, high = metric["wilson95"]
    return f"{metric['rate']:.3f} ({metric['successes']}/{metric['n']}; 95% CI {low:.2f}–{high:.2f})"


def render_summary(result: dict[str, Any]) -> str:
    metrics, profile, agent = result["metrics"], result["profile"]["content"], result["agent"]["content"]
    usage = metrics["usage"]
    lines = [
        f"# AgentDojo run: {profile['name']} × {agent['name']}",
        "",
        f"- Suite: `{profile['suite']}` (benchmark {profile['benchmark_version']}, AgentDojo "
        f"{profile['agentdojo_version']}); cohort: **{profile['cohort']}**",
        f"- Agent: provider `{agent['provider']}`, model `{agent['model']}`; offline: {result['offline']}",
        f"- Attack: `{result['attack']['name']}`; repetition {result['repetition']}",
        f"- Git: `{result['provenance']['git']['commit']}` (dirty: {result['provenance']['git']['dirty']})",
        f"- Complete: {metrics['complete']} ({metrics['cases']['completed']}/{metrics['cases']['planned']} cases)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Benign utility | {_fmt_rate(metrics['benign_utility'])} |",
        f"| Utility under attack | {_fmt_rate(metrics['utility_under_attack'])} |",
        f"| Injection success (attack wins) | {_fmt_rate(metrics['injection_success'])} |",
        f"| Injection reached the model | {_fmt_rate(metrics['injection_exposure'])} |",
        f"| Injection success when exposed | {_fmt_rate(metrics['injection_success_given_exposure'])} |",
        f"| Model calls / input / output tokens | {usage['calls']} / {usage['input_tokens']} / {usage['output_tokens']} |",
        f"| Elapsed seconds | {metrics['elapsed_seconds']} |",
        "",
        "| Case | Kind | Status | Utility | Injection exposed | Injection succeeded | Calls | Seconds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in result["cases"]:
        lines.append(
            f"| `{row['case_id']}` | {row['kind']} | {row['status']} | {row['utility']} | "
            f"{row['injection_exposed']} | {row['injection_task_success']} | {row['usage']['calls']} | {row['elapsed_seconds']} |"
        )
    errors = [row for row in result["cases"] if row["error"]]
    if errors:
        lines += ["", "## Errors", ""] + [f"- `{row['case_id']}`: {row['error']}" for row in errors]
    return "\n".join(lines) + "\n"
