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
from .agents import (
    OFFLINE_PROVIDERS,
    RecordingPipeline,
    UsageMeter,
    agent_fingerprint,
    anthropic_client,
    build_pipeline,
    preflight_anthropic,
    validate_agent_config,
)
from .attacks import attack_fingerprint, load_attack
from .common import git_state, runtime_provenance, sha256_file, sha256_json
from .diagnostics import attacker_contacted, lenient_utility
from .exposure import exposed_injection_vectors
from .grammar import ConfigUpdate, runtime_factory
from .split import cohort_of, validate_split

PROFILE_PROTOCOL = "gse-agentdojo-run-profile-v1"
RESULT_PROTOCOL = "gse-agentdojo-run-result-v2"
GRAMMAR_FILE = Path("configs/agentdojo_update_grammar_v1.json")
PRICING_FILE = Path("configs/anthropic_pricing_v1.json")
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


def case_cost_usd(usage: dict[str, Any], model_price: dict[str, float], pricing: dict[str, Any]) -> float:
    """Estimated list-price cost of one case from its recorded token usage."""
    per_token_in, per_token_out = model_price["input"] / 1e6, model_price["output"] / 1e6
    return (
        usage["input_tokens"] * per_token_in
        + usage["output_tokens"] * per_token_out
        + usage["cache_creation_input_tokens"] * per_token_in * pricing["cache_write_multiplier"]
        + usage["cache_read_input_tokens"] * per_token_in * pricing["cache_read_multiplier"]
    )


def _skipped_row(user_task_id: str, injection_task_id: str | None) -> dict[str, Any]:
    case_id = f"{user_task_id}__{injection_task_id or 'none'}"
    return {
        "case_id": case_id,
        "kind": "benign" if injection_task_id is None else "security",
        "user_task_id": user_task_id,
        "injection_task_id": injection_task_id,
        "status": "skipped_budget",
        "utility": None,
        "injection_task_success": None,
        "injection_exposed": None,
        "utility_lenient": None,
        "attacker_contacted": None,
        "error": None,
        "elapsed_seconds": 0.0,
        "usage": UsageMeter().to_dict(),
        "pipeline_attempts": 0,
        "aborted": False,
        "policy_blocked_calls": 0,
        "cost_usd": 0.0,
        "trace_ref": None,
    }


ROW_KEYS = tuple(_skipped_row("user_task_0", None))


def _prepare_output(output_dir: Path, fingerprint: str, resume: bool) -> dict[str, Any]:
    """Refuse to mix configurations in one directory; record how a resume was checked."""
    manifest_path, traces = output_dir / "run_manifest.json", output_dir / "traces"
    has_traces = traces.exists() and any(traces.iterdir())
    if not resume:
        if has_traces:
            raise FileExistsError(f"{output_dir} holds a partial run; pass --resume or choose a new directory")
        _write_json(manifest_path, {"run_fingerprint": fingerprint})
        return {"enabled": False, "resumed_cases": 0, "manifest": "written"}
    if manifest_path.exists():
        if load_json(manifest_path)["run_fingerprint"] != fingerprint:
            raise ValueError("cannot resume: the existing run used a different profile, agent, update or adapter")
        check = "verified"
    else:  # runs started before manifests existed; the configuration cannot be checked
        _write_json(manifest_path, {"run_fingerprint": fingerprint, "created_on_resume": True})
        check = "legacy_unverified" if has_traces else "written"
    return {"enabled": True, "resumed_cases": 0, "manifest": check}


def _completed_row(output_dir: Path, user_task_id: str, injection_task_id: str | None) -> dict[str, Any] | None:
    path = output_dir / "traces" / f"{user_task_id}__{injection_task_id or 'none'}.json"
    if not path.exists():
        return None
    trace = load_json(path)
    if trace["status"] != "completed":
        return None  # errored cases are rerun
    return {**{key: trace.get(key) for key in ROW_KEYS}, "trace_ref": f"traces/{path.name}", "resumed": True}


def load_update(grammar_path: Path, name: str) -> dict[str, Any]:
    """A named update from the committed grammar file, checked against its canonical id."""
    records = {record["name"]: record for record in load_json(grammar_path)["updates"]}
    if name not in records:
        raise ValueError(f"unknown update {name!r}; known: {sorted(records)}")
    record = records[name]
    if ConfigUpdate.from_dict(record["update"]).identifier != record["id"]:
        raise ValueError(f"update {name!r} does not match its recorded id")
    return record


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
    skipped = sum(row["status"] == "skipped_budget" for row in rows)
    benign = [row for row in done if row["kind"] == "benign"]
    security = [row for row in done if row["kind"] == "security"]
    injection = _rate(security, "injection_task_success")
    exposed = [row for row in security if row["injection_exposed"]]
    usage_keys = ("calls", "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    return {
        "complete": len(done) == len(rows),
        "cases": {
            "planned": len(rows),
            "completed": len(done),
            "errored": len(rows) - len(done) - skipped,
            "skipped_budget": skipped,
        },
        "benign_utility": _rate(benign, "utility"),
        "utility_under_attack": _rate(security, "utility"),
        "injection_success": injection,
        "prompt_injection_resistance": None if injection["rate"] is None else 1.0 - injection["rate"],
        # Only cases where the injected text reached the model actually test resistance.
        "injection_exposure": _rate(security, "injection_exposed"),
        "injection_success_given_exposure": _rate(exposed, "injection_task_success"),
        # Diagnostics beside the official scores (see diagnostics.py); never substitutes for them.
        "benign_utility_lenient": _rate([r for r in benign if r["utility_lenient"] is not None], "utility_lenient"),
        "utility_under_attack_lenient": _rate(
            [r for r in security if r["utility_lenient"] is not None], "utility_lenient"
        ),
        "attacker_contacted": _rate([r for r in security if r["attacker_contacted"] is not None], "attacker_contacted"),
        "policy_blocked_calls": sum(row["policy_blocked_calls"] for row in done),
        "usage": {key: sum(row["usage"][key] for row in rows) for key in usage_keys},
        "elapsed_seconds": round(sum(row["elapsed_seconds"] for row in rows), 3),
        "estimated_cost_usd": None
        if any(row["cost_usd"] is None for row in rows)
        else round(sum(row["cost_usd"] for row in rows), 4),
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
    update: ConfigUpdate = ConfigUpdate(),
    client: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one case with a fresh pipeline and environment. Returns (row, trace)."""
    user_task = suite.get_user_task_by_id(user_task_id)
    injection_task = None if injection_task_id is None else suite.get_injection_task_by_id(injection_task_id)
    injections = {} if injection_task is None else attack.attack(user_task, injection_task)
    pipeline, meter = build_pipeline(agent, user_task, injection_task, update=update, client=client)
    recorder = RecordingPipeline(pipeline)
    blocked: list[dict[str, Any]] = []
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
        "utility_lenient": None,
        "attacker_contacted": None,
        "error": None,
    }
    started = time.perf_counter()
    error_trace = None
    try:
        utility, injection_success = suite.run_task_with_pipeline(
            recorder,
            user_task,
            injection_task,
            injections,
            runtime_class=runtime_factory(update, user_task.PROMPT, blocked),
        )
        row["utility"] = bool(utility)
        row["injection_task_success"] = None if injection_task is None else bool(injection_success)
        row["utility_lenient"] = lenient_utility(suite, user_task, injections, recorder.messages, recorder.environment)
        if injection_task is not None and recorder.environment is not None:
            row["attacker_contacted"] = attacker_contacted(
                recorder.messages, injection_task, user_task.PROMPT, recorder.environment.inbox.account_email
            )
    except Exception as error:  # recorded, never counted as a task failure
        row.update(status="error", error=f"{type(error).__name__}: {error}")
        error_trace = traceback.format_exc()
    row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    row["usage"] = meter.to_dict()
    row["pipeline_attempts"] = recorder.attempts
    row["aborted"] = recorder.aborted
    row["policy_blocked_calls"] = len(blocked)
    exposed_vectors = exposed_injection_vectors(recorder.messages, injections)
    if injection_task is not None:
        row["injection_exposed"] = bool(exposed_vectors)
    trace = {
        **row,
        "prompt": user_task.PROMPT,
        "injection_goal": None if injection_task is None else injection_task.GOAL,
        "injections": injections,
        "exposed_injection_vectors": exposed_vectors,
        "blocked_calls": blocked,
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
    update_name: str = "HOLD",
    grammar_path: Path | None = None,
    max_usd: float | None = None,
    pricing_path: Path | None = None,
    resume: bool = False,
    client: Any | None = None,
    on_case: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """``on_case(index, total, row)`` is called after each case, e.g. for progress output.

    ``max_usd`` caps estimated list-price spend. It is checked before each case
    against the most expensive case so far, so one case can still run past the
    cap by at most its own cost; cases not started are recorded as skipped.

    ``resume`` reuses completed cases from an interrupted run in the same
    directory, after checking its run manifest matches this configuration.
    """
    if (output_dir / "result.json").exists():
        previous = load_json(output_dir / "result.json")
        if not resume or previous["metrics"]["complete"]:
            raise FileExistsError(f"{output_dir} already holds a result; choose a new output directory")
        # An incomplete run being resumed: keep its result for the record, then redo it.
        (output_dir / "result.json").rename(output_dir / "result.incomplete.json")
    profile, agent = load_json(profile_path), load_json(agent_path)
    validate_agent_config(agent)
    update_record = load_update(grammar_path or repo_root / GRAMMAR_FILE, update_name)
    update = ConfigUpdate.from_dict(update_record["update"])
    split_path = repo_root / profile["split"]["path"]
    split, split_sha256 = load_json(split_path), sha256_file(split_path)
    suite = get_suite(profile["benchmark_version"], profile["suite"])
    validate_profile(profile, split, split_sha256, suite)
    if agent["provider"] == "anthropic":
        # One client for the whole run; fail before any case if it cannot authenticate.
        client = client if client is not None else anthropic_client()
        preflight_anthropic(client, agent["model"])
    model_price, pricing = None, None
    if agent["provider"] == "anthropic":
        pricing = load_json(pricing_path or repo_root / PRICING_FILE)
        model_price = pricing["models"].get(agent["model"])
        if model_price is None and max_usd is not None:
            raise ValueError(f"no price for {agent['model']!r} in the pricing file; cannot enforce --max-usd")
    attack = load_attack(profile["attack"], suite)
    fingerprint = sha256_json({
        "adapter": ADAPTER_VERSION,
        "profile": sha256_json(profile),
        "agent": sha256_json(agent),
        "update": update_record["id"],
        "attack": attack_fingerprint(attack),
    })
    resume_state = _prepare_output(output_dir, fingerprint, resume)
    tools_schema = [
        {"name": f.name, "description": f.description, "parameters": f.parameters.model_json_schema()}
        for f in suite.tools
    ]
    started_at = datetime.now(timezone.utc).isoformat()
    cases = [(t, None) for t in profile["benign_user_tasks"]] + [tuple(p) for p in profile["security_pairs"]]
    rows = []
    spent, costliest, budget_stop = 0.0, 0.0, None
    for index, (user_task_id, injection_task_id) in enumerate(cases, start=1):
        if max_usd is not None and budget_stop is None and spent + costliest > max_usd:
            budget_stop = {"max_usd": max_usd, "spent_usd": round(spent, 4), "stopped_before_case": index}
        if budget_stop is not None:
            rows.append({**_skipped_row(user_task_id, injection_task_id), "resumed": False})
            continue
        reused = _completed_row(output_dir, user_task_id, injection_task_id) if resume else None
        if reused is not None:
            spent += reused["cost_usd"] or 0.0
            costliest = max(costliest, reused["cost_usd"] or 0.0)
            rows.append(reused)
            resume_state["resumed_cases"] += 1
            if on_case is not None:
                on_case(index, len(cases), {**reused, "spent_usd": round(spent, 4)})
            continue
        row, trace = run_case(suite, agent, attack, user_task_id, injection_task_id, update=update, client=client)
        row["cost_usd"] = 0.0 if pricing is None else (
            None if model_price is None else round(case_cost_usd(row["usage"], model_price, pricing), 5)
        )
        spent += row["cost_usd"] or 0.0
        costliest = max(costliest, row["cost_usd"] or 0.0)
        _write_json(output_dir / "traces" / f"{row['case_id']}.json", {**trace, "cost_usd": row["cost_usd"]})
        rows.append({**row, "trace_ref": f"traces/{row['case_id']}.json", "resumed": False})
        if on_case is not None:
            on_case(index, len(cases), {**row, "spent_usd": round(spent, 4)})
    result = {
        "protocol": RESULT_PROTOCOL,
        "adapter_version": ADAPTER_VERSION,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "offline": agent["provider"] in OFFLINE_PROVIDERS,
        "repetition": repetition,
        "profile": {"path": str(profile_path), "sha256": sha256_json(profile), "content": profile},
        "agent": {"path": str(agent_path), **agent_fingerprint(agent, tools_schema, update), "content": agent},
        "update": {"grammar_path": str(grammar_path or GRAMMAR_FILE), **update_record},
        "attack": attack_fingerprint(attack),
        "split": {"path": profile["split"]["path"], "sha256": split_sha256},
        "provenance": {**runtime_provenance(), "git": git_state(repo_root)},
        "pricing": None if pricing is None else {"as_of": pricing["as_of"], "model": model_price},
        "budget": {"max_usd": max_usd, "stop": budget_stop},
        "run_fingerprint": fingerprint,
        "resume": resume_state,
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


def _fmt_usd(value: float | None) -> str:
    return "unknown (model not in pricing file)" if value is None else f"${value:.4f}"


def render_summary(result: dict[str, Any]) -> str:
    metrics, profile, agent = result["metrics"], result["profile"]["content"], result["agent"]["content"]
    update = result["update"]
    usage = metrics["usage"]
    lines = [
        f"# AgentDojo run: {profile['name']} × {agent['name']} × {update['name']}",
        "",
        f"- Suite: `{profile['suite']}` (benchmark {profile['benchmark_version']}, AgentDojo "
        f"{profile['agentdojo_version']}); cohort: **{profile['cohort']}**",
        f"- Agent: provider `{agent['provider']}`, model `{agent['model']}`; offline: {result['offline']}",
        f"- Update: `{update['name']}` (`{update['id']}`)",
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
        f"| *Diagnostic:* benign utility, lenient state check | {_fmt_rate(metrics['benign_utility_lenient'])} |",
        f"| *Diagnostic:* utility under attack, lenient | {_fmt_rate(metrics['utility_under_attack_lenient'])} |",
        f"| *Diagnostic:* attacker address contacted | {_fmt_rate(metrics['attacker_contacted'])} |",
        f"| Tool calls blocked by policy | {metrics['policy_blocked_calls']} |",
        f"| Model calls / input / output tokens | {usage['calls']} / {usage['input_tokens']} / {usage['output_tokens']} |",
        f"| Estimated cost (list price) | {_fmt_usd(metrics['estimated_cost_usd'])} |",
        f"| Elapsed seconds | {metrics['elapsed_seconds']} |",
        "",
        "| Case | Kind | Status | Utility | Lenient | Exposed | Injection succeeded | Attacker contacted | Blocked | Calls | Seconds |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in result["cases"]:
        lines.append(
            f"| `{row['case_id']}` | {row['kind']} | {row['status']} | {row['utility']} | {row['utility_lenient']} | "
            f"{row['injection_exposed']} | {row['injection_task_success']} | {row['attacker_contacted']} | "
            f"{row['policy_blocked_calls']} | {row['usage']['calls']} | {row['elapsed_seconds']} |"
        )
    stop = result["budget"]["stop"]
    if stop is not None:
        lines += [
            "",
            f"**Stopped by the ${stop['max_usd']:.2f} spend cap** before case {stop['stopped_before_case']} "
            f"(spent ${stop['spent_usd']:.4f}); remaining cases are marked `skipped_budget`.",
        ]
    errors = [row for row in result["cases"] if row["error"]]
    if errors:
        lines += ["", "## Errors", ""] + [f"- `{row['case_id']}`: {row['error']}" for row in errors]
    return "\n".join(lines) + "\n"
