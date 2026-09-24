"""Agent pipelines selected entirely by configuration.

Providers:

* ``ground_truth_oracle`` — executes the user task's reference tool calls.  No
  model call; must reach full utility and zero injection success.
* ``compromised_oracle`` — executes the injection goal, then the user task.
  No model call; the security evaluator must flag every security case.
* ``anthropic`` — a Claude model through the official SDK, with per-call usage
  recorded.  Requests omit sampling parameters the config leaves null, because
  current models reject ``temperature``.

The two oracles bracket the evaluator: they test that utility and security are
measured correctly before any paid model is run.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
import json
from typing import Any

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.agent_pipeline.llms.anthropic_llm import (
    _anthropic_to_assistant_message,
    _conversation_to_anthropic,
    _function_to_anthropic,
)
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, text_content_block_from_string
from pydantic import BaseModel

from .attacker_scripts import attacker_calls
from .common import sha256_json, sha256_text

AGENT_PROTOCOL = "gse-agentdojo-agent-v1"
PROVIDERS = ("ground_truth_oracle", "compromised_oracle", "anthropic")
OFFLINE_PROVIDERS = ("ground_truth_oracle", "compromised_oracle")
AGENT_FIELDS = {
    "protocol",
    "name",
    "provider",
    "model",
    "system_message",
    "max_tokens",
    "temperature",
    "effort",
    "thinking",
    "tool_output_format",
    "max_tool_iterations",
    "seed",
}


def validate_agent_config(agent: dict[str, Any]) -> None:
    """Reject unknown or missing keys, so no setting is silently defaulted."""
    if agent.get("protocol") != AGENT_PROTOCOL:
        raise ValueError(f"agent protocol must be {AGENT_PROTOCOL!r}")
    if set(agent) != AGENT_FIELDS:
        raise ValueError(
            f"agent config keys mismatch: missing={sorted(AGENT_FIELDS - set(agent))} "
            f"unknown={sorted(set(agent) - AGENT_FIELDS)}"
        )
    if agent["provider"] not in PROVIDERS:
        raise ValueError(f"provider must be one of {PROVIDERS}; got {agent['provider']!r}")
    if agent["provider"] == "anthropic" and not agent["model"]:
        raise ValueError("anthropic provider requires a model")
    if agent["tool_output_format"] not in ("yaml", "json"):
        raise ValueError("tool_output_format must be 'yaml' or 'json'")


def resolve_system_message(agent: dict[str, Any]) -> str:
    """``agentdojo:<name>`` loads an upstream message; any other string is literal."""
    value = agent["system_message"]
    if value.startswith("agentdojo:"):
        return load_system_message(value.removeprefix("agentdojo:"))
    return value


def tool_output_formatter(agent: dict[str, Any]):
    if agent["tool_output_format"] == "json":
        return lambda result: tool_result_to_str(result, dump_fn=json.dumps)
    return tool_result_to_str


@dataclass
class UsageMeter:
    """Accumulates API usage and stop reasons for one benchmark case."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    stop_reasons: Counter = field(default_factory=Counter)
    response_ids: list[str] = field(default_factory=list)
    served_models: list[str] = field(default_factory=list)

    def record(self, completion: Any) -> None:
        usage = completion.usage
        self.calls += 1
        self.input_tokens += usage.input_tokens or 0
        self.output_tokens += usage.output_tokens or 0
        self.cache_creation_input_tokens += getattr(usage, "cache_creation_input_tokens", None) or 0
        self.cache_read_input_tokens += getattr(usage, "cache_read_input_tokens", None) or 0
        self.stop_reasons[str(completion.stop_reason)] += 1
        self.response_ids.append(completion.id)
        if completion.model not in self.served_models:
            self.served_models.append(completion.model)

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "stop_reasons": dict(sorted(self.stop_reasons.items())),
            "response_ids": list(self.response_ids),
            "served_models": list(self.served_models),
        }


def _plain(value: Any) -> Any:
    """Convert SDK response models embedded by AgentDojo into request dicts."""
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


class ClaudeLLM(BasePipelineElement):
    """AgentDojo LLM element for current Claude models via the official SDK."""

    def __init__(self, client: Any, agent: dict[str, Any], meter: UsageMeter) -> None:
        self.client = client
        self.agent = agent
        self.meter = meter
        self.name = agent["model"]

    def build_request(self, messages: Sequence[ChatMessage], runtime: FunctionsRuntime) -> dict[str, Any]:
        system_prompt, anthropic_messages = _conversation_to_anthropic(messages)
        request: dict[str, Any] = {
            "model": self.agent["model"],
            "max_tokens": self.agent["max_tokens"],
            "messages": _plain(anthropic_messages),
        }
        tools = [_function_to_anthropic(tool) for tool in runtime.functions.values()]
        if tools:
            request["tools"] = _plain(tools)
        if system_prompt:
            request["system"] = system_prompt
        if self.agent["temperature"] is not None:
            request["temperature"] = self.agent["temperature"]
        if self.agent["effort"] is not None:
            request["output_config"] = {"effort": self.agent["effort"]}
        if self.agent["thinking"] is not None:
            request["thinking"] = self.agent["thinking"]
        return request

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        request = self.build_request(messages, runtime)
        # Streaming avoids HTTP timeouts on long turns; SDK retries cover 429/5xx.
        with self.client.messages.stream(**request) as stream:
            completion = stream.get_final_message()
        self.meter.record(completion)
        return query, runtime, env, [*messages, _anthropic_to_assistant_message(completion)], extra_args


class _AttackerReference:
    """Duck-typed task for GroundTruthPipeline: executes an injection goal."""

    GROUND_TRUTH_OUTPUT = ""

    def __init__(self, injection_task: BaseInjectionTask) -> None:
        self.injection_task = injection_task

    def ground_truth(self, env: Env):
        return attacker_calls(self.injection_task, env)


def attacker_pipeline(injection_task: BaseInjectionTask) -> GroundTruthPipeline:
    return GroundTruthPipeline(_AttackerReference(injection_task))  # type: ignore[arg-type]


class CompromisedOracle(BasePipelineElement):
    """Obeys the injection first, then completes the user task."""

    def __init__(self, user_task: BaseUserTask, injection_task: BaseInjectionTask | None) -> None:
        self.user_task = user_task
        self.injection_task = injection_task

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        if self.injection_task is not None:
            query, runtime, env, messages, extra_args = attacker_pipeline(self.injection_task).query(
                query, runtime, env, messages, extra_args
            )
            messages = messages[:-1]  # drop the injection's final answer; the user task supplies it
        try:
            return GroundTruthPipeline(self.user_task).query(query, runtime, env, messages, extra_args)
        except ValueError:
            # The attack destroyed data the user task needs (e.g. deleted its email):
            # stop and answer nothing, so the evaluator scores a utility failure.
            final = ChatAssistantMessage(role="assistant", content=[text_content_block_from_string("")], tool_calls=None)
            return query, runtime, env, [*messages, final], extra_args


class RecordingPipeline(BasePipelineElement):
    """Keeps the final message list, which AgentDojo's runner does not return."""

    def __init__(self, inner: AgentPipeline) -> None:
        self.inner = inner
        self.name = inner.name
        self.messages: list[ChatMessage] = []
        self.attempts = 0
        self.aborted = False

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        self.attempts += 1
        try:
            result = self.inner.query(query, runtime, env, messages, extra_args)
        except AbortAgentError as error:
            self.messages, self.aborted = list(error.messages), True
            raise
        self.messages = list(result[3])
        return result


def _anthropic_client() -> Any:
    import anthropic  # imported lazily: offline runs never need credentials

    return anthropic.Anthropic()


def build_pipeline(
    agent: dict[str, Any],
    user_task: BaseUserTask,
    injection_task: BaseInjectionTask | None,
    *,
    client: Any | None = None,
) -> tuple[AgentPipeline, UsageMeter]:
    """Build a fresh pipeline for one case; state never carries across cases."""
    meter = UsageMeter()
    system = SystemMessage(resolve_system_message(agent))
    provider = agent["provider"]
    if provider == "ground_truth_oracle":
        elements = [system, InitQuery(), GroundTruthPipeline(user_task)]
    elif provider == "compromised_oracle":
        elements = [system, InitQuery(), CompromisedOracle(user_task, injection_task)]
    else:
        llm = ClaudeLLM(client if client is not None else _anthropic_client(), agent, meter)
        loop = ToolsExecutionLoop([ToolsExecutor(tool_output_formatter(agent)), llm], max_iters=agent["max_tool_iterations"])
        elements = [system, InitQuery(), llm, loop]
    pipeline = AgentPipeline(elements)
    pipeline.name = agent["name"]
    return pipeline, meter


def agent_fingerprint(agent: dict[str, Any], tools_schema: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "agent_config_sha256": sha256_json(agent),
        "system_message_sha256": sha256_text(resolve_system_message(agent)),
        "tools_schema_sha256": sha256_json(tools_schema),
    }
