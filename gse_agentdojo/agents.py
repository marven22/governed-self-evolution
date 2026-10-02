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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import json
from typing import Any

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.llms.anthropic_llm import (
    _anthropic_to_assistant_message,
    _conversation_to_anthropic,
    _function_to_anthropic,
)
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor, tool_result_to_str
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, ChatToolResultMessage, text_content_block_from_string
import anthropic
from pydantic import BaseModel

from .attacker_scripts import attacker_calls
from .common import sha256_json, sha256_text
from .grammar import ConfigUpdate, apply_system_message, wrap_tool_output

AGENT_PROTOCOL = "gse-agentdojo-agent-v1"
STALLED_STREAM_RETRIES = 2
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
    "prompt_caching",
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
    transport_retries: int = 0
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
            "transport_retries": self.transport_retries,
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
        if self.agent["prompt_caching"]:
            # Caching changes billing, not outputs. Tools render before the system
            # prompt, so this marker caches tools + system once for every case;
            # the top-level field also caches each case's growing conversation.
            request["system"] = [{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}]
            request["cache_control"] = {"type": "ephemeral"}
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
        # Streaming avoids HTTP timeouts on long turns; SDK retries cover 429/5xx
        # before a stream starts. A stream that stalls mid-response raises a
        # timeout (see anthropic_client) and is retried here from scratch.
        for attempt in range(1 + STALLED_STREAM_RETRIES):
            try:
                with self.client.messages.stream(**request) as stream:
                    completion = stream.get_final_message()
                break
            except (anthropic.APITimeoutError, anthropic.APIConnectionError):
                self.meter.transport_retries += 1
                if attempt == STALLED_STREAM_RETRIES:
                    raise
        self.meter.record(completion)
        return query, runtime, env, [*messages, _anthropic_to_assistant_message(completion)], extra_args


class ScriptedExecutor(BasePipelineElement):
    """Executes a fixed call script without a model: the offline oracles' "agent".

    Calls run with ``raise_on_error=False``, so a call refused by a tool policy
    or broken by an attack comes back as an error result, exactly as a model
    would see it. A script cannot know an answer it failed to read, so after any
    failed call its final answer is empty. ``answer=None`` adds no final message.
    """

    def __init__(self, script: Callable[[Env], list[FunctionCall]], answer: str | None, formatter: Callable) -> None:
        self.script = script
        self.answer = answer
        self.formatter = formatter

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        new_messages: list[ChatMessage] = []
        failed = False
        for call in self.script(env):
            new_messages.append(
                ChatAssistantMessage(role="assistant", tool_calls=[call], content=[text_content_block_from_string("")])
            )
            result, error = runtime.run_function(env, call.function, call.args)
            failed = failed or error is not None
            new_messages.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string("" if error else self.formatter(result))],
                    tool_call=call,
                    tool_call_id=None,
                    error=error,
                )
            )
        if self.answer is not None:
            final = "" if failed else self.answer
            new_messages.append(
                ChatAssistantMessage(role="assistant", content=[text_content_block_from_string(final)], tool_calls=None)
            )
        return query, runtime, env, [*messages, *new_messages], extra_args


def reference_executor(user_task: BaseUserTask, formatter: Callable = tool_result_to_str) -> ScriptedExecutor:
    return ScriptedExecutor(user_task.ground_truth, user_task.GROUND_TRUTH_OUTPUT, formatter)


def attacker_executor(
    injection_task: BaseInjectionTask, formatter: Callable = tool_result_to_str, answer: str | None = None
) -> ScriptedExecutor:
    """Carries out an injection goal literally. Run alone, it needs ``answer=""`` to end the episode."""
    return ScriptedExecutor(lambda env: attacker_calls(injection_task, env), answer, formatter)


class RecordingPipeline(BasePipelineElement):
    """Keeps the final message list, which AgentDojo's runner does not return."""

    def __init__(self, inner: AgentPipeline) -> None:
        self.inner = inner
        self.name = inner.name
        self.messages: list[ChatMessage] = []
        self.environment: Env | None = None
        self.attempts = 0
        self.aborted = False

    def query(self, query, runtime, env=EmptyEnv(), messages=[], extra_args={}):
        self.attempts += 1
        try:
            result = self.inner.query(query, runtime, env, messages, extra_args)
        except AbortAgentError as error:
            self.messages, self.aborted = list(error.messages), True
            self.environment = error.task_environment
            raise
        self.messages, self.environment = list(result[3]), result[2]
        return result


def anthropic_client() -> Any:
    # A live stream receives events or pings continuously; 90 s of silence
    # means the connection stalled, so fail it and let ClaudeLLM retry.
    return anthropic.Anthropic(timeout=anthropic.Timeout(600.0, read=90.0, connect=15.0))


def preflight_anthropic(client: Any, model: str) -> None:
    """Check credentials and the model ID with a free Models API call before any case runs."""
    try:
        client.models.retrieve(model)
    except Exception as error:
        raise RuntimeError(
            f"Anthropic preflight failed for model {model!r} ({type(error).__name__}: {error}). "
            "Set the key in this terminal with `export ANTHROPIC_API_KEY=...` and check the model ID. "
            "No case was run and no output was written."
        ) from error


def build_pipeline(
    agent: dict[str, Any],
    user_task: BaseUserTask,
    injection_task: BaseInjectionTask | None,
    *,
    update: ConfigUpdate = ConfigUpdate(),
    client: Any | None = None,
) -> tuple[AgentPipeline, UsageMeter]:
    """Build a fresh pipeline for one case; state never carries across cases.

    The update changes only the system message and tool-output format here; its
    tool-permission level acts in the runtime (``grammar.runtime_factory``).
    """
    meter = UsageMeter()
    system = SystemMessage(apply_system_message(resolve_system_message(agent), update))
    formatter = wrap_tool_output(tool_output_formatter(agent), update)
    provider = agent["provider"]
    if provider == "ground_truth_oracle":
        elements = [system, InitQuery(), reference_executor(user_task, formatter)]
    elif provider == "compromised_oracle":
        attack = [] if injection_task is None else [attacker_executor(injection_task, formatter)]
        elements = [system, InitQuery(), *attack, reference_executor(user_task, formatter)]
    else:
        llm = ClaudeLLM(client if client is not None else anthropic_client(), agent, meter)
        loop = ToolsExecutionLoop([ToolsExecutor(formatter), llm], max_iters=agent["max_tool_iterations"])
        elements = [system, InitQuery(), llm, loop]
    pipeline = AgentPipeline(elements)
    pipeline.name = agent["name"]
    return pipeline, meter


def agent_fingerprint(
    agent: dict[str, Any], tools_schema: list[dict[str, Any]], update: ConfigUpdate = ConfigUpdate()
) -> dict[str, str]:
    return {
        "agent_config_sha256": sha256_json(agent),
        "system_message_sha256": sha256_text(apply_system_message(resolve_system_message(agent), update)),
        "tools_schema_sha256": sha256_json(tools_schema),
    }
