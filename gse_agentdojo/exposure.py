"""Did the injected text actually reach the model?

AgentDojo places an injection where the *reference* solution looks.  An agent
that takes another route never reads it, and its "resistance" is untested.
Tool results are serialized (YAML by default, with escaped line folding), so
matching is done on parsed string values with whitespace normalized.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re
from typing import Any

import yaml

from .grammar import unwrap_tool_output

_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [s for key, item in value.items() for s in _strings(key) + _strings(item)]
    if isinstance(value, list | tuple):
        return [s for item in value for s in _strings(item)]
    return []


def _visible_text(content: str) -> str:
    """All string values of one serialized tool result, plus the raw text."""
    parsed: Any = None
    for loader in (yaml.safe_load, json.loads):
        try:
            parsed = loader(content)
            break
        except Exception:
            continue
    return _normalize(" ".join([content, *_strings(parsed)]))


def exposed_injection_vectors(messages: Sequence[Mapping[str, Any]], injections: Mapping[str, str]) -> list[str]:
    """Injection vector IDs whose text appears in a tool result shown to the model."""
    visible = " ".join(
        _visible_text(unwrap_tool_output(block["content"]))
        for message in messages
        if message["role"] == "tool"
        for block in (message["content"] or [])
        if block.get("type") == "text"
    )
    return sorted(vector for vector, text in injections.items() if _normalize(text) and _normalize(text) in visible)
