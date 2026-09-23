"""Z.ai Server-Sent Events decoding and response reconstruction.

Z.ai uses the OpenAI Chat Completions stream shape, but reasoning deltas are
returned beside visible content and tool arguments are commonly split across
multiple events.  This module keeps those provider details out of the facade
adapter: callers receive visible and reasoning text separately and can turn
the accumulated state into one ordinary Chat Completions response at EOF.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from typing import Any, Optional

from llm_api_adapter.llms.transports import SSEEvent


@dataclass(frozen=True)
class ZaiStreamDelta:
    """The two text channels decoded from one Z.ai SSE event.

    ``visible_text`` is the only channel that may be yielded to a caller.
    Reasoning is retained separately so an adapter can expose it through the
    Core reasoning callbacks without leaking it into normal output.
    """

    visible_text: str = ""
    reasoning_text: str = ""


@dataclass
class ZaiStreamState:
    """Request-local state accumulated while reading one Z.ai stream."""

    response_metadata: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    text_parts: list[str] = field(default_factory=list)
    reasoning_parts: list[str] = field(default_factory=list)
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    finish_reason: str | None = None

    @property
    def visible_text(self) -> str:
        """Return the complete visible response accumulated so far."""

        return "".join(self.text_parts)

    @property
    def reasoning_text(self) -> str:
        """Return reasoning without mixing it into visible response text."""

        return "".join(self.reasoning_parts)


def _accumulate_tool_calls(
    raw_tool_calls: Any,
    tool_calls: dict[int, dict[str, Any]],
) -> None:
    """Merge OpenAI-style tool-call fragments into indexed call state."""

    if not isinstance(raw_tool_calls, list):
        return

    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, Mapping):
            continue
        index = raw_tool_call.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            continue

        target = tool_calls.setdefault(index, {"function": {"arguments": ""}})
        for field_name in ("id", "type"):
            value = raw_tool_call.get(field_name)
            if value is not None:
                target[field_name] = value

        function = raw_tool_call.get("function")
        if not isinstance(function, Mapping):
            continue
        target_function = target.setdefault("function", {})
        name = function.get("name")
        if name is not None:
            target_function["name"] = name

        arguments = function.get("arguments")
        if isinstance(arguments, str):
            target_function["arguments"] = (
                f"{target_function.get('arguments', '')}{arguments}"
            )
        elif isinstance(arguments, Mapping):
            # Mapping arguments are unusual for a stream, but accepting them
            # keeps the final OpenAI-shaped response parseable for compatible
            # Z.ai gateways that decode JSON before handing it to us.
            existing = target_function.get("arguments", "")
            if not existing:
                target_function["arguments"] = dict(arguments)
            else:
                target_function["arguments"] = (
                    f"{existing}{json.dumps(dict(arguments), separators=(',', ':'))}"
                )


def consume_zai_sse_event(
    event: SSEEvent,
    state: ZaiStreamState,
) -> ZaiStreamDelta:
    """Decode one SSE event and update ``state``.

    Keep-alives, ``[DONE]``, malformed payloads, and non-primary choices are
    ignored.  Provider errors are raised by the HTTP client before this parser
    is called, preserving the transport error contract.
    """

    payload = event.data if isinstance(event.data, Mapping) else None
    if payload is None or event.done:
        return ZaiStreamDelta()

    for field_name in ("id", "model", "created"):
        if field_name in payload:
            state.response_metadata[field_name] = payload[field_name]

    usage = payload.get("usage")
    if isinstance(usage, Mapping):
        state.usage.update(dict(usage))

    choices = payload.get("choices")
    if not isinstance(choices, list):
        return ZaiStreamDelta()

    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    for choice in choices:
        if not isinstance(choice, Mapping) or choice.get("index", 0) != 0:
            continue

        delta = choice.get("delta")
        if isinstance(delta, Mapping):
            content = delta.get("content")
            if isinstance(content, str) and content:
                visible_parts.append(content)

            # Z.ai documents ``reasoning_content``; ``reasoning`` is accepted
            # for OpenAI-compatible gateways that use the shorter spelling.
            for field_name in ("reasoning_content", "reasoning"):
                reasoning = delta.get(field_name)
                if isinstance(reasoning, str) and reasoning:
                    reasoning_parts.append(reasoning)

            _accumulate_tool_calls(delta.get("tool_calls"), state.tool_calls)

        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str):
            state.finish_reason = finish_reason

    state.text_parts.extend(visible_parts)
    state.reasoning_parts.extend(reasoning_parts)
    return ZaiStreamDelta(
        visible_text="".join(visible_parts),
        reasoning_text="".join(reasoning_parts),
    )


def assemble_zai_response(
    state: ZaiStreamState,
    *,
    model: Optional[str] = None,
) -> dict[str, Any]:
    """Build a normalized OpenAI Chat Completions response from stream state."""

    response: dict[str, Any] = dict(state.response_metadata)
    response["model"] = response.get("model") or model
    if state.usage:
        response["usage"] = dict(state.usage)

    message: dict[str, Any] = {
        "role": "assistant",
        "content": state.visible_text or None,
    }
    if state.tool_calls:
        message["tool_calls"] = [
            _copy_tool_call(state.tool_calls[index])
            for index in sorted(state.tool_calls)
        ]

    response["choices"] = [
        {
            "index": 0,
            "message": message,
            "finish_reason": state.finish_reason,
        }
    ]
    return response


def _copy_tool_call(tool_call: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one accumulated call without exposing mutable parser state."""

    result: dict[str, Any] = {
        key: value for key, value in tool_call.items() if key != "function"
    }
    function = tool_call.get("function")
    result["function"] = (
        dict(function) if isinstance(function, Mapping) else {"arguments": ""}
    )
    return result


class ZaiStreamAssembler:
    """Convenience wrapper for incremental decoding and final assembly."""

    def __init__(self, *, model: Optional[str] = None) -> None:
        self.model = model
        self.state = ZaiStreamState()

    def consume(self, event: SSEEvent) -> ZaiStreamDelta:
        """Consume one event and return its visible/reasoning delta."""

        return consume_zai_sse_event(event, self.state)

    def finalize(self) -> dict[str, Any]:
        """Return the complete OpenAI-shaped response for this stream."""

        return assemble_zai_response(self.state, model=self.model)


__all__ = [
    "ZaiStreamAssembler",
    "ZaiStreamDelta",
    "ZaiStreamState",
    "assemble_zai_response",
    "consume_zai_sse_event",
]
