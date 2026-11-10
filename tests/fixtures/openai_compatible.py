"""Credential-free Chat Completions and Responses JSON/SSE fixtures.

These deterministic factories describe protocol data only. They make no
network requests and do not imply support by any particular server or model.
Mutable caller input and shared portable schemas are copied before use.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from .structured_output import (
    INVALID_JSON_CONTENT,
    NESTED_PYDANTIC_RESPONSE_JSON,
    PORTABLE_PROFILE_SCHEMAS,
    STRUCTURED_OUTPUT_OUTCOMES,
)


FIXTURE_ID = "chatcmpl-fixture-001"
FIXTURE_MODEL = "fixture-model"
FIXTURE_CREATED = 1_700_000_000
_ABSENT = object()


def usage_payload(
    prompt_tokens: Any = _ABSENT,
    completion_tokens: Any = _ABSENT,
    total_tokens: Any = _ABSENT,
    **provider_fields: Any,
) -> dict[str, Any]:
    """Return raw usage counters, preserving omitted keys and arbitrary detail."""
    result: dict[str, Any] = {}
    for key, value in (
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
        ("total_tokens", total_tokens),
    ):
        if value is not _ABSENT:
            result[key] = deepcopy(value)
    result.update(deepcopy(provider_fields))
    return result


def partial_usage_payload(**counters: Any) -> dict[str, Any]:
    """Return an intentionally partial raw usage object."""
    return usage_payload(**counters)


def zero_usage_payload() -> dict[str, int]:
    """Return explicitly reported zero counters."""
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def invalid_usage_payload(counter: Any = -1, *, field: str = "prompt_tokens") -> dict[str, Any]:
    """Return a raw usage object with a boolean, negative, or malformed count."""
    return {field: deepcopy(counter)}


def portable_schema(name: str = "flat_object") -> dict[str, Any]:
    """Return an independent copy of one existing portable-profile schema."""
    try:
        return deepcopy(PORTABLE_PROFILE_SCHEMAS[name])
    except KeyError as exc:
        raise ValueError(f"unknown portable schema fixture: {name}") from exc


def chat_completion_response(
    message: dict[str, Any] | None = None,
    *,
    content: str | None = "Fixture reply",
    finish_reason: str = "stop",
    usage: Any = _ABSENT,
    response_id: str = FIXTURE_ID,
    model: str = FIXTURE_MODEL,
    created: int = FIXTURE_CREATED,
    **response_fields: Any,
) -> dict[str, Any]:
    """Build one fresh raw Chat Completions response.

    Pass ``usage=None`` for explicit null; omit it for a missing usage key.
    A supplied message replaces the default assistant message.
    """
    result: dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": deepcopy(message) if message is not None else {
                "role": "assistant",
                "content": content,
            },
            "finish_reason": finish_reason,
        }],
    }
    if usage is not _ABSENT:
        result["usage"] = deepcopy(usage)
    result.update(deepcopy(response_fields))
    return result


def text_response(content: str = "Fixture reply", **kwargs: Any) -> dict[str, Any]:
    """Build a completed visible-text response."""
    return chat_completion_response(content=content, **kwargs)


def responses_text_response(
    content: str = "Fixture reply",
    *,
    response_id: str = "resp-fixture-001",
    model: str = FIXTURE_MODEL,
) -> dict[str, Any]:
    """Build a minimal completed Responses envelope containing visible text."""
    return {
        "id": response_id,
        "object": "response",
        "status": "completed",
        "model": model,
        "output": [{
            "id": "msg-fixture-001",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content, "annotations": []}],
        }],
    }


def responses_response(
    output: list[Any] | None = None,
    *,
    status: str = "completed",
    response_id: str = "resp-fixture-001",
    model: str = FIXTURE_MODEL,
    **fields: Any,
) -> dict[str, Any]:
    """Build a fresh, minimal Responses envelope with usage omitted by default."""
    result: dict[str, Any] = {
        "id": response_id,
        "object": "response",
        "status": status,
        "model": model,
        "output": deepcopy(output) if output is not None else [],
    }
    result.update(deepcopy(fields))
    return result


def responses_empty_response(**kwargs: Any) -> dict[str, Any]:
    """Build a valid empty completed Responses result."""
    return responses_response(output=[], status="completed", **kwargs)


def responses_refusal_response(refusal: str = "Fixture refusal", **kwargs: Any) -> dict[str, Any]:
    """Build a completed Responses result containing an assistant refusal."""
    output = [{
        "id": "msg-fixture-001",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "refusal", "refusal": refusal}],
    }]
    return responses_response(output=output, status="completed", **kwargs)


def responses_tool_call_response(
    name: str = "lookup",
    arguments_json: str = '{"query":"fixture"}',
    *,
    call_id: str = "call_fixture_001",
    item_id: str = "fc-fixture-001",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a completed Responses function call with raw JSON arguments."""
    return responses_response(
        output=[{
            "id": item_id,
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments_json,
            "status": "completed",
        }],
        **kwargs,
    )


def responses_incomplete_response(
    content: str = "Partial fixture answer",
    *,
    reason: str = "max_output_tokens",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build an incomplete Responses result with a visible partial answer."""
    output = [{
        "id": "msg-fixture-001",
        "type": "message",
        "status": "incomplete",
        "role": "assistant",
        "content": [{"type": "output_text", "text": content, "annotations": []}],
    }]
    return responses_response(
        output=output,
        status="incomplete",
        incomplete_details={"reason": reason},
        **kwargs,
    )


def tool_call_response(
    name: str = "lookup",
    arguments_json: str = '{"query":"fixture"}',
    *,
    call_id: str = "call_fixture_001",
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a completed tool-only response with raw JSON argument text."""
    return chat_completion_response(
        message={"role": "assistant", "content": None, "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments_json},
        }]},
        finish_reason="tool_calls",
        **kwargs,
    )


def structured_result_response(content: str | None = None, **kwargs: Any) -> dict[str, Any]:
    """Build a completed JSON-text result using the shared portable fixture."""
    if content is None:
        outcome = next(
            item for item in STRUCTURED_OUTPUT_OUTCOMES
            if item.name == "valid_structured_result"
        )
        content = outcome.content or "{}"
    return text_response(content, **kwargs)


def refusal_response(refusal: str = "Fixture refusal", **kwargs: Any) -> dict[str, Any]:
    """Build a refusal outcome distinct from a completed structured result."""
    return chat_completion_response(
        message={"role": "assistant", "content": None, "refusal": refusal},
        **kwargs,
    )


def incomplete_response(content: str = "Partial fixture answer", **kwargs: Any) -> dict[str, Any]:
    """Build an incomplete length-limited outcome."""
    return chat_completion_response(content=content, finish_reason="length", **kwargs)


def invalid_structured_result_response(**kwargs: Any) -> dict[str, Any]:
    """Build malformed completed JSON from the shared structured-output fixtures."""
    outcome = next(item for item in STRUCTURED_OUTPUT_OUTCOMES if item.name == "invalid_json")
    return structured_result_response(outcome.content or INVALID_JSON_CONTENT, **kwargs)


def model_invalid_structured_result_response(**kwargs: Any) -> dict[str, Any]:
    """Build valid JSON that fails the shared nested response-model fixture."""
    outcome = next(
        item for item in STRUCTURED_OUTPUT_OUTCOMES
        if item.name == "pydantic_validation_error"
    )
    return structured_result_response(outcome.content or NESTED_PYDANTIC_RESPONSE_JSON, **kwargs)


def event_lines(payload: Any) -> list[str]:
    """Encode one JSON SSE event as lines accepted by the repository decoder."""
    return [f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}", ""]


def responses_text_sse_lines(
    content: str = "Fixture reply",
    *,
    fragments: list[str] | tuple[str, ...] | None = None,
    terminal: str = "completed",
    model: str = FIXTURE_MODEL,
) -> list[str]:
    """Build Responses text deltas and one terminal event with valid delimiters."""
    pieces = [] if terminal in ("empty", "refusal") else _fragments(content, fragments)
    lines: list[str] = []
    for index, piece in enumerate(pieces):
        lines.extend(event_lines({
            "type": "response.output_text.delta",
            "delta": piece,
            "item_id": "msg-fixture-001",
            "output_index": 0,
            "content_index": 0,
            "sequence_number": index,
        }))
    if terminal == "completed":
        response = responses_text_response(content, model=model)
        event_type = "response.completed"
    elif terminal == "incomplete":
        response = responses_incomplete_response(content, model=model)
        event_type = "response.incomplete"
    elif terminal == "empty":
        response = responses_empty_response(model=model)
        event_type = "response.completed"
    elif terminal == "refusal":
        response = responses_refusal_response(model=model)
        event_type = "response.completed"
    else:
        raise ValueError("terminal must be completed, incomplete, empty, or refusal")
    lines.extend(event_lines({
        "type": event_type,
        "response": response,
        "sequence_number": len(pieces),
    }))
    return lines


def responses_tool_call_sse_lines(
    name: str = "lookup",
    arguments_json: str = '{"query":"fixture"}',
    *,
    argument_fragments: list[str] | tuple[str, ...] | None = None,
    call_id: str = "call_fixture_001",
    item_id: str = "fc-fixture-001",
    model: str = FIXTURE_MODEL,
) -> list[str]:
    """Build fragmented function-call argument deltas and a completed result."""
    arguments = _fragments(arguments_json, argument_fragments)
    lines: list[str] = []
    count = max(len(arguments), 1)
    for index in range(count):
        payload: dict[str, Any] = {
            "type": "response.function_call_arguments.delta",
            "call_id": call_id,
            "item_id": item_id,
            "output_index": 0,
            "delta": arguments[index] if index < len(arguments) else "",
            "sequence_number": index,
        }
        lines.extend(event_lines(payload))
    lines.extend(event_lines({
        "type": "response.completed",
        "response": responses_tool_call_response(
            name=name,
            arguments_json=arguments_json,
            call_id=call_id,
            item_id=item_id,
            model=model,
        ),
        "sequence_number": count,
    }))
    return lines


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": FIXTURE_ID,
        "object": "chat.completion.chunk",
        "created": FIXTURE_CREATED,
        "model": FIXTURE_MODEL,
        "choices": [{"index": 0, "delta": deepcopy(delta), "finish_reason": finish_reason}],
    }


def _fragments(value: str, fragments: list[str] | tuple[str, ...] | None) -> list[str]:
    if fragments is None:
        return [value] if value else []
    pieces = list(fragments)
    if "" in pieces or "".join(pieces) != value:
        raise ValueError("fragments must be non-empty pieces that concatenate to the value")
    return pieces


def _terminal_lines(finish_reason: str, usage: Any) -> list[str]:
    lines = event_lines(_chunk({}, finish_reason))
    if usage is not _ABSENT:
        lines.extend(event_lines({
            "id": FIXTURE_ID,
            "object": "chat.completion.chunk",
            "created": FIXTURE_CREATED,
            "model": FIXTURE_MODEL,
            "choices": [],
            "usage": deepcopy(usage),
        }))
    lines.extend(["data: [DONE]", ""])
    return lines


def text_sse_lines(
    content: str = "Fixture reply",
    *,
    fragments: list[str] | tuple[str, ...] | None = None,
    finish_reason: str = "stop",
    usage: Any = _ABSENT,
) -> list[str]:
    """Build complete Chat Completions SSE wire lines with content fragments."""
    lines: list[str] = []
    for piece in _fragments(content, fragments):
        lines.extend(event_lines(_chunk({"content": piece})))
    lines.extend(_terminal_lines(finish_reason, usage))
    return lines


def tool_call_sse_lines(
    name: str = "lookup",
    arguments_json: str = '{"query":"fixture"}',
    *,
    name_fragments: list[str] | tuple[str, ...] | None = None,
    argument_fragments: list[str] | tuple[str, ...] | None = None,
    call_id: str = "call_fixture_001",
    usage: Any = _ABSENT,
) -> list[str]:
    """Build complete SSE lines with a tool name and JSON arguments fragmented."""
    names = _fragments(name, name_fragments)
    arguments = _fragments(arguments_json, argument_fragments)
    lines: list[str] = []
    count = max(len(names), len(arguments), 1)
    for index in range(count):
        function: dict[str, str] = {}
        if index < len(names):
            function["name"] = names[index]
        if index < len(arguments):
            function["arguments"] = arguments[index]
        call: dict[str, Any] = {"index": 0, "function": function}
        if index == 0:
            call.update({"id": call_id, "type": "function"})
        lines.extend(event_lines(_chunk({"tool_calls": [call]})))
    lines.extend(_terminal_lines("tool_calls", usage))
    return lines


def structured_result_sse_lines(
    content: str | None = None, **kwargs: Any
) -> list[str]:
    """Build a fragmented JSON-text stream using the shared flat-schema fixture."""
    if content is None:
        outcome = next(
            item for item in STRUCTURED_OUTPUT_OUTCOMES
            if item.name == "valid_structured_result"
        )
        content = outcome.content or "{}"
    return text_sse_lines(content, **kwargs)


def refusal_sse_lines(refusal: str = "Fixture refusal") -> list[str]:
    """Build a terminal refusal stream without treating it as validated JSON."""
    return event_lines(_chunk({"refusal": refusal})) + _terminal_lines("stop", _ABSENT)


def incomplete_sse_lines(content: str = "Partial fixture answer") -> list[str]:
    """Build a length-limited stream with an explicit terminal reason."""
    return text_sse_lines(content, finish_reason="length")


def premature_eof_sse_lines(
    content: str = "Partial fixture",
    *,
    content_fragments: list[str] | tuple[str, ...] | None = None,
    tool_name: str | None = None,
    tool_arguments_fragment: str = "{\"partial\":",
) -> list[str]:
    """Build partial SSE data without a finish_reason, usage event, or ``[DONE]``."""
    lines: list[str] = []
    for piece in _fragments(content, content_fragments):
        lines.extend(event_lines(_chunk({"content": piece})))
    if tool_name is not None:
        lines.extend(event_lines(_chunk({"tool_calls": [{
            "index": 0,
            "id": "call_fixture_001",
            "type": "function",
            "function": {"name": tool_name, "arguments": tool_arguments_fragment},
        }]})))
    return lines


__all__ = [
    "FIXTURE_CREATED", "FIXTURE_ID", "FIXTURE_MODEL", "chat_completion_response",
    "event_lines", "incomplete_response", "invalid_structured_result_response",
    "incomplete_sse_lines",
    "invalid_usage_payload", "model_invalid_structured_result_response",
    "partial_usage_payload", "portable_schema", "premature_eof_sse_lines", "refusal_response",
    "refusal_sse_lines",
    "responses_empty_response", "responses_incomplete_response", "responses_refusal_response",
    "responses_response", "responses_text_response", "responses_text_sse_lines",
    "responses_tool_call_response", "responses_tool_call_sse_lines",
    "structured_result_response", "structured_result_sse_lines",
    "text_response",
    "text_sse_lines", "tool_call_response", "tool_call_sse_lines", "usage_payload",
    "zero_usage_payload",
]
