"""Synthetic Anthropic Messages payloads, event traces, and source doubles.

The factories provide provider-shaped raw data for deterministic tests only.
They do not validate or repair protocol data, infer usage, or perform I/O.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, AsyncIterator, Iterable, Iterator, Sequence

from src.llm_api_adapter.llms.transports import SSEEvent


FIXTURE_ID = "msg_fixture_001"
FIXTURE_MODEL = "fixture-messages-model"
_ABSENT = object()


def usage_payload(
    input_tokens: Any = _ABSENT,
    output_tokens: Any = _ABSENT,
    **provider_fields: Any,
) -> dict[str, Any]:
    """Build raw Messages usage, preserving omitted, null, and arbitrary fields."""
    result: dict[str, Any] = {}
    if input_tokens is not _ABSENT:
        result["input_tokens"] = deepcopy(input_tokens)
    if output_tokens is not _ABSENT:
        result["output_tokens"] = deepcopy(output_tokens)
    result.update(deepcopy(provider_fields))
    return result


def messages_response(
    content: Sequence[dict[str, Any]] | None = None,
    *,
    text: str = "Fixture reply",
    stop_reason: Any = "end_turn",
    usage: Any = _ABSENT,
    response_id: str = FIXTURE_ID,
    model: str = FIXTURE_MODEL,
    **message_fields: Any,
) -> dict[str, Any]:
    """Return a fresh raw Messages envelope with caller-controlled content."""
    result: dict[str, Any] = {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": deepcopy(list(content)) if content is not None else [
            {"type": "text", "text": text}
        ],
        "stop_reason": deepcopy(stop_reason),
        "stop_sequence": None,
    }
    if usage is not _ABSENT:
        result["usage"] = deepcopy(usage)
    result.update(deepcopy(message_fields))
    return result


def text_response(text: str = "Fixture reply", **kwargs: Any) -> dict[str, Any]:
    return messages_response(text=text, **kwargs)


def multiple_blocks_response(
    blocks: Sequence[dict[str, Any]] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Return visible text, tool-use, and reasoning blocks in caller order."""
    if blocks is None:
        blocks = [
            {"type": "text", "text": "Visible fixture text"},
            {"type": "tool_use", "id": "toolu_fixture_1", "name": "lookup", "input": {"query": "fixture"}},
            {"type": "thinking", "thinking": "Synthetic reasoning"},
        ]
    return messages_response(content=blocks, **kwargs)


def tool_response(
    *,
    tool_id: str = "toolu_fixture_1",
    name: str = "lookup",
    arguments: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    block = {
        "type": "tool_use", "id": tool_id, "name": name,
        "input": deepcopy(arguments) if arguments is not None else {"query": "fixture"},
    }
    return messages_response(content=[block], stop_reason="tool_use", **kwargs)


def refusal_response(refusal: str = "Synthetic refusal", **kwargs: Any) -> dict[str, Any]:
    return messages_response(
        content=[], stop_reason="refusal", stop_details={"reason": refusal},
        **kwargs,
    )


def incomplete_response(text: str = "Partial fixture answer", **kwargs: Any) -> dict[str, Any]:
    return messages_response(
        text=text, stop_reason="max_tokens", **kwargs
    )


def malformed_response(**fields: Any) -> dict[str, Any]:
    """Return intentionally unvalidated raw message fields (including malformed envelopes)."""
    result: dict[str, Any] = {"id": FIXTURE_ID, "type": "message", "role": "assistant"}
    result.update(deepcopy(fields))
    return result


def _default_blocks() -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": "Fixture reply"},
        {"type": "tool_use", "id": "toolu_fixture_1", "name": "lookup", "input": {"query": "fixture"}},
        {"type": "thinking", "thinking": "Synthetic reasoning"},
    ]


def messages_event_trace(
    blocks: Sequence[dict[str, Any]] | None = None,
    *,
    text_fragments: dict[int, Sequence[str]] | None = None,
    tool_argument_fragments: dict[int, Sequence[str]] | None = None,
    usage_start: Any = _ABSENT,
    usage_delta: Any = _ABSENT,
    include_start_usage: bool = True,
    stop_reason: Any = "end_turn",
    include_stop_reason: bool = True,
    include_message_stop: bool = True,
    close_last_block: bool = True,
    response_id: str = FIXTURE_ID,
    model: str = FIXTURE_MODEL,
) -> list[SSEEvent]:
    """Build raw ordered Messages events; fragment controls use block indexes.

    Text and thinking are emitted as their matching delta types. Tool arguments
    are JSON-encoded only for the default complete trace; supplied fragments are
    preserved verbatim, including malformed or truncated JSON. Message-start
    usage is omitted by default; start and late usage are independently caller
    controlled and are copied without filling missing counters.
    """
    content = deepcopy(list(blocks) if blocks is not None else _default_blocks())
    start_usage = deepcopy(usage_start) if usage_start is not _ABSENT else _ABSENT
    message = {
        "id": response_id, "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None,
    }
    if include_start_usage:
        if start_usage is not _ABSENT:
            message["usage"] = start_usage
    events = [SSEEvent("message_start", {"type": "message_start", "message": message})]
    for index, block in enumerate(content):
        block_type = block.get("type") if isinstance(block, dict) else None
        start_block = deepcopy(block)
        if block_type in {"text", "thinking"}:
            start_block["text" if block_type == "text" else "thinking"] = ""
        elif block_type == "tool_use":
            start_block["input"] = {}
        events.append(SSEEvent("content_block_start", {
            "type": "content_block_start", "index": index, "content_block": start_block,
        }))

        if block_type == "text":
            fragments = (text_fragments or {}).get(index, [block.get("text", "")])
            for fragment in fragments:
                events.append(SSEEvent("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "text_delta", "text": deepcopy(fragment)},
                }))
        elif block_type == "thinking":
            fragments = (text_fragments or {}).get(index, [block.get("thinking", "")])
            for fragment in fragments:
                events.append(SSEEvent("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "thinking_delta", "thinking": deepcopy(fragment)},
                }))
            if "signature" in block:
                events.append(SSEEvent("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "signature_delta", "signature": deepcopy(block["signature"])},
                }))
        elif block_type == "tool_use":
            fragments = (tool_argument_fragments or {}).get(index)
            if fragments is None:
                encoded = json.dumps(block.get("input", {}), ensure_ascii=False, separators=(",", ":"))
                midpoint = max(1, len(encoded) // 2)
                fragments = [encoded[:midpoint], encoded[midpoint:]]
            for fragment in fragments:
                events.append(SSEEvent("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta", "partial_json": deepcopy(fragment)},
                }))

        if close_last_block or index < len(content) - 1:
            events.append(SSEEvent("content_block_stop", {
                "type": "content_block_stop", "index": index,
            }))

    delta: dict[str, Any] = {"type": "message_delta", "delta": {}}
    if include_stop_reason:
        delta["delta"]["stop_reason"] = deepcopy(stop_reason)
    delta["delta"]["stop_sequence"] = None
    if usage_delta is not _ABSENT:
        delta["usage"] = deepcopy(usage_delta)
    events.append(SSEEvent("message_delta", delta))
    if include_message_stop:
        events.append(SSEEvent("message_stop", {"type": "message_stop"}))
    return events


def malformed_terminal_events() -> list[SSEEvent]:
    """A trace whose terminal event has a deliberately malformed data shape."""
    return messages_event_trace()[:-1] + [SSEEvent("message_stop", "malformed terminal")]


def premature_eof_trace(
    *,
    text_fragment: str = "Partial fixture text",
    tool_id: str = "toolu_partial",
    tool_name: str = "lookup",
    partial_json: str = '{"unfinished":',
    response_id: str = FIXTURE_ID,
    model: str = FIXTURE_MODEL,
) -> list[SSEEvent]:
    """Return a raw partial prefix ending inside a tool block before any terminal."""
    message = {
        "id": response_id, "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None, "stop_sequence": None,
    }
    return [
        SSEEvent("message_start", {"type": "message_start", "message": message}),
        SSEEvent("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""},
        }),
        SSEEvent("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": deepcopy(text_fragment)},
        }),
        SSEEvent("content_block_stop", {"type": "content_block_stop", "index": 0}),
        SSEEvent("content_block_start", {
            "type": "content_block_start", "index": 1,
            "content_block": {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}},
        }),
        SSEEvent("content_block_delta", {
            "type": "content_block_delta", "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": deepcopy(partial_json)},
        }),
    ]


def transport_done_events() -> list[SSEEvent]:
    """A transport sentinel without any Messages terminal evidence."""
    return [SSEEvent(None, done=True)]


def unknown_and_ping_events() -> list[SSEEvent]:
    return [
        SSEEvent("ping", {"type": "ping"}),
        SSEEvent("future_fixture_event", {"type": "future_fixture_event"}),
    ]


def events_to_sse_lines(events: Iterable[SSEEvent]) -> list[str]:
    """Encode already-decoded events as independent SSE wire lines."""
    lines: list[str] = []
    for event in events:
        if event.event is not None:
            lines.append(f"event: {event.event}")
        if event.done:
            lines.append("data: [DONE]")
        else:
            lines.append("data: " + json.dumps(event.data, ensure_ascii=False, separators=(",", ":")))
        lines.append("")
    return lines


class ControlledSyncSource(Iterator[Any]):
    """Iterator with deterministic failure boundary and explicit close tracking."""

    def __init__(
        self, values: Iterable[Any] = (), *, failure: BaseException | None = None,
        fail_after: int = 0, cleanup_error: BaseException | None = None,
    ) -> None:
        self._values = iter(values)
        self.failure = failure
        self.fail_after = fail_after
        self.cleanup_error = cleanup_error
        self.yielded = 0
        self.close_calls = 0

    def __iter__(self) -> "ControlledSyncSource":
        return self

    def __next__(self) -> Any:
        if self.failure is not None and self.yielded >= self.fail_after:
            failure, self.failure = self.failure, None
            raise failure
        value = next(self._values)
        self.yielded += 1
        return value

    def close(self) -> None:
        self.close_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


class ControlledAsyncSource(AsyncIterator[Any]):
    """Async iterator matching ControlledSyncSource's explicit ownership semantics."""

    def __init__(
        self, values: Iterable[Any] = (), *, failure: BaseException | None = None,
        fail_after: int = 0, cleanup_error: BaseException | None = None,
    ) -> None:
        self._values = iter(values)
        self.failure = failure
        self.fail_after = fail_after
        self.cleanup_error = cleanup_error
        self.yielded = 0
        self.aclose_calls = 0

    def __aiter__(self) -> "ControlledAsyncSource":
        return self

    async def __anext__(self) -> Any:
        if self.failure is not None and self.yielded >= self.fail_after:
            failure, self.failure = self.failure, None
            raise failure
        try:
            value = next(self._values)
        except StopIteration:
            raise StopAsyncIteration from None
        self.yielded += 1
        return value

    async def aclose(self) -> None:
        self.aclose_calls += 1
        if self.cleanup_error is not None:
            raise self.cleanup_error


class ControlledCallback:
    """Callable recording inputs and raising the supplied error on a chosen call."""

    def __init__(self, failure: BaseException, *, fail_on_call: int = 1) -> None:
        self.failure = failure
        self.fail_on_call = fail_on_call
        self.calls: list[Any] = []

    def __call__(self, value: Any) -> None:
        self.calls.append(value)
        if len(self.calls) == self.fail_on_call:
            raise self.failure


class ControlledAsyncCallback(ControlledCallback):
    async def __call__(self, value: Any) -> None:
        super().__call__(value)


__all__ = [
    "FIXTURE_ID", "FIXTURE_MODEL", "ControlledAsyncCallback", "ControlledAsyncSource",
    "ControlledCallback", "ControlledSyncSource", "events_to_sse_lines",
    "incomplete_response", "malformed_response", "malformed_terminal_events",
    "messages_event_trace", "messages_response", "multiple_blocks_response",
    "premature_eof_trace", "refusal_response", "text_response", "tool_response", "transport_done_events",
    "unknown_and_ping_events", "usage_payload",
]
