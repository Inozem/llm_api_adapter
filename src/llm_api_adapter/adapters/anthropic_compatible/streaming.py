"""Operation-local block reconstruction for decoded Messages SSE events."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any, Callable, Literal, Mapping

from ...errors.llm_api_error import InvalidToolArgumentsError, LLMAPIClientError
from ...llms.transports import SSEEvent


@dataclass
class MessagesBlockState:
    """Raw reconstructed state; repr deliberately omits provider content."""

    message: dict[str, Any] = field(default_factory=dict, repr=False)
    blocks: dict[int, dict[str, Any]] = field(default_factory=dict, repr=False)
    argument_fragments: dict[int, list[str]] = field(default_factory=dict, repr=False)
    signature_fragments: dict[int, list[str]] = field(default_factory=dict, repr=False)
    usage: Any = field(default=None, repr=False)
    usage_snapshots: list[tuple[str, Any]] = field(default_factory=list, repr=False)
    message_delta: dict[str, Any] = field(default_factory=dict, repr=False)
    started: bool = False
    message_stop: bool = False
    open_indexes: set[int] = field(default_factory=set)


EventPolicy = Callable[[SSEEvent], bool]
BlockPolicy = Callable[[int, dict[str, Any]], dict[str, Any]]
DeltaPolicy = Callable[[dict[str, Any], Mapping[str, Any]], bool]
IndexPolicy = Callable[[int, str, MessagesBlockState], bool | Literal["replace"]]


def _protocol_error(detail: str) -> LLMAPIClientError:
    return LLMAPIClientError(detail=detail)


def _record_usage(state: MessagesBlockState, source: str, value: Any) -> None:
    snapshot = deepcopy(value)
    state.usage_snapshots.append((source, snapshot))
    if isinstance(snapshot, Mapping):
        if isinstance(state.usage, Mapping):
            merged = deepcopy(dict(state.usage))
            merged.update(snapshot)
            state.usage = merged
        else:
            state.usage = deepcopy(dict(snapshot))
    else:
        state.usage = snapshot


def _require_mapping(value: Any, detail: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _protocol_error(detail)
    return value


def _event_kind(event: SSEEvent) -> Any:
    if event.event is not None:
        return event.event
    return event.data.get("type") if isinstance(event.data, Mapping) else None


def _valid_index(index: Any) -> bool:
    return isinstance(index, int) and not isinstance(index, bool)


def _check_index(
    state: MessagesBlockState,
    index: Any,
    kind: str,
    index_policy: IndexPolicy | None,
) -> bool:
    # bool is never a protocol index, even when a caller supplies a tolerance hook.
    if not _valid_index(index):
        raise _protocol_error("Messages content block index is invalid")
    if kind == "content_block_start":
        if index in state.blocks:
            decision = index_policy(index, kind, state) if index_policy is not None else True
            if decision == "replace":
                state.open_indexes.discard(index)
                state.blocks.pop(index, None)
            elif not decision:
                return False
            else:
                raise _protocol_error("Messages content block index was started more than once")
        return True
    if index not in state.blocks:
        if index_policy is not None and not index_policy(index, kind, state):
            return False
        raise _protocol_error("Messages content block index has no matching start")
    if index not in state.open_indexes:
        if index_policy is None:
            raise _protocol_error("Messages content block is closed")
        if not index_policy(index, kind, state):
            return False
    return True


def _prepare_block(
    index: int,
    block: dict[str, Any],
    block_policy: BlockPolicy | None,
) -> dict[str, Any]:
    prepared = block_policy(index, block) if block_policy is not None else block
    if not isinstance(prepared, dict):
        raise _protocol_error("Messages content block is malformed")
    kind = prepared.get("type")
    if kind == "text" and not isinstance(prepared.get("text"), str):
        raise _protocol_error("Messages text block is malformed")
    if kind == "thinking" and not isinstance(prepared.get("thinking"), str):
        raise _protocol_error("Messages thinking block is malformed")
    return prepared


def consume_messages_event(
    state: MessagesBlockState,
    event: SSEEvent,
    *,
    event_policy: EventPolicy | None = None,
    block_policy: BlockPolicy | None = None,
    delta_policy: DeltaPolicy | None = None,
    index_policy: IndexPolicy | None = None,
) -> tuple[str, str]:
    """Apply one decoded event and return (visible text, thinking) deltas.

    Policy callbacks are small parser seams: an event hook may skip malformed
    nonterminal input, a block hook may apply a characterized block transform,
    a delta hook may skip or admit a characterized delta, and an index hook may
    ignore unmatched references or replace a duplicate start. They do not
    create another accumulation path.
    """
    if not isinstance(event, SSEEvent):
        raise _protocol_error("Messages event is malformed")
    if event_policy is not None and not event_policy(event):
        return "", ""
    kind = _event_kind(event)
    if kind not in {
        "message_start", "content_block_start", "content_block_delta",
        "content_block_stop", "message_delta", "message_stop",
    }:
        return "", ""
    payload = _require_mapping(event.data, "Messages event data is malformed")
    visible = thinking = ""
    if kind == "message_start":
        message = _require_mapping(payload.get("message"), "Messages start envelope is malformed")
        state.message = deepcopy(dict(message))
        state.started = True
        if "usage" in message:
            _record_usage(state, "message_start", message["usage"])
    elif kind == "content_block_start":
        index = payload.get("index")
        if not _check_index(state, index, kind, index_policy):
            return "", ""
        block = _require_mapping(payload.get("content_block"), "Messages content block is malformed")
        prepared = _prepare_block(index, deepcopy(dict(block)), block_policy)
        state.blocks[index] = prepared
        state.open_indexes.add(index)
    elif kind == "content_block_delta":
        index = payload.get("index")
        if not _check_index(state, index, kind, index_policy):
            return "", ""
        delta = _require_mapping(payload.get("delta"), "Messages content delta is malformed")
        block = state.blocks[index]
        delta_type = delta.get("type")
        if not isinstance(delta_type, str):
            if delta_policy is not None and not delta_policy(block, delta):
                return "", ""
            raise _protocol_error("Messages content delta is malformed")
        compatible = (
            delta_type == "text_delta" and block.get("type") == "text"
            or delta_type == "thinking_delta" and block.get("type") == "thinking"
            or delta_type in {"signature_delta", "input_json_delta"} and block.get("type") == "tool_use"
            or delta_type == "signature_delta" and block.get("type") == "thinking"
        )
        if delta_policy is not None:
            admitted = delta_policy(block, delta)
        else:
            admitted = True
        if not compatible and not admitted:
            return "", ""
        if not compatible and delta_policy is None:
            raise _protocol_error("Messages content delta does not match its block")
        if not admitted:
            return "", ""
        if delta_type == "text_delta":
            if not isinstance(delta.get("text"), str):
                raise _protocol_error("Messages text delta is malformed")
            visible = delta["text"]
            block["text"] = f"{block.get('text', '')}{visible}"
        elif delta_type == "thinking_delta":
            if not isinstance(delta.get("thinking"), str):
                raise _protocol_error("Messages thinking delta is malformed")
            thinking = delta["thinking"]
            block["thinking"] = f"{block.get('thinking', '')}{thinking}"
        elif delta_type == "signature_delta":
            if not isinstance(delta.get("signature"), str):
                raise _protocol_error("Messages signature delta is malformed")
            state.signature_fragments.setdefault(index, []).append(delta["signature"])
        elif delta_type == "input_json_delta":
            if not isinstance(delta.get("partial_json"), str):
                raise _protocol_error("Messages tool argument delta is malformed")
            state.argument_fragments.setdefault(index, []).append(delta["partial_json"])
    elif kind == "content_block_stop":
        index = payload.get("index")
        if not _check_index(state, index, kind, index_policy):
            return "", ""
        block = state.blocks[index]
        if block.get("type") == "tool_use":
            fragments = state.argument_fragments.get(index, [])
            if fragments:
                raw = "".join(fragments)
                try:
                    arguments = json.loads(raw)
                except (ValueError, TypeError) as exc:
                    raise InvalidToolArgumentsError(detail="Messages tool arguments are invalid JSON") from exc
                if not isinstance(arguments, dict):
                    raise InvalidToolArgumentsError(detail="Messages tool arguments must be a JSON object")
                block["input"] = arguments
            elif not isinstance(block.get("input"), dict):
                raise InvalidToolArgumentsError(detail="Messages tool input must be an object")
        signature = "".join(state.signature_fragments.get(index, []))
        if signature:
            block["signature"] = signature
        state.open_indexes.discard(index)
    elif kind == "message_delta":
        delta = _require_mapping(payload.get("delta"), "Messages terminal delta is malformed")
        state.message_delta.update(deepcopy(dict(delta)))
        if "usage" in payload:
            _record_usage(state, "message_delta", payload["usage"])
    elif kind == "message_stop":
        state.message_stop = True
    return visible, thinking


def finalize_messages_stream(
    state: MessagesBlockState, *,
    validate_completion: Callable[[MessagesBlockState], None] | None = None,
) -> dict[str, Any]:
    """Build the common response envelope; completion rules remain overridable."""
    if validate_completion is not None:
        validate_completion(state)
    else:
        valid_envelope = (
            state.started and state.message.get("type") == "message"
            and state.message.get("role") == "assistant"
            and isinstance(state.message.get("id"), str)
            and bool(state.message.get("id"))
            and isinstance(state.message.get("model"), str)
            and bool(state.message.get("model"))
        )
        stop_reason = state.message_delta.get("stop_reason")
        if not valid_envelope or not isinstance(stop_reason, str) or not stop_reason or not state.message_stop:
            raise _protocol_error("Messages stream ended without complete terminal evidence")
        if state.open_indexes:
            raise _protocol_error("Messages stream ended with open content blocks")
    response = deepcopy(state.message)
    response["content"] = [deepcopy(state.blocks[index]) for index in sorted(state.blocks)]
    response.update(deepcopy(state.message_delta))
    if state.usage_snapshots:
        response["usage"] = deepcopy(state.usage)
    return response


__all__ = ["MessagesBlockState", "consume_messages_event", "finalize_messages_stream"]
