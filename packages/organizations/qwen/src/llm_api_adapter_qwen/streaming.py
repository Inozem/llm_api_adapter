"""Qwen Messages SSE normalization without transport or callback ownership."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping, Optional

from llm_api_adapter.adapters.base_adapter import _StreamState
from llm_api_adapter.errors.llm_api_error import InvalidToolArgumentsError
from llm_api_adapter.llms.streaming import StreamChunkBuffer, StreamUsageTracker
from llm_api_adapter.llms.transports import SSEEvent
from llm_api_adapter.models.responses.chat_response import Usage


@dataclass
class QwenMessagesStreamState(_StreamState):
    """State needed to reconstruct one Messages response from SSE events."""

    message_data: dict[str, Any] = field(default_factory=dict)
    content_blocks: dict[int, dict[str, Any]] = field(default_factory=dict)
    input_json_fragments: dict[int, list[str]] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    message_delta: dict[str, Any] = field(default_factory=dict)


class QwenMessagesStreamParser:
    """Accumulate the documented Anthropic-compatible Qwen SSE event trace."""

    @staticmethod
    def new_state(*, buffer_chars: Optional[int]) -> QwenMessagesStreamState:
        """Create fresh shared lifecycle state for one Qwen stream."""
        return QwenMessagesStreamState(
            message_data={"content": []},
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=None,
            reasoning_response=None,
        )

    @classmethod
    def consume_event(
        cls,
        event: SSEEvent,
        state: QwenMessagesStreamState,
    ) -> Optional[str]:
        """Record one event and return its visible text delta, when present."""
        payload = event.data if isinstance(event.data, Mapping) else {}
        event_type = event.event or payload.get("type")
        visible_delta: Optional[str] = None

        if event_type == "message_start":
            cls._record_message_start(payload, state)
        elif event_type == "content_block_start":
            cls._record_content_block_start(payload, state)
        elif event_type == "content_block_delta":
            visible_delta = cls._record_content_block_delta(payload, state)
        elif event_type == "content_block_stop":
            cls._finalize_content_block(payload, state)
        elif event_type == "message_delta":
            cls._record_message_delta(payload, state)

        state.usage_tracker.record(
            state.chunk_buffer,
            cls._normalize_usage(state.usage),
        )
        return visible_delta

    @staticmethod
    def finalize(
        state: QwenMessagesStreamState,
        *,
        model: str,
    ) -> dict[str, Any]:
        """Rebuild an Anthropic-compatible terminal message response."""
        response = dict(state.message_data)
        response["model"] = response.get("model") or model
        response["content"] = [
            state.content_blocks[index] for index in sorted(state.content_blocks)
        ]
        response.update(state.message_delta)
        if state.usage:
            response["usage"] = dict(state.usage)
        return response

    @staticmethod
    def _record_message_start(
        payload: Mapping[str, Any],
        state: QwenMessagesStreamState,
    ) -> None:
        message = payload.get("message")
        if not isinstance(message, Mapping):
            return
        state.message_data = dict(message)
        usage = message.get("usage")
        if isinstance(usage, Mapping):
            state.usage.update(usage)

    @staticmethod
    def _record_content_block_start(
        payload: Mapping[str, Any],
        state: QwenMessagesStreamState,
    ) -> None:
        index = payload.get("index")
        block = payload.get("content_block")
        if isinstance(index, bool) or not isinstance(index, int):
            return
        if not isinstance(block, Mapping):
            return
        copied_block = dict(block)
        if copied_block.get("type") == "text":
            text = copied_block.get("text")
            copied_block["text"] = text if isinstance(text, str) else ""
        elif copied_block.get("type") == "tool_use":
            current_input = copied_block.get("input")
            copied_block["input"] = (
                dict(current_input) if isinstance(current_input, Mapping) else {}
            )
        state.content_blocks[index] = copied_block

    @staticmethod
    def _record_content_block_delta(
        payload: Mapping[str, Any],
        state: QwenMessagesStreamState,
    ) -> Optional[str]:
        index = payload.get("index")
        delta = payload.get("delta")
        if isinstance(index, bool) or not isinstance(index, int):
            return None
        if not isinstance(delta, Mapping):
            return None
        block = state.content_blocks.get(index)
        if block is None:
            return None
        if delta.get("type") == "input_json_delta":
            partial_json = delta.get("partial_json")
            if isinstance(partial_json, str):
                state.input_json_fragments.setdefault(index, []).append(partial_json)
            return None
        if delta.get("type") != "text_delta" or block.get("type") != "text":
            return None
        text = delta.get("text")
        if not isinstance(text, str) or not text:
            return None
        block["text"] = f"{block.get('text', '')}{text}"
        return text

    @staticmethod
    def _finalize_content_block(
        payload: Mapping[str, Any],
        state: QwenMessagesStreamState,
    ) -> None:
        index = payload.get("index")
        if isinstance(index, bool) or not isinstance(index, int):
            return
        block = state.content_blocks.get(index)
        if not block or block.get("type") != "tool_use":
            return
        raw_input = "".join(state.input_json_fragments.get(index, []))
        if not raw_input:
            return
        try:
            parsed_input = json.loads(raw_input)
        except json.JSONDecodeError as error:
            raise InvalidToolArgumentsError(
                detail=(
                    "Qwen tool input JSON parse failed "
                    f"for tool={block.get('name')!r}: {error}"
                ),
            ) from error
        if not isinstance(parsed_input, dict):
            raise InvalidToolArgumentsError(
                detail=(
                    "Qwen tool input must decode to an object "
                    f"for tool={block.get('name')!r}"
                ),
            )
        block["input"] = parsed_input

    @staticmethod
    def _record_message_delta(
        payload: Mapping[str, Any],
        state: QwenMessagesStreamState,
    ) -> None:
        delta = payload.get("delta")
        if isinstance(delta, Mapping):
            state.message_delta.update(delta)
        usage = payload.get("usage")
        if isinstance(usage, Mapping):
            state.usage.update(usage)

    @staticmethod
    def _normalize_usage(raw_usage: Mapping[str, Any]) -> Optional[Usage]:
        input_tokens = QwenMessagesStreamParser._token_count(
            raw_usage.get("input_tokens"),
        )
        output_tokens = QwenMessagesStreamParser._token_count(
            raw_usage.get("output_tokens"),
        )
        if input_tokens is None and output_tokens is None:
            return None
        return Usage(
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            total_tokens=(input_tokens or 0) + (output_tokens or 0),
        )

    @staticmethod
    def _token_count(value: Any) -> Optional[int]:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None


__all__ = ["QwenMessagesStreamParser", "QwenMessagesStreamState"]
