"""Native Claude stream policy hooks over the shared Messages lifecycle."""

from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Optional

from ...models.responses.chat_response import ChatResponse, Usage
from ...llms.transports import SSEEvent
from ..anthropic_compatible.execution import _MessagesStreamState
from ..anthropic_compatible import streaming as common_streaming
from ..anthropic_compatible.streaming import MessagesBlockState


class _AnthropicStreamingMixin:
    """Keep native frame tolerance/accounting while common runners own lifecycle."""

    @staticmethod
    def _native_event(event: Any) -> Optional[SSEEvent]:
        if not isinstance(event, SSEEvent) or not isinstance(event.data, Mapping):
            return None
        payload = event.data
        kind = event.event or payload.get("type")
        if not isinstance(kind, str):
            return None
        if kind in {"content_block_start", "content_block_delta", "content_block_stop"}:
            index = payload.get("index")
            if not isinstance(index, int):
                return None
            if isinstance(index, bool):
                payload = dict(payload, index=int(index))
            if kind == "content_block_start" and not isinstance(payload.get("content_block"), Mapping):
                return None
            if kind == "content_block_delta" and not isinstance(payload.get("delta"), Mapping):
                return None
        if kind == "message_start" and not isinstance(payload.get("message"), Mapping):
            return None
        if kind == "message_delta" and not isinstance(payload.get("delta"), Mapping):
            payload = dict(payload, delta={})
        if kind in {"message_start", "message_delta"}:
            payload = dict(payload)
            message = payload.get("message")
            if isinstance(message, Mapping):
                message = dict(message)
                if "usage" in message and not isinstance(message["usage"], Mapping):
                    message.pop("usage")
                payload["message"] = message
            for key in ("message", "delta", "usage"):
                if key in payload and not isinstance(payload[key], Mapping):
                    payload.pop(key)
        return SSEEvent(event=kind, data=payload, done=event.done)

    @staticmethod
    def _native_block(_index: int, block: Dict[str, Any]) -> Dict[str, Any]:
        if block.get("type") == "text":
            block["text"] = str(block.get("text") or "")
        elif block.get("type") == "thinking":
            block["thinking"] = str(block.get("thinking") or "")
        elif block.get("type") == "tool_use":
            value = block.get("input")
            block["input"] = dict(value) if isinstance(value, Mapping) else {}
        return block

    @staticmethod
    def _native_delta(_block: Dict[str, Any], delta: Mapping[str, Any]) -> bool:
        kind = delta.get("type")
        if not isinstance(kind, str):
            return False
        field = {"text_delta": "text", "thinking_delta": "thinking", "input_json_delta": "partial_json"}.get(kind)
        if field is None:
            return False
        value = delta.get(field)
        return isinstance(value, str) and not (field == "partial_json" and not value)

    @staticmethod
    def _native_index(index: int, kind: str, state: MessagesBlockState) -> bool | Literal["replace"]:
        if kind == "content_block_start" and index in state.blocks:
            return "replace"
        return index in state.blocks

    def _new_messages_stream_state(self, buffer_chars: int, capture_reasoning: bool) -> _MessagesStreamState:
        state = super()._new_messages_stream_state(buffer_chars, capture_reasoning)
        state.blocks.message.update(model=self.model, content=[])
        return state

    def _consume_messages_event(self, state: MessagesBlockState, event: SSEEvent) -> tuple[str, str]:
        normalized = self._native_event(event)
        if normalized is None:
            return "", ""
        return common_streaming.consume_messages_event(
            state, normalized, block_policy=self._native_block,
            delta_policy=self._native_delta, index_policy=self._native_index,
        )

    def _record_prepared_usage(self, event: SSEEvent, state: _MessagesStreamState) -> None:
        normalized = self._native_event(event)
        if normalized is None or normalized.event not in {"message_start", "message_delta"}:
            return
        usage = state.blocks.usage if isinstance(state.blocks.usage, Mapping) else {}
        state.common.usage_tracker.record(
            state.common.chunk_buffer, self._normalize_stream_usage(usage)
        )

    def _finalize_messages_blocks(self, state: MessagesBlockState) -> dict[str, Any]:
        return common_streaming.finalize_messages_stream(
            state, validate_completion=lambda _state: None
        )

    def _normalize_prepared_response(self, raw: Any, *, capture_reasoning: bool) -> ChatResponse:
        return self._parse_chat_response(raw, capture_reasoning=capture_reasoning)

    def _normalize_prepared_stream_response(self, raw: Any, *, capture_reasoning: bool) -> ChatResponse:
        return self._normalize_prepared_response(raw, capture_reasoning=capture_reasoning)

    @staticmethod
    def _normalize_stream_usage(raw_usage: Mapping[str, Any]) -> Optional[Usage]:
        input_tokens = _AnthropicStreamingMixin._token_count(
            raw_usage.get("input_tokens")
        )
        output_tokens = _AnthropicStreamingMixin._token_count(
            raw_usage.get("output_tokens")
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


__all__ = ["_AnthropicStreamingMixin"]
