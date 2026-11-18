"""Anthropic streaming event normalization for sync and async adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterator, Literal, Mapping, Optional

from ..base_adapter import (
    AsyncOnChunk,
    AsyncOnDelta,
    AsyncOnDone,
    AsyncOnReasoning,
    AsyncOnToolCall,
    OnChunk,
    OnDelta,
    OnDone,
    OnReasoning,
    OnToolCall,
    _StreamState,
)
from ...llms.streaming import (
    StreamChunkBuffer,
    StreamReasoningCollector,
    StreamUsageTracker,
)
from ...models.responses.chat_response import ChatResponse, Usage
from ..anthropic_compatible import streaming as common_streaming
from ..anthropic_compatible.streaming import MessagesBlockState
from ...llms.transports import SSEEvent


@dataclass
class _AnthropicStreamState(_StreamState):
    usage: Dict[str, Any]
    shared: MessagesBlockState


class _AnthropicStreamingMixin:
    """Consume Anthropic SSE events and finalize normalized stream responses."""

    @staticmethod
    def _native_event(event: Any) -> Optional[SSEEvent]:
        """Skip malformed native frames and normalize their historical event fallback."""
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
    def _native_delta(block: Dict[str, Any], delta: Mapping[str, Any]) -> bool:
        kind = delta.get("type")
        if not isinstance(kind, str):
            return False
        field = {"text_delta": "text", "thinking_delta": "thinking",
                 "input_json_delta": "partial_json"}.get(kind)
        if field is None:
            return False
        value = delta.get(field)
        return isinstance(value, str) and not (field == "partial_json" and not value)

    @staticmethod
    def _native_index(
        index: int, kind: str, state: MessagesBlockState
    ) -> bool | Literal["replace"]:
        if kind == "content_block_start" and index in state.blocks:
            return "replace"
        return index in state.blocks

    async def _consume_stream_async(
        self,
        events: AsyncIterator[Any],
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[AsyncOnDelta],
        on_tool_call: Optional[AsyncOnToolCall],
        on_done: Optional[AsyncOnDone],
        buffer_chars: Optional[int],
        on_chunk: Optional[AsyncOnChunk],
        capture_reasoning: bool,
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        state = _AnthropicStreamState(
            usage={},
            shared=MessagesBlockState(message={"model": self.model, "content": []}),
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=(
                StreamReasoningCollector() if capture_reasoning else None
            ),
            reasoning_response=ChatResponse() if capture_reasoning else None,
        )

        async for text in self._run_async_stream(
            events,
            state,
            consume_event=self._consume_stream_event_async,
            effective_schema=effective_schema,
            response_model=response_model,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            capture_reasoning=capture_reasoning,
            on_reasoning=on_reasoning,
        ):
            yield text

    async def _consume_stream_event_async(
        self,
        event: Any,
        state: _AnthropicStreamState,
        *,
        on_chunk: Optional[AsyncOnChunk],
        on_delta: Optional[AsyncOnDelta],
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        normalized = self._native_event(event)
        if normalized is None:
            return
        visible, thinking = common_streaming.consume_messages_event(
            state.shared, normalized, block_policy=self._native_block,
            delta_policy=self._native_delta, index_policy=self._native_index,
        )
        state.usage = dict(state.shared.usage) if isinstance(state.shared.usage, Mapping) else {}
        if visible:
            async for text in self._emit_async_stream_chunks(
                state.chunk_buffer.add(visible), on_chunk, on_delta
            ):
                yield text
        if thinking and state.reasoning_collector is not None and state.reasoning_response is not None:
            await self._record_async_reasoning_event(
                state.reasoning_response, state.reasoning_collector, thinking,
                capture_reasoning=True, kind="summary", on_reasoning=on_reasoning,
            )
        if normalized.event in {"message_start", "message_delta"}:
            state.usage_tracker.record(
                state.chunk_buffer,
                self._normalize_stream_usage(state.usage),
            )

    def _consume_stream(
        self,
        events: Iterator[Any],
        effective_schema: Optional[dict],
        response_model: Optional[Any],
        on_delta: Optional[OnDelta],
        on_tool_call: Optional[OnToolCall],
        on_done: Optional[OnDone],
        buffer_chars: Optional[int],
        on_chunk: Optional[OnChunk],
        capture_reasoning: bool,
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        state = _AnthropicStreamState(
            usage={},
            shared=MessagesBlockState(message={"model": self.model, "content": []}),
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=(
                StreamReasoningCollector() if capture_reasoning else None
            ),
            reasoning_response=ChatResponse() if capture_reasoning else None,
        )

        yield from self._run_sync_stream(
            events,
            state,
            consume_event=self._consume_stream_event,
            effective_schema=effective_schema,
            response_model=response_model,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            capture_reasoning=capture_reasoning,
            on_reasoning=on_reasoning,
        )

    def _consume_stream_event(
        self,
        event: Any,
        state: _AnthropicStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        normalized = self._native_event(event)
        if normalized is None:
            return
        visible, thinking = common_streaming.consume_messages_event(
            state.shared, normalized, block_policy=self._native_block,
            delta_policy=self._native_delta, index_policy=self._native_index,
        )
        state.usage = dict(state.shared.usage) if isinstance(state.shared.usage, Mapping) else {}
        if visible:
            yield from self._emit_stream_chunks(state.chunk_buffer.add(visible), on_chunk, on_delta)
        if thinking and state.reasoning_collector is not None and state.reasoning_response is not None:
            self._record_reasoning_event(
                state.reasoning_response, state.reasoning_collector, thinking,
                capture_reasoning=True, kind="summary", on_reasoning=on_reasoning,
            )
        if normalized.event in {"message_start", "message_delta"}:
            state.usage_tracker.record(
                state.chunk_buffer,
                self._normalize_stream_usage(state.usage),
            )

    def _finalize_stream_response(
        self,
        state: _AnthropicStreamState,
        *,
        capture_reasoning: bool,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        parser_kwargs = {"capture_reasoning": True} if capture_reasoning else {}
        chat_response = ChatResponse.from_anthropic_response(
            common_streaming.finalize_messages_stream(
                state.shared, validate_completion=lambda _state: None
            ),
            **parser_kwargs,
        )
        return super()._finalize_stream_response(
            chat_response,
            reasoning_collector=state.reasoning_collector,
            effective_schema=effective_schema,
            response_model=response_model,
        )

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



__all__ = ["_AnthropicStreamState", "_AnthropicStreamingMixin"]
