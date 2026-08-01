"""Google streaming event normalization for sync and async adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional

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


@dataclass
class _GoogleStreamState(_StreamState):
    text_parts: List[str] = field(default_factory=list)
    function_parts: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    usage_metadata: Dict[str, Any] = field(default_factory=dict)
    response_metadata: Dict[str, Any] = field(default_factory=dict)


class _GoogleStreamingMixin:
    """Consume Gemini SSE events and finalize normalized stream responses."""

    def _new_stream_state(
        self,
        buffer_chars: Optional[int],
        capture_reasoning: bool,
    ) -> _GoogleStreamState:
        return _GoogleStreamState(
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=(
                StreamReasoningCollector() if capture_reasoning else None
            ),
            reasoning_response=ChatResponse() if capture_reasoning else None,
        )

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
        state = self._new_stream_state(buffer_chars, capture_reasoning)
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
        state: _GoogleStreamState,
        *,
        on_chunk: Optional[AsyncOnChunk],
        on_delta: Optional[AsyncOnDelta],
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        chunk = event.data if isinstance(event.data, Mapping) else {}
        self._update_stream_state_from_chunk(chunk, state)

        candidates = chunk.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return
        candidate = candidates[0]
        if not isinstance(candidate, Mapping):
            return
        if candidate.get("finishReason") is not None:
            state.finish_reason = str(candidate["finishReason"])
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, Mapping) else []
        if not isinstance(parts, list):
            return
        for part in parts:
            async for text in self._consume_stream_part_async(
                part,
                state,
                on_chunk=on_chunk,
                on_delta=on_delta,
                on_reasoning=on_reasoning,
            ):
                yield text

    async def _consume_stream_part_async(
        self,
        part: Any,
        state: _GoogleStreamState,
        *,
        on_chunk: Optional[AsyncOnChunk],
        on_delta: Optional[AsyncOnDelta],
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        if not isinstance(part, Mapping):
            return
        text = part.get("text")
        if part.get("thought"):
            if (
                state.reasoning_collector is not None
                and state.reasoning_response is not None
                and isinstance(text, str)
                and text
            ):
                await self._record_async_reasoning_event(
                    state.reasoning_response,
                    state.reasoning_collector,
                    text,
                    capture_reasoning=True,
                    kind="summary",
                    on_reasoning=on_reasoning,
                )
        elif isinstance(text, str) and text:
            state.text_parts.append(text)
            async for emitted_text in self._emit_async_stream_chunks(
                state.chunk_buffer.add(text),
                on_chunk,
                on_delta,
            ):
                yield emitted_text
        if isinstance(part.get("functionCall"), Mapping):
            state.function_parts.append(dict(part))

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
        state = self._new_stream_state(buffer_chars, capture_reasoning)
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
        state: _GoogleStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        chunk = event.data if isinstance(event.data, Mapping) else {}
        self._update_stream_state_from_chunk(chunk, state)

        candidates = chunk.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return
        candidate = candidates[0]
        if not isinstance(candidate, Mapping):
            return
        if candidate.get("finishReason") is not None:
            state.finish_reason = str(candidate["finishReason"])
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, Mapping) else []
        if not isinstance(parts, list):
            return
        for part in parts:
            yield from self._consume_stream_part(
                part,
                state,
                on_chunk=on_chunk,
                on_delta=on_delta,
                on_reasoning=on_reasoning,
            )

    def _update_stream_state_from_chunk(
        self,
        chunk: Mapping[str, Any],
        state: _GoogleStreamState,
    ) -> None:
        for field in ("modelVersion", "responseId", "promptFeedback"):
            if field in chunk:
                state.response_metadata[field] = chunk[field]
        chunk_usage = chunk.get("usageMetadata")
        if isinstance(chunk_usage, Mapping):
            state.usage_metadata.update(chunk_usage)
            state.usage_tracker.record(
                state.chunk_buffer,
                self._normalize_stream_usage(state.usage_metadata),
            )

    def _consume_stream_part(
        self,
        part: Any,
        state: _GoogleStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        if not isinstance(part, Mapping):
            return
        text = part.get("text")
        if part.get("thought"):
            if (
                state.reasoning_collector is not None
                and state.reasoning_response is not None
                and isinstance(text, str)
                and text
            ):
                self._record_reasoning_event(
                    state.reasoning_response,
                    state.reasoning_collector,
                    text,
                    capture_reasoning=True,
                    kind="summary",
                    on_reasoning=on_reasoning,
                )
        elif isinstance(text, str) and text:
            state.text_parts.append(text)
            yield from self._emit_stream_chunks(
                state.chunk_buffer.add(text),
                on_chunk,
                on_delta,
            )
        if isinstance(part.get("functionCall"), Mapping):
            state.function_parts.append(dict(part))

    def _finalize_stream_response(
        self,
        state: _GoogleStreamState,
        *,
        capture_reasoning: bool,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        final_response = self._build_stream_response(state)
        parser_kwargs = {"capture_reasoning": True} if capture_reasoning else {}
        chat_response = ChatResponse.from_google_response(
            final_response,
            **parser_kwargs,
        )
        return super()._finalize_stream_response(
            chat_response,
            reasoning_collector=state.reasoning_collector,
            effective_schema=effective_schema,
            response_model=response_model,
        )

    @staticmethod
    def _build_stream_response(state: _GoogleStreamState) -> Dict[str, Any]:
        final_parts: List[Dict[str, Any]] = []
        if state.text_parts:
            final_parts.append({"text": "".join(state.text_parts)})
        final_parts.extend(state.function_parts)
        final_candidate: Dict[str, Any] = {"content": {"parts": final_parts}}
        if state.finish_reason is not None:
            final_candidate["finishReason"] = state.finish_reason
        final_response: Dict[str, Any] = {
            **state.response_metadata,
            "candidates": [final_candidate],
        }
        if state.usage_metadata:
            final_response["usageMetadata"] = state.usage_metadata
        return final_response

    @staticmethod
    def _normalize_stream_usage(raw_usage: Mapping[str, Any]) -> Optional[Usage]:
        input_tokens = _GoogleStreamingMixin._token_count(
            raw_usage.get("promptTokenCount")
        )
        candidate_tokens = _GoogleStreamingMixin._token_count(
            raw_usage.get("candidatesTokenCount")
        )
        thoughts_tokens = _GoogleStreamingMixin._token_count(
            raw_usage.get("thoughtsTokenCount")
        )
        total_tokens = _GoogleStreamingMixin._token_count(
            raw_usage.get("totalTokenCount")
        )
        if (
            input_tokens is None
            and candidate_tokens is None
            and thoughts_tokens is None
            and total_tokens is None
        ):
            return None
        return Usage(
            input_tokens=input_tokens or 0,
            output_tokens=(candidate_tokens or 0) + (thoughts_tokens or 0),
            total_tokens=total_tokens or 0,
        )

    @staticmethod
    def _token_count(value: Any) -> Optional[int]:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        return None


__all__ = ["_GoogleStreamState", "_GoogleStreamingMixin"]
