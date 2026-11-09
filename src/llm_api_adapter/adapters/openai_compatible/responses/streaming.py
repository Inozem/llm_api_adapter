"""Responses event stream consumers and their per-stream state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional

from ...base_adapter import (
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
from ....llms.streaming import (
    StreamChunkBuffer,
    StreamReasoningCollector,
    StreamUsageTracker,
)
from ....models.responses.chat_response import ChatResponse


@dataclass
class _ResponsesStreamState(_StreamState):
    final_response: Optional[dict] = None
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    text_parts: List[str] = field(default_factory=list)
    function_calls: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class _ResponsesStreamingMixin:
    """Consume Responses events through the shared stream lifecycle."""

    async def _consume_responses_stream_async(
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
        state = _ResponsesStreamState(
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
            consume_event=self._consume_responses_stream_event_async,
            finalize_response=self._finalize_responses_stream,
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

    async def _consume_responses_stream_event_async(
        self,
        event: Any,
        state: _ResponsesStreamState,
        *,
        on_chunk: Optional[AsyncOnChunk],
        on_delta: Optional[AsyncOnDelta],
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        payload = event.data if isinstance(event.data, Mapping) else {}
        event_type = event.event or payload.get("type")
        response_data = payload.get("response")
        if isinstance(response_data, Mapping):
            state.response_metadata.update(response_data)
        state.usage_tracker.record(
            state.chunk_buffer,
            self._normalize_stream_usage(
                self._response_event_usage(payload, response_data),
                input_field="input_tokens",
                output_field="output_tokens",
            ),
        )

        if event_type in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ):
            await self._handle_responses_reasoning_delta_async(
                payload,
                event_type,
                state,
                on_reasoning,
            )
        elif event_type == "response.output_text.delta":
            delta = payload.get("delta")
            if isinstance(delta, str) and delta:
                state.text_parts.append(delta)
                async for text in self._emit_async_stream_chunks(
                    state.chunk_buffer.add(delta),
                    on_chunk,
                    on_delta,
                ):
                    yield text
        elif event_type in (
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        ):
            self._handle_responses_function_call_event(
                payload,
                event_type,
                state,
            )
        elif event_type in (
            "response.completed",
            "response.incomplete",
        ) and isinstance(response_data, Mapping):
            state.final_response = dict(response_data)

    async def _handle_responses_reasoning_delta_async(
        self,
        payload: Mapping[str, Any],
        event_type: str,
        state: _ResponsesStreamState,
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> None:
        delta = payload.get("delta")
        if (
            state.reasoning_collector is None
            or state.reasoning_response is None
            or not isinstance(delta, str)
            or not delta
        ):
            return
        await self._record_async_reasoning_event(
            state.reasoning_response,
            state.reasoning_collector,
            delta,
            capture_reasoning=True,
            kind=(
                "summary"
                if event_type == "response.reasoning_summary_text.delta"
                else "content"
            ),
            on_reasoning=on_reasoning,
        )

    def _consume_responses_stream(
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
        state = _ResponsesStreamState(
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
            consume_event=self._consume_responses_stream_event,
            finalize_response=self._finalize_responses_stream,
            effective_schema=effective_schema,
            response_model=response_model,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            capture_reasoning=capture_reasoning,
            on_reasoning=on_reasoning,
        )

    def _consume_responses_stream_event(
        self,
        event: Any,
        state: _ResponsesStreamState,
        *,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        payload = event.data if isinstance(event.data, Mapping) else {}
        event_type = event.event or payload.get("type")
        response_data = payload.get("response")
        if isinstance(response_data, Mapping):
            state.response_metadata.update(response_data)
        state.usage_tracker.record(
            state.chunk_buffer,
            self._normalize_stream_usage(
                self._response_event_usage(payload, response_data),
                input_field="input_tokens",
                output_field="output_tokens",
            ),
        )

        if event_type in (
            "response.reasoning_summary_text.delta",
            "response.reasoning_text.delta",
        ):
            self._handle_responses_reasoning_delta(
                payload,
                event_type,
                state,
                on_reasoning,
            )
        elif event_type == "response.output_text.delta":
            yield from self._handle_responses_output_text_delta(
                payload,
                state,
                on_chunk,
                on_delta,
            )
        elif event_type in (
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        ):
            self._handle_responses_function_call_event(
                payload,
                event_type,
                state,
            )
        elif event_type in (
            "response.completed",
            "response.incomplete",
        ) and isinstance(response_data, Mapping):
            state.final_response = dict(response_data)

    def _handle_responses_reasoning_delta(
        self,
        payload: Mapping[str, Any],
        event_type: str,
        state: _ResponsesStreamState,
        on_reasoning: Optional[OnReasoning],
    ) -> None:
        delta = payload.get("delta")
        if (
            state.reasoning_collector is None
            or state.reasoning_response is None
            or not isinstance(delta, str)
            or not delta
        ):
            return
        self._record_reasoning_event(
            state.reasoning_response,
            state.reasoning_collector,
            delta,
            capture_reasoning=True,
            kind=(
                "summary"
                if event_type == "response.reasoning_summary_text.delta"
                else "content"
            ),
            on_reasoning=on_reasoning,
        )

    def _handle_responses_output_text_delta(
        self,
        payload: Mapping[str, Any],
        state: _ResponsesStreamState,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
    ) -> Iterator[str]:
        delta = payload.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        state.text_parts.append(delta)
        yield from self._emit_stream_chunks(
            state.chunk_buffer.add(delta),
            on_chunk,
            on_delta,
        )

    @staticmethod
    def _handle_responses_function_call_event(
        payload: Mapping[str, Any],
        event_type: str,
        state: _ResponsesStreamState,
    ) -> None:
        call_key = str(
            payload.get("call_id")
            or payload.get("item_id")
            or payload.get("output_index")
            or len(state.function_calls)
        )
        call = state.function_calls.setdefault(
            call_key,
            {"type": "function_call", "arguments": ""},
        )
        call["call_id"] = payload.get("call_id") or call.get("call_id")
        call["id"] = payload.get("item_id") or call.get("id")
        call["name"] = payload.get("name") or call.get("name")
        if event_type.endswith(".delta") and isinstance(payload.get("delta"), str):
            call["arguments"] = f"{call.get('arguments', '')}{payload['delta']}"
        elif "arguments" in payload:
            call["arguments"] = payload["arguments"]

    def _finalize_responses_stream(
        self,
        state: _ResponsesStreamState,
        *,
        capture_reasoning: bool,
        effective_schema: Optional[dict],
        response_model: Optional[Any],
    ) -> ChatResponse:
        if state.final_response is None:
            final_response = self._build_responses_stream_response(state)
        else:
            final_response = state.final_response
        chat_response = self._parse_chat_response(
            final_response,
            use_responses_api=True,
            capture_reasoning=capture_reasoning,
        )
        return self._finalize_stream_response(
            chat_response,
            reasoning_collector=state.reasoning_collector,
            effective_schema=effective_schema,
            response_model=response_model,
        )

    @staticmethod
    def _response_event_usage(
        payload: Mapping[str, Any],
        response_data: Any,
    ) -> Any:
        if isinstance(payload.get("usage"), Mapping):
            return payload["usage"]
        if isinstance(response_data, Mapping):
            return response_data.get("usage")
        return None
