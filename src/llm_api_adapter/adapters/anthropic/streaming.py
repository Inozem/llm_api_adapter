"""Anthropic streaming event normalization for sync and async adapters."""

from __future__ import annotations

from dataclasses import dataclass
import json
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
from ...errors.llm_api_error import InvalidToolArgumentsError
from ...llms.streaming import (
    StreamChunkBuffer,
    StreamReasoningCollector,
    StreamUsageTracker,
)
from ...models.responses.chat_response import ChatResponse, Usage


@dataclass
class _AnthropicStreamState(_StreamState):
    message_data: Dict[str, Any]
    content_blocks: Dict[int, Dict[str, Any]]
    input_json_fragments: Dict[int, List[str]]
    usage: Dict[str, Any]
    message_delta: Dict[str, Any]


class _AnthropicStreamingMixin:
    """Consume Anthropic SSE events and finalize normalized stream responses."""

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
            message_data={"model": self.model, "content": []},
            content_blocks={},
            input_json_fragments={},
            usage={},
            message_delta={},
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=(
                StreamReasoningCollector() if capture_reasoning else None
            ),
            reasoning_response=ChatResponse() if capture_reasoning else None,
        )

        async for event in self._aiter_provider_stream_events(events):
            async for text in self._consume_stream_event_async(
                event,
                state,
                on_chunk=on_chunk,
                on_delta=on_delta,
                on_reasoning=on_reasoning,
            ):
                yield text

        chat_response = self._finalize_stream_response(
            state,
            capture_reasoning=capture_reasoning,
            effective_schema=effective_schema,
            response_model=response_model,
        )
        async for text in self._complete_async_stream(
            chat_response,
            state.chunk_buffer,
            on_chunk,
            on_delta,
            on_tool_call,
            on_done,
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
        payload = event.data if isinstance(event.data, Mapping) else {}
        event_type = event.event or payload.get("type")
        if event_type == "message_start":
            state.message_data = self._handle_message_start(
                payload,
                state.usage,
                state.message_data,
            )
            state.usage_tracker.record(
                state.chunk_buffer,
                self._normalize_stream_usage(state.usage),
            )
        elif event_type == "content_block_start":
            self._start_content_block(payload, state.content_blocks)
        elif event_type == "content_block_delta":
            async for text in self._consume_content_block_delta_async(
                payload,
                state.content_blocks,
                state.input_json_fragments,
                state.chunk_buffer,
                on_chunk,
                on_delta,
                state.reasoning_collector,
                state.reasoning_response,
                on_reasoning,
            ):
                yield text
        elif event_type == "content_block_stop":
            self._finalize_content_block(
                payload,
                state.content_blocks,
                state.input_json_fragments,
            )
        elif event_type == "message_delta":
            self._handle_message_delta(
                payload,
                state.message_delta,
                state.usage,
            )
            state.usage_tracker.record(
                state.chunk_buffer,
                self._normalize_stream_usage(state.usage),
            )

    async def _consume_content_block_delta_async(
        self,
        payload: Mapping[str, Any],
        content_blocks: Dict[int, Dict[str, Any]],
        input_json_fragments: Dict[int, List[str]],
        chunk_buffer: StreamChunkBuffer,
        on_chunk: Optional[AsyncOnChunk],
        on_delta: Optional[AsyncOnDelta],
        reasoning_collector: Optional[StreamReasoningCollector],
        reasoning_response: Optional[ChatResponse],
        on_reasoning: Optional[AsyncOnReasoning],
    ) -> AsyncIterator[str]:
        index = payload.get("index")
        delta = payload.get("delta")
        if not isinstance(index, int) or not isinstance(delta, Mapping):
            return
        block = content_blocks.get(index)
        if block is None:
            return
        if delta.get("type") == "text_delta":
            text = delta.get("text")
            if isinstance(text, str) and text:
                block["text"] = f"{block.get('text', '')}{text}"
                async for emitted_text in self._emit_async_stream_chunks(
                    chunk_buffer.add(text),
                    on_chunk,
                    on_delta,
                ):
                    yield emitted_text
        elif delta.get("type") == "thinking_delta":
            thinking_text = delta.get("thinking")
            if isinstance(thinking_text, str) and thinking_text:
                block["thinking"] = f"{block.get('thinking', '')}{thinking_text}"
                if reasoning_collector is not None and reasoning_response is not None:
                    await self._record_async_reasoning_event(
                        reasoning_response,
                        reasoning_collector,
                        thinking_text,
                        capture_reasoning=True,
                        kind="summary",
                        on_reasoning=on_reasoning,
                    )
        elif delta.get("type") == "input_json_delta":
            partial_json = delta.get("partial_json")
            if isinstance(partial_json, str):
                input_json_fragments.setdefault(index, []).append(partial_json)

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
            message_data={"model": self.model, "content": []},
            content_blocks={},
            input_json_fragments={},
            usage={},
            message_delta={},
            chunk_buffer=StreamChunkBuffer(buffer_chars),
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=(
                StreamReasoningCollector() if capture_reasoning else None
            ),
            reasoning_response=ChatResponse() if capture_reasoning else None,
        )

        for event in self._iter_provider_stream_events(events):
            yield from self._consume_stream_event(
                event,
                state,
                on_chunk=on_chunk,
                on_delta=on_delta,
                on_reasoning=on_reasoning,
            )

        chat_response = self._finalize_stream_response(
            state,
            capture_reasoning=capture_reasoning,
            effective_schema=effective_schema,
            response_model=response_model,
        )
        yield from self._complete_stream(
            chat_response,
            state.chunk_buffer,
            on_chunk,
            on_delta,
            on_tool_call,
            on_done,
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
        payload = event.data if isinstance(event.data, Mapping) else {}
        event_type = event.event or payload.get("type")
        if event_type == "message_start":
            state.message_data = self._handle_message_start(
                payload,
                state.usage,
                state.message_data,
            )
            state.usage_tracker.record(
                state.chunk_buffer,
                self._normalize_stream_usage(state.usage),
            )
        elif event_type == "content_block_start":
            self._start_content_block(payload, state.content_blocks)
        elif event_type == "content_block_delta":
            yield from self._consume_content_block_delta(
                payload,
                state.content_blocks,
                state.input_json_fragments,
                state.chunk_buffer,
                on_chunk,
                on_delta,
                state.reasoning_collector,
                state.reasoning_response,
                on_reasoning,
            )
        elif event_type == "content_block_stop":
            self._finalize_content_block(
                payload,
                state.content_blocks,
                state.input_json_fragments,
            )
        elif event_type == "message_delta":
            self._handle_message_delta(
                payload,
                state.message_delta,
                state.usage,
            )
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
            self._build_stream_response(
                state.message_data,
                state.content_blocks,
                state.message_delta,
                state.usage,
            ),
            **parser_kwargs,
        )
        return super()._finalize_stream_response(
            chat_response,
            reasoning_collector=state.reasoning_collector,
            effective_schema=effective_schema,
            response_model=response_model,
        )

    def _handle_message_start(
        self,
        payload: Mapping[str, Any],
        usage: Dict[str, Any],
        message_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        message = payload.get("message")
        if not isinstance(message, Mapping):
            return message_data
        initial_usage = message.get("usage")
        if isinstance(initial_usage, Mapping):
            usage.update(initial_usage)
        return dict(message)

    def _start_content_block(
        self,
        payload: Mapping[str, Any],
        content_blocks: Dict[int, Dict[str, Any]],
    ) -> None:
        index = payload.get("index")
        block = payload.get("content_block")
        if not isinstance(index, int) or not isinstance(block, Mapping):
            return
        content_blocks[index] = dict(block)
        if block.get("type") == "text":
            content_blocks[index]["text"] = str(block.get("text") or "")
        elif block.get("type") == "thinking":
            content_blocks[index]["thinking"] = str(block.get("thinking") or "")
        elif block.get("type") == "tool_use":
            current_input = block.get("input")
            content_blocks[index]["input"] = (
                dict(current_input) if isinstance(current_input, Mapping) else {}
            )

    def _consume_content_block_delta(
        self,
        payload: Mapping[str, Any],
        content_blocks: Dict[int, Dict[str, Any]],
        input_json_fragments: Dict[int, List[str]],
        chunk_buffer: StreamChunkBuffer,
        on_chunk: Optional[OnChunk],
        on_delta: Optional[OnDelta],
        reasoning_collector: Optional[StreamReasoningCollector],
        reasoning_response: Optional[ChatResponse],
        on_reasoning: Optional[OnReasoning],
    ) -> Iterator[str]:
        index = payload.get("index")
        delta = payload.get("delta")
        if not isinstance(index, int) or not isinstance(delta, Mapping):
            return
        block = content_blocks.get(index)
        if block is None:
            return
        if delta.get("type") == "text_delta":
            text = delta.get("text")
            if isinstance(text, str) and text:
                block["text"] = f"{block.get('text', '')}{text}"
                yield from self._emit_stream_chunks(
                    chunk_buffer.add(text),
                    on_chunk,
                    on_delta,
                )
        elif delta.get("type") == "thinking_delta":
            thinking_text = delta.get("thinking")
            if isinstance(thinking_text, str) and thinking_text:
                block["thinking"] = f"{block.get('thinking', '')}{thinking_text}"
                if reasoning_collector is not None and reasoning_response is not None:
                    self._record_reasoning_event(
                        reasoning_response,
                        reasoning_collector,
                        thinking_text,
                        capture_reasoning=True,
                        kind="summary",
                        on_reasoning=on_reasoning,
                    )
        elif delta.get("type") == "input_json_delta":
            partial_json = delta.get("partial_json")
            if isinstance(partial_json, str):
                input_json_fragments.setdefault(index, []).append(partial_json)

    def _finalize_content_block(
        self,
        payload: Mapping[str, Any],
        content_blocks: Dict[int, Dict[str, Any]],
        input_json_fragments: Dict[int, List[str]],
    ) -> None:
        index = payload.get("index")
        block = content_blocks.get(index) if isinstance(index, int) else None
        if not block or block.get("type") != "tool_use":
            return
        raw_input = "".join(input_json_fragments.get(index, []))
        if not raw_input:
            return
        try:
            parsed_input = json.loads(raw_input)
        except json.JSONDecodeError as error:
            raise InvalidToolArgumentsError(
                detail=(
                    "Anthropic tool input JSON parse failed "
                    f"for tool={block.get('name')!r}: {error}"
                )
            ) from error
        if not isinstance(parsed_input, dict):
            raise InvalidToolArgumentsError(
                detail=(
                    "Anthropic tool input must decode to dict "
                    f"for tool={block.get('name')!r}"
                )
            )
        block["input"] = parsed_input

    def _handle_message_delta(
        self,
        payload: Mapping[str, Any],
        message_delta: Dict[str, Any],
        usage: Dict[str, Any],
    ) -> None:
        delta = payload.get("delta")
        if isinstance(delta, Mapping):
            message_delta.update(delta)
        delta_usage = payload.get("usage")
        if isinstance(delta_usage, Mapping):
            usage.update(delta_usage)

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

    def _build_stream_response(
        self,
        message_data: Dict[str, Any],
        content_blocks: Dict[int, Dict[str, Any]],
        message_delta: Dict[str, Any],
        usage: Dict[str, Any],
    ) -> Dict[str, Any]:
        final_response = dict(message_data)
        final_response["content"] = [
            content_blocks[index] for index in sorted(content_blocks)
        ]
        final_response.update(message_delta)
        if usage:
            final_response["usage"] = usage
        return final_response


__all__ = ["_AnthropicStreamState", "_AnthropicStreamingMixin"]
