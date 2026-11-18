"""Shared, operation-local execution for Anthropic Messages connections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator

from ...models.responses.chat_response import ChatResponse
from ...llms.streaming import StreamChunkBuffer, StreamReasoningCollector, StreamUsageTracker
from ...llms.transports import SSEEvent, TransportRequest
from ..base_adapter import _ChatRequestContext, _StreamState
from .payloads import normalize_messages_response
from .streaming import MessagesBlockState, consume_messages_event, finalize_messages_stream


@dataclass(frozen=True)
class _PreparedMessagesOperation:
    """A request and all policy/client bindings captured for one operation."""

    chat_context: _ChatRequestContext = field(repr=False)
    transport_request: TransportRequest = field(repr=False)
    client: Any = field(repr=False)
    capture_reasoning: bool = False


@dataclass
class _MessagesStreamState:
    common: _StreamState = field(repr=False)
    blocks: MessagesBlockState = field(default_factory=MessagesBlockState, repr=False)


class _MessagesPreparedRunners:
    """Run a prepared operation without owning adapter-wide route state."""

    def _finalize_messages_operation(
        self, operation: _PreparedMessagesOperation, raw: Any
    ) -> ChatResponse:
        response = self._normalize_prepared_response(
            raw, capture_reasoning=operation.capture_reasoning
        )
        return self._finalize_chat_response(
            response,
            effective_schema=operation.chat_context.effective_schema,
            response_model=operation.chat_context.response_model,
        )

    def _normalize_prepared_response(
        self, raw: Any, *, capture_reasoning: bool
    ) -> ChatResponse:
        """Overridable response-policy hook for native Messages integrations."""
        return normalize_messages_response(raw, capture_reasoning=capture_reasoning)

    def _normalize_prepared_stream_response(
        self, raw: Any, *, capture_reasoning: bool
    ) -> ChatResponse:
        """Overridable stream reconstruction normalization policy hook."""
        return normalize_messages_response(raw, capture_reasoning=capture_reasoning)

    def _finalize_messages_blocks(self, state: MessagesBlockState) -> dict[str, Any]:
        """Overridable terminal policy hook for native stream consumers."""
        return finalize_messages_stream(state)

    def _run_prepared_chat(self, operation: _PreparedMessagesOperation) -> ChatResponse:
        try:
            result = operation.client._execute_prepared_request(operation.transport_request)
            return self._finalize_messages_operation(operation, result.json())
        except Exception as error:
            self.handle_error(error)

    async def _run_prepared_achat(self, operation: _PreparedMessagesOperation) -> ChatResponse:
        try:
            raw = await operation.client._execute_prepared_request(operation.transport_request)
            return self._finalize_messages_operation(operation, raw)
        except Exception as error:
            self.handle_error(error)

    def _new_messages_stream_state(
        self, buffer_chars: int, capture_reasoning: bool
    ) -> _MessagesStreamState:
        buffer = StreamChunkBuffer(buffer_chars)
        common = _StreamState(
            chunk_buffer=buffer,
            usage_tracker=StreamUsageTracker(),
            reasoning_collector=(StreamReasoningCollector() if capture_reasoning else None),
            reasoning_response=ChatResponse() if capture_reasoning else None,
        )
        return _MessagesStreamState(common=common)

    def _consume_prepared_event(
        self, event: SSEEvent, state: _MessagesStreamState, *,
        on_chunk: Any, on_delta: Any, on_reasoning: Any,
    ) -> Iterator[str]:
        try:
            visible, thinking = self._consume_messages_event(state.blocks, event)
        except Exception as error:
            self.handle_error(error)
        if visible:
            yield from self._emit_stream_chunks(
                state.common.chunk_buffer.add(visible), on_chunk, on_delta
            )
        if thinking and state.common.reasoning_collector is not None:
            self._record_reasoning_event(
                state.common.reasoning_response, state.common.reasoning_collector, thinking,
                capture_reasoning=True, kind="summary", on_reasoning=on_reasoning,
            )

    async def _consume_prepared_event_async(
        self, event: SSEEvent, state: _MessagesStreamState, *,
        on_chunk: Any, on_delta: Any, on_reasoning: Any,
    ) -> AsyncIterator[str]:
        try:
            visible, thinking = self._consume_messages_event(state.blocks, event)
        except Exception as error:
            self.handle_error(error)
        if visible:
            async for text in self._emit_async_stream_chunks(
                state.common.chunk_buffer.add(visible), on_chunk, on_delta
            ):
                yield text
        if thinking and state.common.reasoning_collector is not None:
            await self._record_async_reasoning_event(
                state.common.reasoning_response, state.common.reasoning_collector, thinking,
                capture_reasoning=True, kind="summary", on_reasoning=on_reasoning,
            )

    def _consume_messages_event(
        self, state: MessagesBlockState, event: SSEEvent
    ) -> tuple[str, str]:
        """Overridable decoded-event policy hook; accumulation stays shared."""
        return consume_messages_event(state, event)

    def _finalize_prepared_stream(
        self, operation: _PreparedMessagesOperation, state: _MessagesStreamState,
        *, capture_reasoning: bool, effective_schema: Any, response_model: Any,
    ) -> ChatResponse:
        try:
            raw = self._finalize_messages_blocks(state.blocks)
            result = self._normalize_prepared_stream_response(
                raw, capture_reasoning=capture_reasoning
            )
        except Exception as error:
            self.handle_error(error)
        return self._finalize_stream_response(
            result,
            reasoning_collector=state.common.reasoning_collector,
            effective_schema=effective_schema,
            response_model=response_model,
        )

    def _run_prepared_stream(self, operation: _PreparedMessagesOperation, *,
                             state: _MessagesStreamState | None = None,
                             buffer_chars: int | None, on_delta: Any,
                             on_tool_call: Any, on_done: Any, on_chunk: Any,
                             on_reasoning: Any) -> Iterator[str]:
        state = state or self._new_messages_stream_state(buffer_chars, operation.capture_reasoning)
        try:
            events = self._managed_sync_events(
                operation.client._execute_prepared_stream(operation.transport_request)
            )
        except Exception as error:
            self.handle_error(error)
        primary: BaseException | None = None
        try:
            yield from self._run_sync_stream(
                events, state.common,
                consume_event=lambda event, common, **callbacks: self._consume_prepared_event(
                    event, state, **callbacks
                ),
                finalize_response=lambda common, **kwargs: self._finalize_prepared_stream(
                    operation, state, **kwargs
                ),
                effective_schema=operation.chat_context.effective_schema,
                response_model=operation.chat_context.response_model,
                on_delta=on_delta,
                on_tool_call=on_tool_call,
                on_done=on_done,
                on_chunk=on_chunk,
                capture_reasoning=operation.capture_reasoning,
                on_reasoning=on_reasoning,
            )
        except BaseException as exc:
            primary = exc
            raise
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException:
                    if primary is None:
                        raise

    async def _run_prepared_astream(self, operation: _PreparedMessagesOperation, *,
                                    state: _MessagesStreamState | None = None,
                                    buffer_chars: int | None, on_delta: Any,
                                    on_tool_call: Any, on_done: Any, on_chunk: Any,
                                    on_reasoning: Any) -> AsyncIterator[str]:
        state = state or self._new_messages_stream_state(buffer_chars, operation.capture_reasoning)
        try:
            events = operation.client._execute_prepared_stream(operation.transport_request)
        except Exception as error:
            self.handle_error(error)
        lifecycle = self._run_async_stream(
            events, state.common,
            consume_event=lambda event, common, **callbacks: self._consume_prepared_event_async(
                event, state, **callbacks
            ),
            finalize_response=lambda common, **kwargs: self._finalize_prepared_stream(
                operation, state, **kwargs
            ),
            effective_schema=operation.chat_context.effective_schema,
            response_model=operation.chat_context.response_model,
            on_delta=on_delta,
            on_tool_call=on_tool_call,
            on_done=on_done,
            on_chunk=on_chunk,
            capture_reasoning=operation.capture_reasoning,
            on_reasoning=on_reasoning,
        )
        async with self._closing_async_source(events):
            async with self._closing_async_source(lifecycle):
                async for text in lifecycle:
                    yield text

    @staticmethod
    def _managed_sync_events(source: Any) -> Iterator[SSEEvent]:
        """Own a synchronous source and suppress close failures during unwind."""
        iterator = iter(source)
        try:
            while True:
                try:
                    event = next(iterator)
                except StopIteration:
                    break
                yield event
        except BaseException:
            close = getattr(iterator, "close", None) or getattr(source, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException:
                    pass
            raise
        else:
            close = getattr(iterator, "close", None) or getattr(source, "close", None)
            if close is not None:
                close()

__all__ = ["_MessagesPreparedRunners", "_PreparedMessagesOperation"]
