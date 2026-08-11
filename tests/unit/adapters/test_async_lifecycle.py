import asyncio

import pytest

from src.llm_api_adapter.adapters.base_adapter import LLMAdapterBase, _StreamState
from src.llm_api_adapter.errors.llm_api_error import LLMAPIError
from src.llm_api_adapter.llms.streaming import (
    StreamChunkBuffer,
    StreamReasoningCollector,
    StreamUsageTracker,
)
from src.llm_api_adapter.models.responses.chat_response import ChatResponse
from src.llm_api_adapter.models.responses.stream_chunk import StreamChunk
from src.llm_api_adapter.models.tools import ToolCall


class AsyncLifecycleAdapter(LLMAdapterBase):
    def chat(self, messages, **kwargs):
        raise NotImplementedError

    def stream_chat(self, messages, **kwargs):
        raise NotImplementedError


@pytest.fixture
def adapter():
    return AsyncLifecycleAdapter(
        company="openai",
        api_key="dummy_key",
        model="gpt-5",
    )


def make_chunk(text="hello"):
    return StreamChunk(
        text=text,
        index=0,
        elapsed_s=0.0,
        delta_s=0.0,
    )


def make_stream_state(buffer_chars=10):
    return _StreamState(
        chunk_buffer=StreamChunkBuffer(buffer_chars=buffer_chars),
        usage_tracker=StreamUsageTracker(),
        reasoning_collector=None,
        reasoning_response=None,
    )


@pytest.mark.unit
def test_sync_stream_lifecycle_delegates_events_and_completes_callbacks(adapter):
    state = make_stream_state()
    order = []

    def consume_event(event, state, *, on_chunk, on_delta, on_reasoning):
        yield from adapter._emit_stream_chunks(
            state.chunk_buffer.add(event),
            on_chunk,
            on_delta,
        )

    def finalize_response(state, **kwargs):
        assert kwargs["capture_reasoning"] is False
        return ChatResponse(content="complete")

    output = list(
        adapter._run_sync_stream(
            iter(["hello"]),
            state,
            consume_event=consume_event,
            finalize_response=finalize_response,
            effective_schema=None,
            response_model=None,
            on_delta=lambda text: order.append(("delta", text)),
            on_tool_call=None,
            on_done=lambda response: order.append(("done", response.content)),
            on_chunk=None,
            capture_reasoning=False,
            on_reasoning=None,
        )
    )

    assert output == ["hello"]
    assert order == [("delta", "hello"), ("done", "complete")]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_stream_lifecycle_delegates_events_and_completes_callbacks(adapter):
    state = make_stream_state()
    order = []

    async def events():
        yield "hello"

    async def consume_event(event, state, *, on_chunk, on_delta, on_reasoning):
        async for text in adapter._emit_async_stream_chunks(
            state.chunk_buffer.add(event),
            on_chunk,
            on_delta,
        ):
            yield text

    def finalize_response(state, **kwargs):
        assert kwargs["capture_reasoning"] is False
        return ChatResponse(content="complete")

    async def on_done(response):
        order.append(("done", response.content))

    output = [
        text
        async for text in adapter._run_async_stream(
            events(),
            state,
            consume_event=consume_event,
            finalize_response=finalize_response,
            effective_schema=None,
            response_model=None,
            on_delta=lambda text: order.append(("delta", text)),
            on_tool_call=None,
            on_done=on_done,
            on_chunk=None,
            capture_reasoning=False,
            on_reasoning=None,
        )
    ]

    assert output == ["hello"]
    assert order == [("delta", "hello"), ("done", "complete")]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_chunk_callbacks_are_serial_and_yield_follows_callbacks(adapter):
    order = []

    async def on_chunk(chunk):
        order.append(("chunk", chunk.text))

    def on_delta(text):
        order.append(("delta", text))

    async for text in adapter._emit_async_stream_chunks(
        [make_chunk()],
        on_chunk,
        on_delta,
    ):
        order.append(("yield", text))

    assert order == [
        ("chunk", "hello"),
        ("delta", "hello"),
        ("yield", "hello"),
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_completion_flushes_then_calls_tools_and_done(adapter):
    order = []
    buffer = StreamChunkBuffer(buffer_chars=10)
    assert list(buffer.add("hello")) == []
    response = ChatResponse(
        content="hello",
        tool_calls=[ToolCall(name="weather", arguments={})],
    )

    async def on_chunk(chunk):
        order.append(("chunk", chunk.text))

    def on_delta(text):
        order.append(("delta", text))

    async def on_tool_call(tool_call):
        order.append(("tool", tool_call.name))

    def on_done(done_response):
        order.append(("done", done_response.content))

    output = []
    async for text in adapter._complete_async_stream(
        response,
        buffer,
        on_chunk,
        on_delta,
        on_tool_call,
        on_done,
    ):
        output.append(text)

    assert output == ["hello"]
    assert order == [
        ("chunk", "hello"),
        ("delta", "hello"),
        ("tool", "weather"),
        ("done", "hello"),
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_reasoning_callback_observes_response_update(adapter):
    response = ChatResponse()
    collector = StreamReasoningCollector(clock=iter([0.0, 0.1]).__next__)
    observed = []

    async def on_reasoning(event):
        observed.append((event.text, len(response.reasoning_events)))

    event = await adapter._record_async_reasoning_event(
        response,
        collector,
        "thinking",
        capture_reasoning=True,
        on_reasoning=on_reasoning,
    )

    assert event is not None
    assert response.reasoning_events == [event]
    assert observed == [("thinking", 1)]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_provider_event_wrapper_preserves_provider_errors(adapter):
    async def events():
        yield {"type": "first"}
        raise LLMAPIError(detail="provider failed")

    with pytest.raises(LLMAPIError, match="provider failed"):
        [event async for event in adapter._aiter_provider_stream_events(events())]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_provider_event_wrapper_does_not_catch_cancellation(adapter):
    async def events():
        raise asyncio.CancelledError
        yield  # pragma: no cover

    with pytest.raises(asyncio.CancelledError):
        [event async for event in adapter._aiter_provider_stream_events(events())]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_callback_errors_propagate_unchanged(adapter):
    expected = RuntimeError("callback failed")

    async def on_delta(_):
        raise expected

    with pytest.raises(RuntimeError) as exc_info:
        [
            text
            async for text in adapter._emit_async_stream_chunks(
                [make_chunk()],
                None,
                on_delta,
            )
        ]

    assert exc_info.value is expected


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_stream_close_before_completion_skips_tool_and_done(adapter):
    order = []
    buffer = StreamChunkBuffer(buffer_chars=10)
    list(buffer.add("hello"))
    response = ChatResponse(
        content="hello",
        tool_calls=[ToolCall(name="weather", arguments={})],
    )

    async def on_chunk(_):
        order.append("chunk")

    async def on_delta(_):
        order.append("delta")

    async def on_tool_call(_):
        order.append("tool")

    async def on_done(_):
        order.append("done")

    stream = adapter._complete_async_stream(
        response,
        buffer,
        on_chunk,
        on_delta,
        on_tool_call,
        on_done,
    )
    assert await stream.__anext__() == "hello"
    await stream.aclose()

    assert order == ["chunk", "delta"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_stream_cancellation_skips_completion_callbacks(adapter):
    entered = asyncio.Event()
    never = asyncio.Event()
    order = []
    buffer = StreamChunkBuffer(buffer_chars=10)
    list(buffer.add("hello"))
    response = ChatResponse(content="hello")

    async def on_done(_):
        order.append("done")

    async def consume():
        async for text in adapter._complete_async_stream(
            response,
            buffer,
            None,
            None,
            None,
            on_done,
        ):
            order.append(text)
            entered.set()
            await never.wait()

    task = asyncio.create_task(consume())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert order == ["hello"]
