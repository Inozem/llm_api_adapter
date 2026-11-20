"""Native Claude adapter ownership of the shared prepared runners."""

from __future__ import annotations

import asyncio
import pytest

from src.llm_api_adapter.adapters.anthropic.adapter import AnthropicAdapter
from src.llm_api_adapter.adapters.anthropic_compatible.adapter import AnthropicCompatibleAdapter
from src.llm_api_adapter.adapters.anthropic_compatible.execution import _MessagesPreparedRunners
from src.llm_api_adapter.adapters.anthropic_compatible.execution import _PreparedMessagesOperation
from src.llm_api_adapter.adapters.anthropic_compatible import execution as common_execution
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.models.responses.chat_response import ChatResponse
from src.llm_api_adapter.llms.transports import SSEEvent

pytestmark = pytest.mark.unit


def test_native_adapter_is_compatible_subclass_and_keeps_native_dataclass_shape():
    from dataclasses import fields

    assert issubclass(AnthropicAdapter, AnthropicCompatibleAdapter)
    assert [field.name for field in fields(AnthropicAdapter)] == [
        "api_key", "model", "company", "transport", "pricing",
        "is_reasoning", "is_adaptive_thinking", "service_provider", "model_spec",
    ]
    adapter = AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5")
    assert "base_url" not in repr(adapter)
    with pytest.raises(TypeError):
        AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5", base_url="https://example.invalid")


def test_native_operation_is_private_and_generic_operation_shape_is_unchanged():
    from dataclasses import fields

    assert [field.name for field in fields(_PreparedMessagesOperation)] == [
        "chat_context", "transport_request", "client", "capture_reasoning",
    ]
    adapter = AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5")
    operation = adapter._prepare_native_operation(
        [UserMessage(content="sensitive prompt")], max_tokens=10,
        temperature=1.0, top_p=1.0, reasoning_level=None, timeout_s=None,
        tools=None, tool_choice=None, parallel_tool_calls=None, json_schema=None,
        response_model=None, capture_reasoning=False, stream=True,
    )
    assert "sensitive prompt" not in repr(operation)
    with pytest.raises(TypeError):
        operation.call_kwargs["model"] = "modified"


def test_direct_legacy_stream_consumer_uses_common_consumption_helper(monkeypatch):
    adapter = AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5")
    original = common_execution._MessagesPreparedRunners._consume_prepared_source
    calls = []

    def spy(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(common_execution._MessagesPreparedRunners, "_consume_prepared_source", spy)
    done = []
    output = list(adapter._consume_stream(
        iter(_native_text_events()), None, None, None, None, done.append,
        None, None, False, None,
    ))
    assert output == ["x"]
    assert len(done) == 1
    assert done[0].content == "x"
    assert len(calls) == 1


def test_sync_stream_accepts_exact_original_consumer_override(monkeypatch):
    adapter = AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5")
    stream_calls = []
    consumer_calls = []

    def consume(events, effective_schema, response_model, on_delta, on_tool_call,
                on_done, buffer_chars, on_chunk, capture_reasoning, on_reasoning):
        consumer_calls.append((events, effective_schema, response_model))
        return iter(())

    class SyncClient:
        def __init__(self, **_kwargs): pass
        def stream(self, **kwargs):
            stream_calls.append(kwargs)
            return iter(())

    import src.llm_api_adapter.adapters.anthropic.adapter as native
    monkeypatch.setattr(native, "ClaudeSyncClient", SyncClient)
    monkeypatch.setattr(adapter, "_consume_stream", consume)
    assert list(adapter.stream_chat([], 16)) == []
    assert len(stream_calls) == 1
    assert len(consumer_calls) == 1


@pytest.mark.parametrize("method", ["chat", "achat", "stream_chat", "astream_chat"])
def test_native_public_methods_enter_shared_prepared_runner(monkeypatch, method):
    adapter = AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5")
    calls = []
    runner_name = {
        "chat": "_run_prepared_chat", "achat": "_run_prepared_achat",
        "stream_chat": "_run_prepared_stream", "astream_chat": "_run_prepared_astream",
    }[method]
    original = getattr(_MessagesPreparedRunners, runner_name)

    def spy(self, operation, *args, **kwargs):
        calls.append(operation)
        return original(self, operation, *args, **kwargs)

    monkeypatch.setattr(_MessagesPreparedRunners, runner_name, spy)

    class SyncClient:
        def __init__(self, **_kwargs): pass
        def chat_completion(self, **_kwargs):
            return {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 1}}
        def stream(self, **_kwargs):
            from src.llm_api_adapter.llms.transports import SSEEvent
            return iter([
                SSEEvent(event="message_start", data={"type": "message_start", "message": {"model": adapter.model, "content": [], "usage": {"input_tokens": 1}}}),
                SSEEvent(event="content_block_start", data={"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
                SSEEvent(event="content_block_delta", data={"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
                SSEEvent(event="content_block_stop", data={"type": "content_block_stop", "index": 0}),
                SSEEvent(event="message_delta", data={"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
            ])

    class AsyncClient(SyncClient):
        async def chat_completion(self, **_kwargs):
            return super().chat_completion()
        async def stream(self, **kwargs):
            for event in super().stream(**kwargs):
                yield event

    import src.llm_api_adapter.adapters.anthropic.adapter as native
    monkeypatch.setattr(native, "ClaudeSyncClient", SyncClient)
    monkeypatch.setattr(native, "ClaudeAsyncClient", AsyncClient)

    async def consume_async(iterator):
        return [item async for item in iterator]

    if method == "chat":
        result = adapter.chat([UserMessage(content="hi")], max_tokens=10)
        assert isinstance(result, ChatResponse)
    elif method == "achat":
        assert isinstance(asyncio.run(adapter.achat([UserMessage(content="hi")], max_tokens=10)), ChatResponse)
    elif method == "stream_chat":
        assert list(adapter.stream_chat([UserMessage(content="hi")], max_tokens=10)) == ["ok"]
    else:
        assert asyncio.run(consume_async(adapter.astream_chat([UserMessage(content="hi")], max_tokens=10))) == ["ok"]
    assert len(calls) == 1


def _native_text_events():
    return [
        SSEEvent(event="message_start", data={"type": "message_start", "message": {"model": "claude-sonnet-4-5", "content": []}}),
        SSEEvent(event="content_block_start", data={"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        SSEEvent(event="content_block_delta", data={"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "x"}}),
    ]


@pytest.mark.asyncio
async def test_native_public_astream_aclose_immediately_closes_source(monkeypatch):
    adapter = AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5")
    closed = []
    done = []

    async def source():
        try:
            for event in _native_text_events():
                yield event
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    class AsyncClient:
        def __init__(self, **_kwargs): pass
        def stream(self, **_kwargs): return source()

    import src.llm_api_adapter.adapters.anthropic.adapter as native
    monkeypatch.setattr(native, "ClaudeAsyncClient", AsyncClient)
    stream = adapter.astream_chat([UserMessage(content="hi")], max_tokens=10, buffer_chars=1, on_done=done.append)
    try:
        assert await anext(stream) == "x"
        await stream.aclose()
        assert closed == [True]
        assert done == []
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_native_public_astream_cancel_preserves_cancel_over_cleanup_error(monkeypatch):
    adapter = AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5")
    entered_wait = asyncio.Event()
    closed = []
    done = []
    cleanup_failure = RuntimeError("secondary source cleanup")

    async def source():
        try:
            for event in _native_text_events():
                yield event
        finally:
            closed.append(True)
            raise cleanup_failure

    class AsyncClient:
        def __init__(self, **_kwargs): pass
        def stream(self, **_kwargs): return source()

    import src.llm_api_adapter.adapters.anthropic.adapter as native
    monkeypatch.setattr(native, "ClaudeAsyncClient", AsyncClient)

    async def wait_in_callback(_text):
        entered_wait.set()
        await asyncio.Event().wait()

    stream = adapter.astream_chat(
        [UserMessage(content="hi")], max_tokens=10, on_delta=wait_in_callback,
        on_done=done.append,
    )

    async def drain():
        async for _chunk in stream:
            pass

    task = asyncio.create_task(drain())
    try:
        await asyncio.wait_for(entered_wait.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert closed == [True]
        assert done == []
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, RuntimeError):
            pass
        await stream.aclose()


@pytest.mark.asyncio
async def test_native_public_astream_callback_error_wins_over_cleanup_error(monkeypatch):
    adapter = AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5")
    closed = []
    done = []
    callback_failure = RuntimeError("caller callback failure")
    cleanup_failure = RuntimeError("secondary source cleanup")

    async def source():
        try:
            for event in _native_text_events():
                yield event
        finally:
            closed.append(True)
            raise cleanup_failure

    class AsyncClient:
        def __init__(self, **_kwargs): pass
        def stream(self, **_kwargs): return source()

    import src.llm_api_adapter.adapters.anthropic.adapter as native
    monkeypatch.setattr(native, "ClaudeAsyncClient", AsyncClient)

    def fail(_text): raise callback_failure

    stream = adapter.astream_chat([UserMessage(content="hi")], max_tokens=10, on_delta=fail, on_done=done.append)
    try:
        with pytest.raises(RuntimeError) as raised:
            await anext(stream)
        assert raised.value is callback_failure
        assert closed == [True]
        assert done == []
    finally:
        await stream.aclose()
