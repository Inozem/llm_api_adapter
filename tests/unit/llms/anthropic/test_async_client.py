from unittest.mock import AsyncMock, Mock, patch
from dataclasses import fields

import pytest

from src.llm_api_adapter.llms.anthropic.async_client import ClaudeAsyncClient
from src.llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from src.llm_api_adapter.llms.anthropic import async_client as async_client_module
from src.llm_api_adapter.adapters.anthropic_adapter import ClaudeAsyncClient as ShimAsyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent


@pytest.fixture
def client():
    return ClaudeAsyncClient(api_key="test_api_key")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_chat_completion_uses_async_transport_and_payload_normalization(client):
    request = AsyncMock(return_value={"id": "msg_123"})

    with patch(
        "src.llm_api_adapter.llms.anthropic.async_client.async_request",
        request,
    ):
        result = await client.chat_completion(
            "claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Hi"}],
            budget_tokens=2048,
            capture_reasoning=True,
        )

    assert result == {"id": "msg_123"}
    request.assert_awaited_once()
    assert request.await_args.args[0].endswith("/messages")
    assert request.await_args.kwargs["headers"] == {
        "x-api-key": "test_api_key",
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    assert request.await_args.kwargs["payload"] == {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": "Hi"}],
        "thinking": {
            "type": "enabled",
            "budget_tokens": 2048,
            "display": "summarized",
        },
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_stream_closes_event_iterator_when_consumer_stops(client):
    closed = []

    async def source():
        try:
            yield SSEEvent(event="message_start", data={"type": "message_start"})
            yield SSEEvent(event="message_stop", data={"type": "message_stop"})
        finally:
            closed.append(True)

    with patch(
        "src.llm_api_adapter.llms.anthropic.async_client.async_stream_request",
        return_value=source(),
    ) as request:
        events = client.stream("claude-sonnet-4-5", messages=[])
        assert await events.__anext__() == SSEEvent(
            event="message_start",
            data={"type": "message_start"},
        )
        await events.aclose()

    assert request.call_args.args[0].endswith("/messages")
    assert request.call_args.kwargs["payload"]["stream"] is True
    assert closed == [True]


@pytest.mark.unit
def test_async_client_inherits_sync_constructor_and_dataclass_contract():
    client = ClaudeAsyncClient("secret-key", "https://proxy.test/v1", "v-test", "requests")
    equivalent = ClaudeAsyncClient("secret-key", "https://proxy.test/v1", "v-test", "requests")

    assert ShimAsyncClient is ClaudeAsyncClient
    assert issubclass(ClaudeAsyncClient, ClaudeSyncClient)
    assert client == equivalent
    assert client._sync_transport is not equivalent._sync_transport
    assert client != ClaudeAsyncClient("secret-key", "https://other.test/v1", "v-test", "requests")
    assert client.endpoint == "https://proxy.test/v1"
    assert client.api_version == "v-test"
    assert client.transport == "requests"
    assert "secret-key" not in repr(client)
    assert [field.name for field in fields(client)] == [
        "api_key", "endpoint", "api_version", "transport", "_sync_transport",
    ]
    assert [field.init for field in fields(client)] == [True] * 4 + [False]
    assert (fields(client)[-1].repr, fields(client)[-1].compare) == (False, False)
    defaults = ClaudeAsyncClient("test_api_key")
    assert defaults.endpoint == "https://api.anthropic.com/v1"
    assert defaults.api_version == "2023-06-01"
    assert defaults.transport == "requests"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_custom_endpoint_version_timeout_and_bound_error_handlers():
    client = ClaudeAsyncClient(
        "custom-key", endpoint="https://gateway.test/prefix/v9",
        api_version="2025-01-01", transport="requests",
    )
    request = AsyncMock(return_value={"id": "msg_custom"})
    with patch.object(async_client_module, "async_request", request):
        assert await client.chat_completion("custom-model", timeout_s=7, messages=[]) == {"id": "msg_custom"}
    request.assert_awaited_once()
    assert request.await_args.args[0] == "https://gateway.test/prefix/v9/messages"
    assert request.await_args.kwargs["timeout"] == 7
    assert request.await_args.kwargs["headers"] == {
        "x-api-key": "custom-key",
        "anthropic-version": "2025-01-01",
        "Content-Type": "application/json",
    }
    http_handler = request.await_args.kwargs["http_error_handler"]
    assert http_handler.__self__ is client
    assert http_handler.__func__ is ClaudeSyncClient._handle_http_error

    closed = []
    async def source():
        try:
            yield SSEEvent(event="message_start", data={"type": "message_start"})
        finally:
            closed.append(True)

    stream_request = Mock(return_value=source())
    with patch.object(async_client_module, "async_stream_request", stream_request):
        assert [event async for event in client.stream("custom-model", timeout_s=11, messages=[])] == [
            SSEEvent(event="message_start", data={"type": "message_start"}),
        ]
    stream_request.assert_called_once()
    assert stream_request.call_args.args[0] == "https://gateway.test/prefix/v9/messages"
    assert stream_request.call_args.kwargs["timeout"] == 11
    assert stream_request.call_args.kwargs["headers"]["x-api-key"] == "custom-key"
    assert stream_request.call_args.kwargs["headers"]["anthropic-version"] == "2025-01-01"
    stream_http_handler = stream_request.call_args.kwargs["http_error_handler"]
    stream_error_handler = stream_request.call_args.kwargs["stream_error_handler"]
    assert stream_http_handler.__self__ is client
    assert stream_http_handler.__func__ is ClaudeSyncClient._handle_http_error
    assert stream_error_handler.__self__ is client
    assert stream_error_handler.__func__ is ClaudeSyncClient._handle_stream_error
    assert closed == [True]

@pytest.mark.asyncio
@pytest.mark.unit
async def test_native_async_public_methods_reuse_compatible_request_preparation(monkeypatch, client):
    from src.llm_api_adapter.llms.anthropic_compatible.sync_client import (
        AnthropicCompatibleSyncClient,
    )

    original = AnthropicCompatibleSyncClient._request_data
    prepared = []
    payload_hooks = []
    payload_objects = []
    endpoint_at_hook = []
    original_endpoint = client.endpoint
    original_payload = client._prepare_chat_payload_for_model

    def observe_request(self, model, kwargs, *, stream):
        prepared.append(stream)
        return original(self, model, kwargs, stream=stream)

    def observe_payload(model, kwargs):
        payload = original_payload(model, kwargs)
        payload_hooks.append(model)
        payload_objects.append(payload)
        endpoint_at_hook.append(client.endpoint)
        client.endpoint = "https://mutated.test/v9"
        return payload

    monkeypatch.setattr(AnthropicCompatibleSyncClient, "_request_data", observe_request)
    monkeypatch.setattr(client, "_prepare_chat_payload_for_model", observe_payload)
    request = AsyncMock(return_value={"ok": True})

    async def events():
        yield SSEEvent("raw", {})

    stream_request = Mock(return_value=events())
    with patch.object(async_client_module, "async_request", request), patch.object(
        async_client_module, "async_stream_request", stream_request
    ):
        assert await client.chat_completion("claude-opus-4-8", messages=[]) == {"ok": True}
        client.endpoint = original_endpoint
        assert [event async for event in client.stream("claude-opus-4-8", messages=[])] == [
            SSEEvent("raw", {})
        ]

    assert prepared == [False, True]
    assert payload_hooks == ["claude-opus-4-8", "claude-opus-4-8"]
    assert endpoint_at_hook == [original_endpoint, original_endpoint]
    assert request.await_args.args[0] == stream_request.call_args.args[0] == (
        f"{original_endpoint}/messages"
    )
    assert request.await_args.kwargs["payload"] is payload_objects[0]
    assert stream_request.call_args.kwargs["payload"] is payload_objects[1]
    assert request.await_args.kwargs["payload"]["model"] == "claude-opus-4-8"
    assert stream_request.call_args.kwargs["payload"]["stream"] is True


@pytest.mark.asyncio
@pytest.mark.unit
async def test_shared_messages_async_json_dispatch_keeps_native_request_policy(client):
    from src.llm_api_adapter.llms.anthropic_compatible.async_client import (
        _AsyncMessagesExecutionMixin,
    )

    request = AsyncMock(return_value={"minimal": True})
    calls = []
    original = _AsyncMessagesExecutionMixin._execute_prepared_request

    async def observe(self, prepared):
        calls.append(prepared)
        return await original(self, prepared)

    with patch.object(_AsyncMessagesExecutionMixin, "_execute_prepared_request", observe), patch.object(
        async_client_module, "async_request", request
    ):
        result = await client.chat_completion(
            "claude-sonnet-4-5", timeout_s=4.5,
            messages=[{"role": "user", "content": "hello"}],
        )

    assert result == {"minimal": True}
    assert len(calls) == 1
    assert calls[0].url == "https://api.anthropic.com/v1/messages"
    assert calls[0].timeout == 4.5
    assert calls[0].payload["messages"] == [{"role": "user", "content": "hello"}]
    assert request.await_count == 1
    assert request.await_args.kwargs["http_error_handler"] == client._handle_http_error


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("error_kind", ["public", "json"])
async def test_shared_messages_async_json_error_preserves_exception_identity(client, error_kind):
    import json
    from src.llm_api_adapter.errors.llm_api_error import LLMAPIClientError

    error = (
        LLMAPIClientError(detail="native public failure")
        if error_kind == "public"
        else json.JSONDecodeError("malformed", "{", 0)
    )
    request = AsyncMock(side_effect=error)
    with patch.object(async_client_module, "async_request", request):
        with pytest.raises(type(error)) as raised:
            await client.chat_completion("claude-sonnet-4-5", messages=[])
    assert raised.value is error


@pytest.mark.asyncio
@pytest.mark.unit
async def test_shared_messages_async_stream_preserves_late_binding_and_raw_eof(client):
    from src.llm_api_adapter.llms.anthropic_compatible.async_client import (
        _AsyncMessagesExecutionMixin,
    )

    event = SSEEvent(event="future_event", data={"partial_json": "{"})
    source = _raw_async_events(event)
    prepared = []
    helper_calls = []
    original = _AsyncMessagesExecutionMixin._execute_prepared_stream

    def observe(self, request):
        prepared.append(request)
        return original(self, request)

    client.endpoint = "https://proxy.test/prefix/v2"
    with patch.object(_AsyncMessagesExecutionMixin, "_execute_prepared_stream", observe), patch.object(
        async_client_module, "async_stream_request", side_effect=lambda *a, **kw: _record_stream(
            helper_calls, source, *a, **kw
        )
    ):
        events = client.stream("claude-sonnet-4-5", timeout_s=8, messages=[])
        client.api_version = "late-version"
        assert prepared == []
        assert [item async for item in events] == [event]

    assert prepared[0].url == "https://proxy.test/prefix/v2/messages"
    assert prepared[0].timeout == 8
    assert prepared[0].payload["stream"] is True
    assert helper_calls[0]["headers"]["anthropic-version"] == "late-version"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_shared_messages_async_stream_preserves_primary_error_and_closes_once(client):
    from src.llm_api_adapter.llms.anthropic_compatible.async_client import (
        _AsyncMessagesExecutionMixin,
    )
    from tests.fixtures.anthropic_compatible import ControlledAsyncSource

    primary = RuntimeError("source failed")
    cleanup = RuntimeError("cleanup failed")
    source = ControlledAsyncSource(
        [SSEEvent("one", {})], failure=primary, fail_after=1, cleanup_error=cleanup
    )
    with patch.object(async_client_module, "async_stream_request", return_value=source):
        events = client.stream("claude-sonnet-4-5", messages=[])
        assert await events.__anext__() == SSEEvent("one", {})
        with pytest.raises(RuntimeError) as raised:
            await events.__anext__()

    assert source.aclose_calls == 1
    assert raised.value is primary
    assert type(client)._handle_http_error is ClaudeSyncClient._handle_http_error


@pytest.mark.asyncio
@pytest.mark.unit
async def test_shared_messages_async_stream_propagates_normal_cleanup_failure(client):
    from tests.fixtures.anthropic_compatible import ControlledAsyncSource

    source = ControlledAsyncSource(
        [SSEEvent("one", {})], cleanup_error=RuntimeError("close failed")
    )
    with patch.object(async_client_module, "async_stream_request", return_value=source):
        with pytest.raises(RuntimeError, match="close failed"):
            [item async for item in client.stream("claude-sonnet-4-5", messages=[])]
    assert source.aclose_calls == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_shared_messages_async_stream_cancellation_closes_source(client):
    import asyncio
    from tests.fixtures.anthropic_compatible import ControlledAsyncSource

    started = asyncio.Event()
    closed = asyncio.Event()

    class BlockedSource(ControlledAsyncSource):
        async def __anext__(self):
            started.set()
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()
            await super().aclose()

    source = BlockedSource(cleanup_error=RuntimeError("secondary cleanup failure"))
    with patch.object(async_client_module, "async_stream_request", return_value=source):
        events = client.stream("claude-sonnet-4-5", messages=[])
        pending = asyncio.create_task(events.__anext__())
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, timeout=1)
        finally:
            if not pending.done():
                pending.cancel()
                try:
                    await pending
                except asyncio.CancelledError:
                    pass

    assert closed.is_set()
    assert source.aclose_calls == 1


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True])
async def test_shared_messages_async_httpx_mock_transport_uses_native_wire_dispatch(
    client, monkeypatch, streaming
):
    import json
    import httpx
    from types import SimpleNamespace
    from src.llm_api_adapter.llms import async_streaming

    real_async_client = httpx.AsyncClient
    observed = []
    responses = []
    clients = []

    async def handler(request):
        observed.append(request)
        if streaming:
            response = httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b"event: future_event\ndata: {\"type\":\"future_event\"}\n\n",
                request=request,
            )
        else:
            response = httpx.Response(200, json={"ok": True}, request=request)
        responses.append(response)
        return response

    def make_client():
        value = real_async_client(transport=httpx.MockTransport(handler))
        clients.append(value)
        return value

    fake_httpx = SimpleNamespace(
        AsyncClient=make_client,
        TimeoutException=httpx.TimeoutException,
        HTTPStatusError=httpx.HTTPStatusError,
        RequestError=httpx.RequestError,
    )
    monkeypatch.setattr(async_streaming, "_require_httpx", lambda: fake_httpx)
    client.endpoint = "https://wire.test/prefix/v7"
    client.api_version = "wire-version"

    with patch("socket.getaddrinfo", side_effect=AssertionError("DNS must not be used")), patch(
        "requests.post", side_effect=AssertionError("sync HTTP must not be used")
    ):
        if streaming:
            events = [item async for item in client.stream(
                "wire-model", timeout_s=3, messages=[{"role": "user", "content": "x"}]
            )]
            assert events == [SSEEvent("future_event", {"type": "future_event"})]
        else:
            assert await client.chat_completion(
                "wire-model", timeout_s=3, messages=[{"role": "user", "content": "x"}]
            ) == {"ok": True}

    assert len(observed) == 1
    request = observed[0]
    assert str(request.url) == "https://wire.test/prefix/v7/messages"
    assert request.headers["x-api-key"] == "test_api_key"
    assert request.headers["anthropic-version"] == "wire-version"
    assert request.extensions["timeout"]["read"] == 3
    assert json.loads(request.content) == {
        "model": "wire-model",
        "messages": [{"role": "user", "content": "x"}],
        **({"stream": True} if streaming else {}),
    }
    assert len(clients) == 1 and clients[0].is_closed
    assert responses[0].is_closed


async def _raw_async_events(*events):
    for event in events:
        yield event


async def _record_stream(calls, source, *args, **kwargs):
    calls.append({"args": args, **kwargs})
    async for event in source:
        yield event


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("streaming", [False, True])
async def test_shared_messages_async_parent_http_handler_patch_propagates(client, streaming):
    patched = Mock()

    def replacement(self, error):
        patched(self, error)

    request = AsyncMock(return_value={"minimal": True})
    stream_request = Mock(return_value=_raw_async_events(SSEEvent("event", {})))
    with patch.object(ClaudeSyncClient, "_handle_http_error", replacement), patch.object(
        async_client_module, "async_request", request
    ), patch.object(async_client_module, "async_stream_request", stream_request):
        if streaming:
            events = [item async for item in client.stream("claude-sonnet-4-5", messages=[])]
            assert events == [SSEEvent("event", {})]
            handler = stream_request.call_args.kwargs["http_error_handler"]
        else:
            assert await client.chat_completion("claude-sonnet-4-5", messages=[]) == {"minimal": True}
            handler = request.await_args.kwargs["http_error_handler"]

        assert handler.__self__ is client
        assert handler.__func__ is replacement
        sentinel = object()
        handler(sentinel)
        patched.assert_called_once_with(client, sentinel)
