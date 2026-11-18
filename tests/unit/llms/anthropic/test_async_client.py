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
