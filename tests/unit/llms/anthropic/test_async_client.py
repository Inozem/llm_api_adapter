from unittest.mock import AsyncMock, patch

import pytest

from src.llm_api_adapter.llms.anthropic.async_client import ClaudeAsyncClient
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
