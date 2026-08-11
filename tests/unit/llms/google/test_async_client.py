from unittest.mock import AsyncMock, patch

import pytest

from src.llm_api_adapter.llms.google.async_client import GeminiAsyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent


@pytest.fixture
def client():
    return GeminiAsyncClient(api_key="test_api_key")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_chat_completion_uses_async_transport_and_payload_normalization(client):
    request = AsyncMock(return_value={"candidates": [{"content": "Hello"}]})

    with patch(
        "src.llm_api_adapter.llms.google.async_client.async_request",
        request,
    ):
        result = await client.chat_completion(
            "gemini-2.5-flash-lite",
            contents=[],
            generationConfig={"thinkingConfig": {"thinkingBudget": 1}},
        )

    assert result == {"candidates": [{"content": "Hello"}]}
    request.assert_awaited_once()
    assert request.await_args.args[0].endswith(
        "/models/gemini-2.5-flash-lite:generateContent"
    )
    assert request.await_args.kwargs["headers"] == {
        "x-goog-api-key": "test_api_key",
        "Content-Type": "application/json",
    }
    assert request.await_args.kwargs["payload"] == {
        "model": "gemini-2.5-flash-lite",
        "contents": [],
        "generationConfig": {"thinkingConfig": {"thinkingBudget": 1}},
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_stream_closes_event_iterator_when_consumer_stops(client):
    closed = []

    async def source():
        try:
            yield SSEEvent(
                event=None,
                data={"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]},
            )
        finally:
            closed.append(True)

    with patch(
        "src.llm_api_adapter.llms.google.async_client.async_stream_request",
        return_value=source(),
    ) as request:
        events = client.stream("gemini-2.5-flash", contents=[])
        assert await events.__anext__() == SSEEvent(
            event=None,
            data={"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]},
        )
        await events.aclose()

    assert request.call_args.args[0].endswith(
        "/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    )
    assert request.call_args.kwargs["payload"] == {
        "model": "gemini-2.5-flash",
        "contents": [],
    }
    assert closed == [True]
