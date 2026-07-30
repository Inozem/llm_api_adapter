from unittest.mock import AsyncMock, patch

import pytest

from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent


@pytest.fixture
def client():
    return OpenAIAsyncClient(api_key="test_api_key")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_complete_uses_responses_api_and_reuses_payload_normalization(client):
    request = AsyncMock(return_value={"id": "resp_123"})

    with patch(
        "src.llm_api_adapter.llms.openai.async_client.async_request",
        request,
    ):
        result = await client.complete(
            "gpt-5",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
            reasoning_effort="high",
            capture_reasoning=True,
        )

    assert result == {"id": "resp_123"}
    request.assert_awaited_once()
    url = request.await_args.args[0]
    payload = request.await_args.kwargs["payload"]
    assert url.endswith("/responses")
    assert payload == {
        "model": "gpt-5",
        "input": [{"role": "user", "content": "Hi"}],
        "max_output_tokens": 10,
        "reasoning": {"effort": "high", "summary": "auto"},
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_complete_uses_chat_completions_for_legacy_models(client):
    request = AsyncMock(return_value={"choices": []})

    with patch(
        "src.llm_api_adapter.llms.openai.async_client.async_request",
        request,
    ):
        await client.complete(
            "gpt-4o",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
        )

    payload = request.await_args.kwargs["payload"]
    assert request.await_args.args[0].endswith("/chat/completions")
    assert payload == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_stream_selects_responses_and_closes_event_iterator(client):
    closed = []

    async def source():
        try:
            yield SSEEvent(
                event="response.output_text.delta",
                data={"type": "response.output_text.delta", "delta": "Hi"},
            )
        finally:
            closed.append(True)

    with patch(
        "src.llm_api_adapter.llms.openai.async_client.async_stream_request",
        return_value=source(),
    ) as request:
        events = [event async for event in client.stream("gpt-5", messages=[])]

    assert events == [
        SSEEvent(
            event="response.output_text.delta",
            data={"type": "response.output_text.delta", "delta": "Hi"},
        )
    ]
    assert request.call_args.args[0].endswith("/responses")
    assert request.call_args.kwargs["payload"]["stream"] is True
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_stream_closes_event_iterator_when_consumer_stops(client):
    closed = []

    async def source():
        try:
            yield SSEEvent(event=None, data={"text": "first"})
            yield SSEEvent(event=None, data={"text": "second"})
        finally:
            closed.append(True)

    with patch(
        "src.llm_api_adapter.llms.openai.async_client.async_stream_request",
        return_value=source(),
    ):
        events = client.stream("gpt-4o", messages=[])
        assert await events.__anext__() == SSEEvent(
            event=None,
            data={"text": "first"},
        )
        await events.aclose()

    assert closed == [True]
