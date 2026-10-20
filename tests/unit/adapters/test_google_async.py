from unittest.mock import AsyncMock, patch

import pytest

from src.llm_api_adapter.adapters.google_adapter import GoogleAdapter
from src.llm_api_adapter.llms.google.async_client import GeminiAsyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent
from src.llm_api_adapter.models.messages.chat_message import UserMessage


@pytest.fixture
def adapter():
    return GoogleAdapter(
        api_key="test_api_key",
        model="gemini-2.5-pro",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_achat_uses_async_client_and_preserves_response_contract(adapter):
    response = {
        "modelVersion": "gemini-2.5-pro",
        "candidates": [{
            "content": {"parts": [{"text": "Hello"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 2,
            "candidatesTokenCount": 1,
            "totalTokenCount": 3,
        },
    }
    chat_completion = AsyncMock(return_value=response)

    with patch.object(GeminiAsyncClient, "chat_completion", chat_completion):
        result = await adapter.achat([UserMessage("hi")])

    assert result.content == "Hello"
    assert result.usage.input_tokens == 2
    chat_completion.assert_awaited_once()
    assert chat_completion.await_args.kwargs["model"] == "gemini-2.5-pro"
    assert chat_completion.await_args.kwargs["contents"] == [
        {"role": "user", "parts": [{"text": "hi"}]}
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_achat_extracts_provider_reported_automatic_cache_read_usage(adapter):
    response = {
        "modelVersion": "gemini-2.5-pro",
        "candidates": [{
            "content": {"parts": [{"text": "cached"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 2,
            "totalTokenCount": 17,
            "cachedContentTokenCount": 4,
        },
    }
    chat_completion = AsyncMock(return_value=response)

    with patch.object(GeminiAsyncClient, "chat_completion", chat_completion):
        result = await adapter.achat([UserMessage("hi")])

    assert result.usage.input_tokens == 15
    assert result.usage.cached_tokens == 4
    chat_completion.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_achat_preserves_corrected_completed_usage_and_prices_it(adapter):
    response = {
        "modelVersion": "gemini-2.5-pro",
        "candidates": [{
            "content": {"parts": [{"text": "completed"}]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 3,
            "thoughtsTokenCount": 2,
            "totalTokenCount": 25,
        },
    }

    with patch.object(
        GeminiAsyncClient,
        "chat_completion",
        AsyncMock(return_value=response),
    ):
        result = await adapter.achat([UserMessage("hi")])

    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 5
    assert result.usage.total_tokens == 25
    assert result.cost_input == pytest.approx(13.75e-6)
    assert result.cost_output == pytest.approx(50e-6)
    assert result.cost_total == pytest.approx(63.75e-6)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_astream_chat_awaits_reasoning_and_completion_callbacks(adapter):
    async def events():
        yield SSEEvent(
            event=None,
            data={
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Plan", "thought": True}]
                    }
                }],
            },
        )
        yield SSEEvent(
            event=None,
            data={
                "candidates": [{
                    "content": {"parts": [{"text": "Answer"}]}
                }],
            },
        )
        yield SSEEvent(
            event=None,
            data={
                "candidates": [{
                    "content": {"parts": []},
                    "finishReason": "STOP",
                }],
            },
        )

    order = []

    async def on_reasoning(event):
        order.append(("reasoning", event.text))

    async def on_delta(text):
        order.append(("delta", text))

    async def on_done(response):
        order.append(("done", response.content))

    with patch.object(GeminiAsyncClient, "stream", return_value=events()):
        output = [
            text
            async for text in adapter.astream_chat(
                [UserMessage("hi")],
                capture_reasoning=True,
                on_reasoning=on_reasoning,
                on_delta=on_delta,
                on_done=on_done,
            )
        ]

    assert output == ["Answer"]
    assert order == [
        ("reasoning", "Plan"),
        ("delta", "Answer"),
        ("done", "Answer"),
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_astream_chat_extracts_automatic_cache_read_usage(adapter):
    async def events():
        yield SSEEvent(
            event=None,
            data={
                "candidates": [{
                    "content": {"parts": [{"text": "cached"}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 15,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 17,
                    "cachedContentTokenCount": 4,
                },
            },
        )

    done = []
    with patch.object(GeminiAsyncClient, "stream", return_value=events()):
        output = [
            text
            async for text in adapter.astream_chat(
                [UserMessage("hi")],
                on_done=done.append,
            )
        ]

    assert output == ["cached"]
    assert done[0].usage.cached_tokens == 4


@pytest.mark.asyncio
@pytest.mark.unit
async def test_astream_chat_retains_usage_updates_and_completes_without_visible_text(adapter):
    async def events():
        yield SSEEvent(
            event=None,
            data={
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 3,
                    "thoughtsTokenCount": 2,
                    "totalTokenCount": 25,
                    "cachedContentTokenCount": 4,
                }
            },
        )
        yield SSEEvent(
            event=None,
            data={
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 4,
                    "totalTokenCount": 30,
                    "cachedContentTokenCount": None,
                },
                "candidates": [{
                    "content": {"parts": []},
                    "finishReason": "STOP",
                }],
            },
        )

    done = []
    with patch.object(
        GeminiAsyncClient,
        "stream",
        return_value=events(),
    ):
        output = [
            text
            async for text in adapter.astream_chat(
                [UserMessage("hi")],
                on_done=done.append,
            )
        ]

    assert output == []
    assert len(done) == 1
    assert done[0].usage.input_tokens == 12
    assert done[0].usage.output_tokens == 6
    assert done[0].usage.total_tokens == 30
    assert done[0].usage.cached_tokens is None
    assert done[0].cost_input is None
    assert done[0].cost_output == pytest.approx(60e-6)
    assert done[0].cost_total is None
