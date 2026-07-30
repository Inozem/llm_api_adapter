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
