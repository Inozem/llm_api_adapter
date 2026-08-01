from unittest.mock import AsyncMock, patch

import pytest

from src.llm_api_adapter.adapters.anthropic_adapter import AnthropicAdapter
from src.llm_api_adapter.llms.anthropic.async_client import ClaudeAsyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent
from src.llm_api_adapter.models.messages.chat_message import UserMessage


@pytest.fixture
def adapter():
    return AnthropicAdapter(
        api_key="test_api_key",
        model="claude-sonnet-4-5",
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_achat_uses_async_client_and_preserves_response_contract(adapter):
    response = {
        "id": "msg_123",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "Hello"}],
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }
    chat_completion = AsyncMock(return_value=response)

    with patch.object(ClaudeAsyncClient, "chat_completion", chat_completion):
        result = await adapter.achat([UserMessage("hi")], max_tokens=64)

    assert result.content == "Hello"
    assert result.usage.input_tokens == 2
    chat_completion.assert_awaited_once()
    assert chat_completion.await_args.kwargs["model"] == "claude-sonnet-4-5"
    assert chat_completion.await_args.kwargs["messages"] == [
        {"role": "user", "content": "hi"}
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_astream_chat_awaits_reasoning_and_completion_callbacks(adapter):
    async def events():
        yield SSEEvent(
            event="message_start",
            data={
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "model": "claude-sonnet-4-5",
                    "content": [],
                    "usage": {"input_tokens": 2, "output_tokens": 0},
                },
            },
        )
        yield SSEEvent(
            event="content_block_start",
            data={
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": ""},
            },
        )
        yield SSEEvent(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Plan"},
            },
        )
        yield SSEEvent(
            event="content_block_start",
            data={
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "text", "text": ""},
            },
        )
        yield SSEEvent(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": "Answer"},
            },
        )
        yield SSEEvent(
            event="message_delta",
            data={
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 3},
            },
        )
        yield SSEEvent(event="message_stop", data={"type": "message_stop"})

    order = []

    async def on_reasoning(event):
        order.append(("reasoning", event.text))

    async def on_delta(text):
        order.append(("delta", text))

    async def on_done(response):
        order.append(("done", response.content))

    with patch.object(ClaudeAsyncClient, "stream", return_value=events()):
        output = [
            text
            async for text in adapter.astream_chat(
                [UserMessage("hi")],
                max_tokens=2048,
                reasoning_level=1024,
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
