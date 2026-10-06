from unittest.mock import AsyncMock, patch

import pytest

from src.llm_api_adapter.adapters.openai_adapter import OpenAIAdapter
from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.models.responses.chat_response import Usage


@pytest.fixture
def adapter():
    return OpenAIAdapter(api_key="test_api_key", model="gpt-5")


@pytest.fixture
def legacy_adapter():
    return OpenAIAdapter(api_key="test_api_key", model="gpt-4o")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_achat_uses_async_client_and_preserves_response_contract(adapter):
    response = {
        "id": "resp_123",
        "model": "gpt-5",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello"}],
            }
        ],
    }
    complete = AsyncMock(return_value=response)

    with patch.object(OpenAIAsyncClient, "complete", complete):
        result = await adapter.achat([UserMessage("hi")], max_tokens=32)

    assert result.content == "Hello"
    complete.assert_awaited_once()
    assert complete.await_args.kwargs["model"] == "gpt-5"
    assert complete.await_args.kwargs["input"] == [
        {"role": "user", "content": "hi"}
    ]
    assert complete.await_args.kwargs["max_tokens"] == 32


@pytest.mark.asyncio
@pytest.mark.unit
async def test_achat_extracts_provider_reported_automatic_cache_usage():
    cache_adapter = OpenAIAdapter(api_key="test_api_key", model="gpt-5.6-sol")
    response = {
        "id": "resp_cache",
        "model": "gpt-5.6-sol",
        "status": "completed",
        "usage": {
            "input_tokens": 15,
            "output_tokens": 2,
            "total_tokens": 17,
            "input_tokens_details": {
                "cached_tokens": 4,
                "cache_write_tokens": 3,
            },
        },
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "cached"}],
        }],
    }
    complete = AsyncMock(return_value=response)

    with patch.object(OpenAIAsyncClient, "complete", complete):
        result = await cache_adapter.achat([UserMessage("hi")])

    assert result.usage.input_tokens == 15
    assert result.usage.cached_tokens == 4
    assert result.usage.cache_write_tokens == 3
    complete.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_astream_chat_orders_async_callbacks_before_yield(legacy_adapter):
    async def events():
            yield SSEEvent(
                event=None,
                data={
                "choices": [
                    {"index": 0, "delta": {"content": "Hello"}},
                ]
            }
        )
            yield SSEEvent(
                event=None,
                data={
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "!"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    order = []

    async def on_chunk(chunk):
        order.append(("chunk", chunk.text))

    async def on_delta(text):
        order.append(("delta", text))

    async def on_done(response):
        order.append(("done", response.content))

    with patch.object(OpenAIAsyncClient, "stream", return_value=events()):
        output = []
        async for text in legacy_adapter.astream_chat(
            [UserMessage("hi")],
            on_chunk=on_chunk,
            on_delta=on_delta,
            on_done=on_done,
        ):
            output.append(text)
            order.append(("yield", text))

    assert output == ["Hello", "!"]
    assert order == [
        ("chunk", "Hello"),
        ("delta", "Hello"),
        ("yield", "Hello"),
        ("chunk", "!"),
        ("delta", "!"),
        ("yield", "!"),
        ("done", "Hello!"),
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_astream_chat_awaits_reasoning_callback_and_keeps_it_hidden(adapter):
    async def events():
        yield SSEEvent(
            event="response.reasoning_summary_text.delta",
            data={
                "type": "response.reasoning_summary_text.delta",
                "delta": "Plan",
            },
        )
        yield SSEEvent(
            event="response.output_text.delta",
            data={"type": "response.output_text.delta", "delta": "Answer"},
        )
        yield SSEEvent(
            event="response.completed",
            data={
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "model": "gpt-5",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": "Answer"}
                            ],
                        }
                    ],
                },
            },
        )

    reasoning = []

    async def on_reasoning(event):
        reasoning.append((event.text, event.kind))

    with patch.object(OpenAIAsyncClient, "stream", return_value=events()):
        output = [
            text
            async for text in adapter.astream_chat(
                [UserMessage("hi")],
                capture_reasoning=True,
                on_reasoning=on_reasoning,
            )
        ]

    assert output == ["Answer"]
    assert reasoning == [("Plan", "summary")]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_astream_chat_copies_final_automatic_cache_usage_to_chunks():
    cache_adapter = OpenAIAdapter(api_key="test_api_key", model="gpt-5.6-sol")

    async def events():
        yield SSEEvent(
            event="response.output_text.delta",
            data={"type": "response.output_text.delta", "delta": "Answer"},
        )
        yield SSEEvent(
            event="response.completed",
            data={
                "type": "response.completed",
                "response": {
                    "id": "resp_cache",
                    "model": "gpt-5.6-sol",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 15,
                        "output_tokens": 2,
                        "total_tokens": 17,
                        "input_tokens_details": {
                            "cached_tokens": 4,
                            "cache_write_tokens": 3,
                        },
                    },
                    "output": [{
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Answer"}],
                    }],
                },
            },
        )

    chunks = []
    done = []
    with patch.object(OpenAIAsyncClient, "stream", return_value=events()):
        output = [
            text
            async for text in cache_adapter.astream_chat(
                [UserMessage("hi")],
                buffer_chars=20,
                on_chunk=chunks.append,
                on_done=done.append,
            )
        ]

    expected_usage = Usage(15, 2, 17, 4, 3)
    assert output == ["Answer"]
    assert chunks[0].usage == expected_usage
    assert done[0].usage == expected_usage
