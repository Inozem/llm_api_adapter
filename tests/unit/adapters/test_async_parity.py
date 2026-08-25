import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, Type
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.llm_api_adapter.adapters.anthropic.adapter import AnthropicAdapter
from src.llm_api_adapter.adapters.google.adapter import GoogleAdapter
from src.llm_api_adapter.adapters.openai.adapter import OpenAIAdapter
from src.llm_api_adapter.errors.llm_api_error import LLMAPITimeoutError
from src.llm_api_adapter.llm_registry.llm_registry import Pricing
from src.llm_api_adapter.llms.anthropic.async_client import ClaudeAsyncClient
from src.llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from src.llm_api_adapter.llms.google.async_client import GeminiAsyncClient
from src.llm_api_adapter.llms.google.sync_client import GeminiSyncClient
from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.models.messages.file_parts import ImagePart
from src.llm_api_adapter.models.responses.chat_response import Usage
from src.llm_api_adapter.models.tools import ToolSpec


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

LOOKUP_TOOL = ToolSpec(
    name="lookup_weather",
    description="Look up the weather.",
    json_schema={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)


@dataclass(frozen=True)
class OrganizationCase:
    name: str
    organization: str
    adapter_cls: Type[Any]
    sync_client_cls: Type[Any]
    async_client_cls: Type[Any]
    completion_method: str
    model: str
    max_tokens: int
    message_key: str
    tools_key: str
    response_factory: Callable[[], Dict[str, Any]]
    stream_factory: Callable[[], AsyncIterator[SSEEvent]]
    reasoning_level: str | int
    expected_reasoning_payload: Dict[str, Any]


def _openai_response() -> Dict[str, Any]:
    return {
        "id": "resp_123",
        "model": "gpt-5",
        "created_at": 123,
        "status": "completed",
        "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
        "output": [
            {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Plan"}],
            },
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": '{"answer":"ok"}'}
                ],
            },
        ],
    }


def _anthropic_response() -> Dict[str, Any]:
    return {
        "id": "msg_123",
        "model": "claude-sonnet-4-5",
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": "Plan"},
            {"type": "text", "text": '{"answer":"ok"}'},
        ],
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }


def _google_response() -> Dict[str, Any]:
    return {
        "modelVersion": "gemini-2.5-pro",
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "Plan", "thought": True},
                    {"text": '{"answer":"ok"}'},
                ]
            },
            "finishReason": "STOP",
        }],
        "usageMetadata": {
            "promptTokenCount": 2,
            "candidatesTokenCount": 1,
            "totalTokenCount": 3,
        },
    }


async def _openai_stream() -> AsyncIterator[SSEEvent]:
    yield SSEEvent(
        event="response.reasoning_summary_text.delta",
        data={
            "type": "response.reasoning_summary_text.delta",
            "delta": "Plan",
        },
    )
    yield SSEEvent(
        event="response.output_text.delta",
        data={"type": "response.output_text.delta", "delta": "Hel"},
    )
    yield SSEEvent(
        event="response.output_text.delta",
        data={"type": "response.output_text.delta", "delta": "lo!"},
    )
    yield SSEEvent(
        event="response.completed",
        data={
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "model": "gpt-5",
                "status": "completed",
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 2,
                    "total_tokens": 4,
                },
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"type": "summary_text", "text": "Plan"}],
                    },
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Hello!"}],
                    },
                ],
            },
        },
    )


async def _anthropic_stream() -> AsyncIterator[SSEEvent]:
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
    for text in ("Hel", "lo!"):
        yield SSEEvent(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "text_delta", "text": text},
            },
        )
    yield SSEEvent(
        event="message_delta",
        data={
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 2},
        },
    )
    yield SSEEvent(event="message_stop", data={"type": "message_stop"})


async def _google_stream() -> AsyncIterator[SSEEvent]:
    yield SSEEvent(
        event=None,
        data={
            "candidates": [{
                "content": {"parts": [{"text": "Plan", "thought": True}]}
            }]
        },
    )
    yield SSEEvent(
        event=None,
        data={"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
    )
    yield SSEEvent(
        event=None,
        data={
            "candidates": [{
                "content": {"parts": [{"text": "lo!"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 2,
                "candidatesTokenCount": 2,
                "totalTokenCount": 4,
            },
        },
    )


PROVIDERS = (
    pytest.param(
        OrganizationCase(
            name="openai",
            organization="openai",
            adapter_cls=OpenAIAdapter,
            sync_client_cls=OpenAISyncClient,
            async_client_cls=OpenAIAsyncClient,
            completion_method="complete",
            model="gpt-5.5",
            max_tokens=64,
            message_key="input",
            tools_key="tools",
            response_factory=_openai_response,
            stream_factory=_openai_stream,
            reasoning_level="very_high",
            expected_reasoning_payload={"reasoning_effort": "xhigh"},
        ),
        id="openai",
    ),
    pytest.param(
        OrganizationCase(
            name="anthropic-numeric",
            organization="anthropic",
            adapter_cls=AnthropicAdapter,
            sync_client_cls=ClaudeSyncClient,
            async_client_cls=ClaudeAsyncClient,
            completion_method="chat_completion",
            model="claude-sonnet-4-5",
            max_tokens=64_000,
            message_key="messages",
            tools_key="tools",
            response_factory=_anthropic_response,
            stream_factory=_anthropic_stream,
            reasoning_level="medium",
            expected_reasoning_payload={"budget_tokens": 32_512},
        ),
        id="anthropic-numeric",
    ),
    pytest.param(
        OrganizationCase(
            name="anthropic-adaptive",
            organization="anthropic",
            adapter_cls=AnthropicAdapter,
            sync_client_cls=ClaudeSyncClient,
            async_client_cls=ClaudeAsyncClient,
            completion_method="chat_completion",
            model="claude-sonnet-4-6",
            max_tokens=64,
            message_key="messages",
            tools_key="tools",
            response_factory=_anthropic_response,
            stream_factory=_anthropic_stream,
            reasoning_level="very_high",
            expected_reasoning_payload={"effort": "max"},
        ),
        id="anthropic-adaptive",
    ),
    pytest.param(
        OrganizationCase(
            name="google-numeric",
            organization="google",
            adapter_cls=GoogleAdapter,
            sync_client_cls=GeminiSyncClient,
            async_client_cls=GeminiAsyncClient,
            completion_method="chat_completion",
            model="gemini-2.5-pro",
            max_tokens=64,
            message_key="contents",
            tools_key="tools",
            response_factory=_google_response,
            stream_factory=_google_stream,
            reasoning_level="medium",
            expected_reasoning_payload={
                "thinkingBudget": 16_448,
                "includeThoughts": True,
            },
        ),
        id="google-numeric",
    ),
    pytest.param(
        OrganizationCase(
            name="google-categorical",
            organization="google",
            adapter_cls=GoogleAdapter,
            sync_client_cls=GeminiSyncClient,
            async_client_cls=GeminiAsyncClient,
            completion_method="chat_completion",
            model="gemini-3.1-pro-preview",
            max_tokens=64,
            message_key="contents",
            tools_key="tools",
            response_factory=_google_response,
            stream_factory=_google_stream,
            reasoning_level="very_high",
            expected_reasoning_payload={
                "thinkingLevel": "high",
                "includeThoughts": True,
            },
        ),
        id="google-categorical",
    ),
)


def _make_adapter(case: OrganizationCase):
    adapter = case.adapter_cls(api_key="test_api_key", model=case.model)
    adapter.pricing = Pricing.from_dict(
        [
            {
                "up_to_prompt_tokens": 200,
                "input_per_1m": 1.0,
                "output_per_1m": 2.0,
            },
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 3.0,
                "output_per_1m": 4.0,
            }
        ],
        currency="EUR",
    )
    return adapter


def _chat_kwargs(case: OrganizationCase) -> Dict[str, Any]:
    return {
        "messages": [UserMessage("hi")],
        "max_tokens": case.max_tokens,
        "json_schema": RESPONSE_SCHEMA,
        "capture_reasoning": True,
        "reasoning_level": case.reasoning_level,
    }


def _reasoning_payload(case: OrganizationCase, payload: Dict[str, Any]) -> Dict[str, Any]:
    if case.organization == "google":
        return payload["generationConfig"]["thinkingConfig"]
    if case.organization == "openai":
        return {"reasoning_effort": payload["reasoning_effort"]}
    return {
        key: payload[key]
        for key in ("budget_tokens", "effort")
        if key in payload
    }


def _response_vector(response) -> tuple:
    return (
        response.content,
        response.usage,
        response.parsed_json,
        [(event.text, event.kind) for event in response.reasoning_events],
        response.currency,
        response.cost_input,
        response.cost_output,
        response.cost_total,
    )


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", PROVIDERS)
async def test_async_chat_matches_sync_response_contract(case):
    sync_adapter = _make_adapter(case)
    async_adapter = _make_adapter(case)
    response = case.response_factory()

    sync_completion = Mock(return_value=response)
    with patch.object(
        case.sync_client_cls,
        case.completion_method,
        new=sync_completion,
    ):
        sync_response = sync_adapter.chat(**_chat_kwargs(case))

    async_completion = AsyncMock(return_value=case.response_factory())
    with patch.object(
        case.async_client_cls,
        case.completion_method,
        new=async_completion,
    ):
        async_response = await async_adapter.achat(**_chat_kwargs(case))

    assert _reasoning_payload(
        case,
        sync_completion.call_args.kwargs,
    ) == case.expected_reasoning_payload
    assert _reasoning_payload(
        case,
        async_completion.await_args.kwargs,
    ) == case.expected_reasoning_payload
    assert _response_vector(async_response) == _response_vector(sync_response)
    assert async_response.usage == Usage(input_tokens=2, output_tokens=1, total_tokens=3)
    assert async_response.parsed_json == {"answer": "ok"}
    assert [event.text for event in async_response.reasoning_events] == ["Plan"]
    assert async_response.cost_total == pytest.approx(4e-6)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", PROVIDERS)
async def test_async_chat_accepts_common_files_and_tools(case):
    adapter = _make_adapter(case)
    message = UserMessage(
        "describe this image",
        files=[ImagePart(data=b"image-bytes", media_type="image/png")],
    )

    with patch.object(
        case.async_client_cls,
        case.completion_method,
        new=AsyncMock(return_value=case.response_factory()),
    ) as completion:
        response = await adapter.achat(
            [message],
            max_tokens=case.max_tokens,
            tools=[LOOKUP_TOOL],
            tool_choice="auto",
        )

    assert response.content
    payload = completion.await_args.kwargs
    assert payload[case.message_key]
    assert payload[case.tools_key]


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", PROVIDERS)
async def test_async_stream_has_common_chunks_usage_reasoning_and_callbacks(case):
    adapter = _make_adapter(case)
    chunks = []
    done = []
    order = []

    async def on_reasoning(event):
        order.append(("reasoning", event.text))

    async def on_chunk(chunk):
        chunks.append(chunk)
        order.append(("chunk", chunk.text))

    async def on_delta(text):
        order.append(("delta", text))

    async def on_done(response):
        done.append(response)
        order.append(("done", response.content))

    stream = Mock(return_value=case.stream_factory())
    with patch.object(
        case.async_client_cls,
        "stream",
        new=stream,
    ):
        yielded = []
        async for text in adapter.astream_chat(
            [UserMessage("hi")],
            max_tokens=case.max_tokens,
            reasoning_level=case.reasoning_level,
            buffer_chars=4,
            capture_reasoning=True,
            on_reasoning=on_reasoning,
            on_chunk=on_chunk,
            on_delta=on_delta,
            on_done=on_done,
        ):
            yielded.append(text)
            order.append(("yield", text))

    assert _reasoning_payload(
        case,
        stream.call_args.kwargs,
    ) == case.expected_reasoning_payload
    assert yielded == ["Hell", "o!"]
    assert [chunk.text for chunk in chunks] == yielded
    assert [chunk.index for chunk in chunks] == [0, 1]
    assert chunks[-1].usage == Usage(input_tokens=2, output_tokens=2, total_tokens=4)
    assert done[0].usage == Usage(input_tokens=2, output_tokens=2, total_tokens=4)
    assert done[0].cost_input == pytest.approx(2e-6)
    assert done[0].cost_output == pytest.approx(4e-6)
    assert done[0].cost_total == pytest.approx(6e-6)
    assert [event.text for event in done[0].reasoning_events] == ["Plan"]
    assert order == [
        ("reasoning", "Plan"),
        ("chunk", "Hell"),
        ("delta", "Hell"),
        ("yield", "Hell"),
        ("chunk", "o!"),
        ("delta", "o!"),
        ("yield", "o!"),
        ("done", "Hello!"),
    ]


@pytest.mark.unit
@pytest.mark.parametrize("case", PROVIDERS)
def test_sync_stream_uses_the_same_reasoning_payload(case, monkeypatch):
    adapter = _make_adapter(case)
    consumer_name = (
        "_consume_responses_stream"
        if case.organization == "openai"
        else "_consume_stream"
    )
    monkeypatch.setattr(adapter, consumer_name, lambda *args, **kwargs: iter(()))
    stream = Mock(return_value=iter(()))

    with patch.object(case.sync_client_cls, "stream", new=stream):
        assert list(
            adapter.stream_chat(
                [UserMessage("hi")],
                max_tokens=case.max_tokens,
                reasoning_level=case.reasoning_level,
                capture_reasoning=True,
            )
        ) == []

    assert _reasoning_payload(
        case,
        stream.call_args.kwargs,
    ) == case.expected_reasoning_payload


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", PROVIDERS)
async def test_async_chat_preserves_provider_errors(case):
    adapter = _make_adapter(case)
    expected = LLMAPITimeoutError(detail=f"{case.name} timeout")

    with patch.object(
        case.async_client_cls,
        case.completion_method,
        new=AsyncMock(side_effect=expected),
    ):
        with pytest.raises(LLMAPITimeoutError) as exc_info:
            await adapter.achat(
                [UserMessage("hi")],
                max_tokens=case.max_tokens,
            )

    assert exc_info.value is expected


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("case", PROVIDERS)
async def test_async_stream_propagates_cancellation_without_on_done(case):
    adapter = _make_adapter(case)
    done = []

    async def cancelled_events():
        raise asyncio.CancelledError
        yield  # pragma: no cover

    with patch.object(
        case.async_client_cls,
        "stream",
        new=Mock(return_value=cancelled_events()),
    ):
        with pytest.raises(asyncio.CancelledError):
            [
                text
                async for text in adapter.astream_chat(
                    [UserMessage("hi")],
                    max_tokens=case.max_tokens,
                    on_done=done.append,
                )
            ]

    assert done == []
