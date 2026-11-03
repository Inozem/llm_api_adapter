from unittest.mock import AsyncMock, patch

import pytest

from src.llm_api_adapter.adapters.openai_adapter import OpenAIAdapter
from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.llms.transports import SSEFrameDecoder
from src.llm_api_adapter.llms.streaming import SSEEvent
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.models.responses.chat_response import ChatResponse
from tests.fixtures.openai_compatible import text_response, text_sse_lines


class _MarkedNormalizerAdapter(OpenAIAdapter):
    def __post_init__(self):
        super().__post_init__()
        self.normalized_responses = []

    def _normalize_chat_completions_response(self, response):
        self.normalized_responses.append(response)
        return ChatResponse(
            model=response.get("model"),
            response_id="normalized-by-override",
            content=response["choices"][0]["message"]["content"],
        )


class _RejectingNormalizerAdapter(OpenAIAdapter):
    def _normalize_chat_completions_response(self, response):
        raise ValueError("normalization rejected")


class _ForbiddenNormalizerAdapter(OpenAIAdapter):
    def _normalize_chat_completions_response(self, response):
        raise AssertionError("Chat Completions normalizer used for Responses API")


def _legacy_adapter(cls=_MarkedNormalizerAdapter):
    return cls(api_key="test_api_key", model="gpt-4o")


def _legacy_response(content="Answer"):
    return text_response(content)


def _decode_sse_lines(lines):
    decoder = SSEFrameDecoder()
    events = [event for line in lines if (event := decoder.feed(line)) is not None]
    final_event = decoder.finish()
    if final_event is not None:
        events.append(final_event)
    return events


def _legacy_events(content="Answer"):
    return iter(_decode_sse_lines(text_sse_lines(content)))


def _responses_response():
    return {
        "id": "resp_test",
        "model": "gpt-5",
        "status": "completed",
        "output": [{"type": "message", "content": [
            {"type": "output_text", "text": "Answer"}
        ]}],
    }


def _responses_events(response):
    return [
        SSEEvent(
            event="response.output_text.delta",
            data={"type": "response.output_text.delta", "delta": "Answer"},
        ),
        SSEEvent(
            event="response.completed",
            data={"type": "response.completed", "response": response},
        ),
    ]


@pytest.mark.unit
def test_sync_chat_completions_paths_use_one_overridable_normalizer():
    adapter = _legacy_adapter()
    with patch.object(
        OpenAISyncClient, "complete", return_value=_legacy_response()
    ), patch.object(OpenAISyncClient, "stream", return_value=_legacy_events()):
        ordinary = adapter.chat([UserMessage("hi")])
        completed = []
        streamed = list(adapter.stream_chat(
            [UserMessage("hi")], on_done=completed.append
        ))

    assert ordinary.response_id == "normalized-by-override"
    assert ordinary.content == "Answer"
    assert streamed == ["Answer"]
    assert completed[0].response_id == "normalized-by-override"
    assert completed[0].content == "Answer"
    assert len(adapter.normalized_responses) == 2
    assert adapter.normalized_responses[1]["choices"][0]["message"]["content"] == "Answer"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_chat_completions_paths_use_one_overridable_normalizer():
    adapter = _legacy_adapter()

    async def events():
        for event in _legacy_events():
            yield event

    with patch.object(
        OpenAIAsyncClient, "complete", new=AsyncMock(return_value=_legacy_response())
    ), patch.object(OpenAIAsyncClient, "stream", return_value=events()):
        ordinary = await adapter.achat([UserMessage("hi")])
        completed = []
        streamed = [
            text async for text in adapter.astream_chat(
                [UserMessage("hi")], on_done=completed.append
            )
        ]

    assert ordinary.response_id == "normalized-by-override"
    assert ordinary.content == "Answer"
    assert streamed == ["Answer"]
    assert completed[0].response_id == "normalized-by-override"
    assert completed[0].content == "Answer"
    assert len(adapter.normalized_responses) == 2


@pytest.mark.unit
def test_responses_chat_completions_normalizer_is_bypassed():
    adapter = _ForbiddenNormalizerAdapter(api_key="test_api_key", model="gpt-5")
    response = _responses_response()
    events = iter(_responses_events(response))
    with patch.object(OpenAISyncClient, "complete", return_value=response), patch.object(
        OpenAISyncClient, "stream", return_value=events
    ):
        assert adapter.chat([UserMessage("hi")]).content == "Answer"
        assert list(adapter.stream_chat([UserMessage("hi")])) == ["Answer"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_responses_chat_completions_normalizer_is_bypassed():
    adapter = _ForbiddenNormalizerAdapter(api_key="test_api_key", model="gpt-5")
    response = _responses_response()

    async def events():
        for event in _responses_events(response):
            yield event

    with patch.object(
        OpenAIAsyncClient, "complete", new=AsyncMock(return_value=response)
    ), patch.object(OpenAIAsyncClient, "stream", return_value=events()):
        assert (await adapter.achat([UserMessage("hi")])).content == "Answer"
        assert [text async for text in adapter.astream_chat([UserMessage("hi")])] == ["Answer"]


@pytest.mark.unit
def test_sync_stream_normalization_failure_precedes_buffer_flush_and_completion():
    adapter = _legacy_adapter(_RejectingNormalizerAdapter)
    closed = []

    def events():
        try:
            yield from _legacy_events("pending")
        finally:
            closed.append(True)

    chunks = []
    done = []
    with patch.object(OpenAISyncClient, "stream", return_value=events()), pytest.raises(
        ValueError, match="normalization rejected"
    ):
        list(adapter.stream_chat(
            [UserMessage("hi")], buffer_chars=32,
            on_chunk=chunks.append, on_done=done.append,
        ))

    assert chunks == []
    assert done == []
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_stream_normalization_failure_precedes_buffer_flush_and_completion():
    adapter = _legacy_adapter(_RejectingNormalizerAdapter)
    closed = []

    async def events():
        try:
            for event in _legacy_events("pending"):
                yield event
        finally:
            closed.append(True)

    chunks = []
    done = []
    with patch.object(OpenAIAsyncClient, "stream", return_value=events()):
        with pytest.raises(ValueError, match="normalization rejected"):
            async for _ in adapter.astream_chat(
                [UserMessage("hi")], buffer_chars=32,
                on_chunk=chunks.append, on_done=done.append,
            ):
                pass

    assert chunks == []
    assert done == []
    assert closed == [True]
