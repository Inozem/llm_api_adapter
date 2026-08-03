import json
from unittest.mock import patch

import httpx
import pytest
import respx
import requests
import requests_mock
from pydantic import BaseModel

from src.llm_api_adapter.errors.llm_api_error import JSONSchemaError
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.models.responses.chat_response import Usage
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


def _sse(*events: tuple[str | None, dict]) -> str:
    chunks = []
    for event_name, payload in events:
        lines = []
        if event_name is not None:
            lines.append(f"event: {event_name}")
        lines.append(f"data: {json.dumps(payload)}")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks) + "\n\n"


STRUCTURED_JSON = '{"answer":"ok"}'
STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class StreamAnswer(BaseModel):
    answer: str


def _structured_events(provider: str, content: str):
    first, second = content[:10], content[10:]
    if provider == "openai":
        return [
            (
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": first},
            ),
            (
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": second},
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": "structured_resp_123",
                        "model": "gpt-5",
                        "status": "completed",
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {"type": "output_text", "text": content}
                                ],
                            }
                        ],
                    },
                },
            ),
        ]
    if provider == "anthropic":
        return [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "structured_msg_123",
                        "model": "claude-sonnet-4-5",
                        "content": [],
                        "usage": {"input_tokens": 2, "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": first},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": second},
                },
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 2},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
    if provider == "google":
        return [
            (
                None,
                {"candidates": [{"content": {"parts": [{"text": first}]}}]},
            ),
            (
                None,
                {
                    "candidates": [{
                        "content": {"parts": [{"text": second}]},
                        "finishReason": "STOP",
                    }],
                    "usageMetadata": {
                        "promptTokenCount": 2,
                        "candidatesTokenCount": 2,
                        "totalTokenCount": 4,
                    },
                },
            ),
        ]
    raise AssertionError(f"Unsupported provider: {provider}")


def _content_fragments(content: str):
    return [fragment for fragment in (content[:10], content[10:]) if fragment]


_STRUCTURED_STREAM_CASES = [
    pytest.param(
        "openai",
        "gpt-5",
        "https://api.openai.com/v1/responses",
        {"max_tokens": 64},
        id="openai-responses",
    ),
    pytest.param(
        "anthropic",
        "claude-sonnet-4-5",
        "https://api.anthropic.com/v1/messages",
        {"max_tokens": 64},
        id="anthropic",
    ),
    pytest.param(
        "google",
        "gemini-2.5-pro",
        (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        ),
        {},
        id="google",
    ),
]


@pytest.mark.integration
def test_openai_responses_streams_through_universal_adapter():
    body = _sse(
        (
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "Hello"},
        ),
        (
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_123",
                    "model": "gpt-5",
                    "status": "completed",
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3,
                    },
                    "output": [{
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    }],
                },
            },
        ),
    )
    with (
        requests_mock.Mocker() as mock,
        patch(
            "src.llm_api_adapter.llms.streaming.requests.post",
            wraps=requests.post,
        ) as mock_post,
    ):
        mock.post(
            "https://api.openai.com/v1/responses",
            text=body,
            headers={"Content-Type": "text/event-stream"},
        )
        adapter = UniversalLLMAPIAdapter(
            organization="openai",
            model="gpt-5",
            api_key="dummy_key",
        )
        done = []

        with pytest.warns(
            UserWarning,
            match="Parameter 'top_p' is not supported for model 'gpt-5'",
        ):
            assert list(adapter.stream_chat([UserMessage("Hi")], on_done=done.append)) == ["Hello"]

    request = mock.last_request
    assert mock_post.call_args.args[0] == "https://api.openai.com/v1/responses"
    assert mock_post.call_args.kwargs["stream"] is True
    assert request.headers["Authorization"] == "Bearer dummy_key"
    assert request.headers["Content-Type"] == "application/json"
    assert request.json()["stream"] is True
    assert request.json()["input"] == [{"role": "user", "content": "Hi"}]
    assert done[0].response_id == "resp_123"
    assert done[0].usage.total_tokens == 3


@pytest.mark.integration
def test_anthropic_messages_streams_through_universal_adapter():
    body = _sse(
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "model": "claude-sonnet-4-5",
                    "content": [],
                    "usage": {"input_tokens": 2, "output_tokens": 0},
                },
            },
        ),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        ),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    with (
        requests_mock.Mocker() as mock,
        patch(
            "src.llm_api_adapter.llms.streaming.requests.post",
            wraps=requests.post,
        ) as mock_post,
    ):
        mock.post(
            "https://api.anthropic.com/v1/messages",
            text=body,
            headers={"Content-Type": "text/event-stream"},
        )
        adapter = UniversalLLMAPIAdapter(
            organization="anthropic",
            model="claude-sonnet-4-5",
            api_key="dummy_key",
        )
        done = []

        assert list(adapter.stream_chat(
            [UserMessage("Hi")],
            max_tokens=64,
            on_done=done.append,
        )) == ["Hello"]

    request = mock.last_request
    assert mock_post.call_args.args[0] == "https://api.anthropic.com/v1/messages"
    assert mock_post.call_args.kwargs["stream"] is True
    assert request.headers["x-api-key"] == "dummy_key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.headers["Content-Type"] == "application/json"
    assert request.json()["stream"] is True
    assert request.json()["messages"] == [{"role": "user", "content": "Hi"}]
    assert done[0].content == "Hello"
    assert done[0].usage.total_tokens == 3


@pytest.mark.integration
def test_google_streams_through_universal_adapter():
    body = _sse(
        (
            None,
            {
                "candidates": [{
                    "content": {"parts": [{"text": "Hello"}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {
                    "promptTokenCount": 2,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 3,
                },
            },
        ),
    )
    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-pro:streamGenerateContent?alt=sse"
    )
    with (
        requests_mock.Mocker() as mock,
        patch(
            "src.llm_api_adapter.llms.streaming.requests.post",
            wraps=requests.post,
        ) as mock_post,
    ):
        mock.post(url, text=body, headers={"Content-Type": "text/event-stream"})
        adapter = UniversalLLMAPIAdapter(
            organization="google",
            model="gemini-2.5-pro",
            api_key="dummy_key",
        )
        done = []

        assert list(adapter.stream_chat([UserMessage("Hi")], on_done=done.append)) == ["Hello"]

    request = mock.last_request
    assert mock_post.call_args.args[0] == url
    assert mock_post.call_args.kwargs["stream"] is True
    assert request.headers["x-goog-api-key"] == "dummy_key"
    assert request.headers["Content-Type"] == "application/json"
    assert request.json()["contents"] == [{"role": "user", "parts": [{"text": "Hi"}]}]
    assert done[0].content == "Hello"
    assert done[0].usage.total_tokens == 3


_BUFFERED_STREAM_SCENARIOS = [
    pytest.param(
        "openai",
        "gpt-5",
        "https://api.openai.com/v1/responses",
        [
            (
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "Hel"},
            ),
            (
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "lo!"},
            ),
            (
                "response.completed",
                {
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
                        "output": [{
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Hello!"}],
                        }],
                    },
                },
            ),
        ],
        {},
        Usage(input_tokens=2, output_tokens=2, total_tokens=4),
        id="openai-responses",
    ),
    pytest.param(
        "anthropic",
        "claude-sonnet-4-5",
        "https://api.anthropic.com/v1/messages",
        [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_123",
                        "model": "claude-sonnet-4-5",
                        "content": [],
                        "usage": {"input_tokens": 2, "output_tokens": 0},
                    },
                },
            ),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hel"},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "lo!"},
                },
            ),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 2},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ],
        {"max_tokens": 64},
        Usage(input_tokens=2, output_tokens=2, total_tokens=4),
        id="anthropic",
    ),
    pytest.param(
        "google",
        "gemini-2.5-pro",
        (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        ),
        [
            (
                None,
                {"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]},
            ),
            (
                None,
                {
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
            ),
        ],
        {},
        Usage(input_tokens=2, output_tokens=2, total_tokens=4),
        id="google",
    ),
]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("organization", "model", "url", "events", "stream_kwargs", "expected_usage"),
    _BUFFERED_STREAM_SCENARIOS,
)
def test_streaming_contract_is_consistent_for_all_providers(
    organization,
    model,
    url,
    events,
    stream_kwargs,
    expected_usage,
):
    body = _sse(*events)
    with requests_mock.Mocker() as mock:
        mock.post(url, text=body, headers={"Content-Type": "text/event-stream"})
        adapter = UniversalLLMAPIAdapter(
            organization=organization,
            model=model,
            api_key="dummy_key",
        )
        chunks = []
        completed_responses = []
        callback_order = []
        yielded = []

        def on_chunk(chunk):
            chunks.append(chunk)
            callback_order.append(("chunk", chunk.text))

        def on_done(response):
            completed_responses.append(response)
            callback_order.append(("done", response.content))

        for text in adapter.stream_chat(
            [UserMessage("Hi")],
            buffer_chars=4,
            on_chunk=on_chunk,
            on_delta=lambda text: callback_order.append(("delta", text)),
            on_done=on_done,
            **stream_kwargs,
        ):
            yielded.append(text)
            callback_order.append(("yield", text))

    assert yielded == ["Hell", "o!"]
    assert "".join(yielded) == "Hello!"
    assert [chunk.text for chunk in chunks] == yielded
    assert all(len(chunk.text) <= 4 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == [0, 1]
    assert [chunk.elapsed_s for chunk in chunks] == sorted(
        chunk.elapsed_s for chunk in chunks
    )
    assert all(chunk.delta_s >= 0 for chunk in chunks)
    assert chunks[-1].usage == expected_usage
    assert any(chunk.output_tokens_delta is not None for chunk in chunks)
    assert completed_responses[0].usage == expected_usage
    assert callback_order == [
        ("chunk", "Hell"),
        ("delta", "Hell"),
        ("yield", "Hell"),
        ("chunk", "o!"),
        ("delta", "o!"),
        ("yield", "o!"),
        ("done", "Hello!"),
    ]


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model", "url", "stream_kwargs"),
    _STRUCTURED_STREAM_CASES,
)
@pytest.mark.parametrize(
    ("option_name", "option_value"),
    [
        pytest.param("json_schema", STRUCTURED_SCHEMA, id="json-schema"),
        pytest.param("response_model", StreamAnswer, id="response-model"),
    ],
)
async def test_async_structured_stream_finalization(
    provider,
    model,
    url,
    stream_kwargs,
    option_name,
    option_value,
):
    fragments = _content_fragments(STRUCTURED_JSON)
    with respx.mock(assert_all_called=False) as router:
        route = router.post(url)
        route.return_value = httpx.Response(
            200,
            text=_sse(*_structured_events(provider, STRUCTURED_JSON)),
            headers={"content-type": "text/event-stream"},
        )
        adapter = UniversalLLMAPIAdapter(
            organization=provider,
            model=model,
            api_key="dummy_key",
        )
        done = []
        order = []

        async def on_delta(text):
            order.append(("delta", text))

        async def on_done(response):
            done.append(response)
            order.append(("done", response.content))

        yielded = []
        yielded_kwargs = {
            "messages": [UserMessage("Return a structured answer.")],
            **stream_kwargs,
            option_name: option_value,
            "on_delta": on_delta,
            "on_done": on_done,
        }
        async for text in adapter.astream_chat(**yielded_kwargs):
            yielded.append(text)
            order.append(("yield", text))

    assert route.call_count == 1
    assert yielded == fragments
    assert order == [
        ("delta", fragments[0]),
        ("yield", fragments[0]),
        ("delta", fragments[1]),
        ("yield", fragments[1]),
        ("done", STRUCTURED_JSON),
    ]
    assert len(done) == 1
    assert done[0].content == STRUCTURED_JSON
    assert done[0].parsed_json == {"answer": "ok"}
    if option_name == "response_model":
        assert done[0].parsed_model == StreamAnswer(answer="ok")
    else:
        assert done[0].parsed_model is None


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize(
    ("provider", "model", "url", "stream_kwargs"),
    _STRUCTURED_STREAM_CASES,
)
@pytest.mark.parametrize(
    ("option_name", "option_value", "invalid_content"),
    [
        pytest.param(
            "json_schema",
            STRUCTURED_SCHEMA,
            '{"answer":',
            id="json-schema-invalid-json",
        ),
        pytest.param(
            "response_model",
            StreamAnswer,
            "{}",
            id="response-model-validation-error",
        ),
    ],
)
async def test_async_structured_stream_failure_skips_on_done(
    provider,
    model,
    url,
    stream_kwargs,
    option_name,
    option_value,
    invalid_content,
):
    fragments = _content_fragments(invalid_content)
    with respx.mock(assert_all_called=False) as router:
        route = router.post(url)
        route.return_value = httpx.Response(
            200,
            text=_sse(*_structured_events(provider, invalid_content)),
            headers={"content-type": "text/event-stream"},
        )
        adapter = UniversalLLMAPIAdapter(
            organization=provider,
            model=model,
            api_key="dummy_key",
        )
        done = []
        yielded = []
        yielded_kwargs = {
            "messages": [UserMessage("Return a structured answer.")],
            **stream_kwargs,
            option_name: option_value,
            "on_done": done.append,
        }
        with pytest.raises(JSONSchemaError):
            async for text in adapter.astream_chat(**yielded_kwargs):
                yielded.append(text)

    assert route.call_count == 1
    assert yielded == fragments
    assert done == []
