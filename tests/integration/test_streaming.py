import json
from unittest.mock import patch

import pytest
import requests
import requests_mock

from src.llm_api_adapter.models.messages.chat_message import UserMessage
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
