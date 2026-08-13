from unittest.mock import patch, Mock

import pytest
import requests

from src.llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIClientError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from src.llm_api_adapter.llms.anthropic.sync_client import ClaudeSyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent


class StreamingResponse:
    def __init__(self, lines, status_code=200, error_payload=None):
        self.lines = lines
        self.status_code = status_code
        self.error_payload = error_payload or {}
        self.close = Mock()

    def iter_lines(self, decode_unicode=True):
        return iter(self.lines)

    def json(self):
        return self.error_payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

@pytest.fixture
def client():
    return ClaudeSyncClient(api_key="test_api_key")

@pytest.fixture
def mock_post_success():
    mock_response = Mock()
    mock_response.json.return_value = {
        "completion": "Hello"
    }
    mock_response.raise_for_status = Mock()
    with patch(
        "src.llm_api_adapter.llms.anthropic.sync_client.requests.post",
        return_value=mock_response
    ) as mock_post:
        yield mock_post, mock_response

@pytest.mark.unit
def test_chat_completion_success(client, mock_post_success):
    mock_post, _ = mock_post_success
    result = client.chat_completion(
        "claude-sonnet-4-5", messages=[{"role": "user", "content": "Hi"}]
    )
    assert isinstance(result, dict)
    assert "completion" in result
    assert result["completion"] == "Hello"
    mock_post.assert_called_once()
    headers = mock_post.call_args[1]["headers"]
    assert headers["x-api-key"] == "test_api_key"

@pytest.mark.parametrize("exception,expected_exception", [
    (requests.exceptions.Timeout("timeout"), LLMAPITimeoutError),
    (requests.exceptions.RequestException("generic error"), LLMAPIClientError),
])
@pytest.mark.unit
@patch("src.llm_api_adapter.llms.anthropic.sync_client.requests.post")
def test_send_request_exceptions(
    mock_post, client, exception, expected_exception
):
    mock_post.side_effect = exception
    with pytest.raises(expected_exception):
        client._send_request("http://example.com", {})

@pytest.mark.parametrize("status_code, error_type, expected_exception", [
    (401, "invalid_api_key", LLMAPIAuthorizationError),
    (429, "rate_limit_exceeded", LLMAPIRateLimitError),
    (400, "bad_request", LLMAPIClientError),
    (500, "server_error", LLMAPIServerError),
])
@pytest.mark.unit
@patch("src.llm_api_adapter.llms.openai.sync_client.requests.post")
def test_send_request_http_errors(
    mock_post, client, status_code, error_type, expected_exception
):
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.json.return_value = {
        "error": {"type": error_type, "message": "Error message"}
    }
    http_err = requests.exceptions.HTTPError(response=mock_response)
    mock_post.side_effect = http_err
    with pytest.raises(expected_exception):
        client._send_request("http://example.com", {})

@pytest.mark.unit
@patch("src.llm_api_adapter.llms.anthropic.sync_client.requests.post")
def test_send_request_fallback_error_parsing(mock_post, client):
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.json.side_effect = ValueError("Invalid JSON")
    http_err = requests.exceptions.HTTPError(response=mock_response)
    mock_post.side_effect = http_err
    with pytest.raises(LLMAPIClientError):
        client._send_request("http://example.com", {})


# ---------------------------
# _prepare_chat_payload_for_model
# ---------------------------

@pytest.mark.unit
def test_prepare_payload_adaptive_thinking_strips_top_p_and_sets_thinking(client):
    kwargs = {"messages": [], "top_p": 1.0, "is_adaptive_thinking": True, "effort": "high"}
    payload = client._prepare_chat_payload_for_model("claude-opus-4-8", kwargs)
    assert "top_p" not in payload
    assert "is_adaptive_thinking" not in payload
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"]["effort"] == "high"


@pytest.mark.unit
def test_prepare_payload_adaptive_thinking_no_effort_strips_top_p_only(client):
    kwargs = {"messages": [], "top_p": 1.0, "is_adaptive_thinking": True}
    payload = client._prepare_chat_payload_for_model("claude-opus-4-8", kwargs)
    assert "top_p" not in payload
    assert "thinking" not in payload
    assert "is_adaptive_thinking" not in payload


@pytest.mark.unit
def test_prepare_payload_legacy_sets_budget_tokens_thinking(client):
    kwargs = {"messages": [], "budget_tokens": 4096, "is_adaptive_thinking": False}
    payload = client._prepare_chat_payload_for_model("claude-opus-4-5", kwargs)
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert "is_adaptive_thinking" not in payload


@pytest.mark.unit
def test_prepare_payload_capture_reasoning_requests_summarized_thinking(client):
    kwargs = {
        "messages": [],
        "budget_tokens": 4096,
        "is_adaptive_thinking": False,
        "capture_reasoning": True,
    }
    payload = client._prepare_chat_payload_for_model("claude-opus-4-5", kwargs)
    assert payload["thinking"] == {
        "type": "enabled",
        "budget_tokens": 4096,
        "display": "summarized",
    }
    assert "capture_reasoning" not in payload


@pytest.mark.unit
def test_prepare_payload_is_adaptive_thinking_never_in_result(client):
    for flag in (True, False):
        kwargs = {"messages": [], "is_adaptive_thinking": flag}
        payload = client._prepare_chat_payload_for_model("claude-opus-4-8", kwargs)
        assert "is_adaptive_thinking" not in payload


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_yields_raw_messages_events_and_ignores_ping(mock_post, client):
    response = StreamingResponse([
        "event: message_start",
        'data: {"type":"message_start","message":{"id":"msg_123","content":[]}}',
        "",
        "event: ping",
        'data: {"type":"ping"}',
        "",
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
        "",
        "event: message_stop",
        'data: {"type":"message_stop"}',
        "",
    ])
    mock_post.return_value = response

    events = list(client.stream(
        "claude-opus-4-8",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=64,
    ))

    assert events == [
        SSEEvent(
            event="message_start",
            data={"type": "message_start", "message": {"id": "msg_123", "content": []}},
        ),
        SSEEvent(
            event="content_block_delta",
            data={
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        ),
        SSEEvent(event="message_stop", data={"type": "message_stop"}),
    ]
    assert mock_post.call_args.args[0] == "https://api.anthropic.com/v1/messages"
    assert mock_post.call_args.kwargs["json"] == {
        "model": "claude-opus-4-8",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 64,
        "stream": True,
    }
    assert mock_post.call_args.kwargs["stream"] is True
    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_reuses_adaptive_thinking_payload_preparation(mock_post, client):
    response = StreamingResponse([])
    mock_post.return_value = response

    assert list(client.stream(
        "claude-opus-4-8",
        messages=[],
        top_p=1.0,
        is_adaptive_thinking=True,
        effort="high",
    )) == []

    assert mock_post.call_args.kwargs["json"] == {
        "model": "claude-opus-4-8",
        "messages": [],
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
        "stream": True,
    }


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_preserves_input_json_deltas_for_later_tool_assembly(mock_post, client):
    response = StreamingResponse([
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\\"city\\\": \\\"Tel"}}',
        "",
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" Aviv\\\"}"}}',
        "",
        "event: content_block_stop",
        'data: {"type":"content_block_stop","index":1}',
        "",
    ])
    mock_post.return_value = response

    events = list(client.stream("claude-opus-4-8", messages=[]))

    assert [event.data["delta"]["partial_json"] for event in events[:2]] == [
        '{"city": "Tel',
        ' Aviv"}',
    ]
    assert events[-1] == SSEEvent(
        event="content_block_stop",
        data={"type": "content_block_stop", "index": 1},
    )


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_closes_response_when_consumer_stops_early(mock_post, client):
    response = StreamingResponse([
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
        "",
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
        "",
    ])
    mock_post.return_value = response

    events = client.stream("claude-opus-4-8", messages=[])
    next(events)
    events.close()

    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_maps_http_errors_through_anthropic_error_handler(mock_post, client):
    response = StreamingResponse(
        [],
        status_code=429,
        error_payload={"error": {"type": "rate_limit_error", "message": "Slow down"}},
    )
    mock_post.return_value = response

    with pytest.raises(LLMAPIRateLimitError, match="Slow down"):
        list(client.stream("claude-opus-4-8", messages=[]))

    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_maps_in_stream_error_through_anthropic_error_handler(mock_post, client):
    response = StreamingResponse([
        "event: error",
        'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}',
        "",
    ])
    mock_post.return_value = response

    with pytest.raises(LLMAPIServerError, match="Overloaded"):
        list(client.stream("claude-opus-4-8", messages=[]))

    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_passes_unknown_event_types_through(mock_post, client):
    response = StreamingResponse([
        "event: future_event",
        'data: {"type":"future_event","value":"kept"}',
        "",
    ])
    mock_post.return_value = response

    assert list(client.stream("claude-opus-4-8", messages=[])) == [
        SSEEvent(event="future_event", data={"type": "future_event", "value": "kept"})
    ]
