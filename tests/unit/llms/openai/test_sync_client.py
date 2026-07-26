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
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
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
    return OpenAISyncClient(api_key="test_api_key")

@pytest.fixture
def mock_post_success():
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Hello"}}]
    }
    mock_response.raise_for_status = Mock()
    with patch(
        "src.llm_api_adapter.llms.openai.sync_client.requests.post",
        return_value=mock_response
    ) as mock_post:
        yield mock_post, mock_response

@pytest.mark.unit
def test_chat_completion_success(client, mock_post_success):
    mock_post, _ = mock_post_success
    result = client.chat_completion(
        "gpt-4", messages=[{"role": "user", "content": "Hi"}]
    )
    assert isinstance(result, dict)
    assert "choices" in result
    choice_message = result["choices"][0]["message"]["content"]
    assert choice_message == "Hello"
    mock_post.assert_called_once()
    headers = mock_post.call_args[1]["headers"]
    assert headers["Authorization"] == "Bearer test_api_key"

@pytest.mark.parametrize("exception,expected_exception", [
    (requests.exceptions.Timeout("timeout"), LLMAPITimeoutError),
    (requests.exceptions.RequestException("generic error"), LLMAPIClientError),
])
@pytest.mark.unit
@patch("src.llm_api_adapter.llms.openai.sync_client.requests.post")
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
@patch("src.llm_api_adapter.llms.openai.sync_client.requests.post")
def test_send_request_fallback_error_parsing(mock_post, client):
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.json.side_effect = ValueError("Invalid JSON")
    http_err = requests.exceptions.HTTPError(response=mock_response)
    mock_post.side_effect = http_err
    with pytest.raises(LLMAPIClientError):
        client._send_request("http://example.com", {})


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_uses_chat_completions_and_yields_raw_chunks(mock_post, client):
    response = StreamingResponse([
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        "",
        "data: [DONE]",
        "",
    ])
    mock_post.return_value = response

    events = list(client.stream("gpt-4o", messages=[{"role": "user", "content": "Hi"}]))

    assert events == [
        SSEEvent(event=None, data={"choices": [{"delta": {"content": "Hello"}}]})
    ]
    assert mock_post.call_args.args[0] == "https://api.openai.com/v1/chat/completions"
    assert mock_post.call_args.kwargs["json"] == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    assert mock_post.call_args.kwargs["stream"] is True
    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_uses_responses_api_and_preserves_named_events(mock_post, client):
    response = StreamingResponse([
        "event: response.output_text.delta",
        'data: {"type":"response.output_text.delta","delta":"Hello"}',
        "",
    ])
    mock_post.return_value = response

    events = list(client.stream(
        "gpt-5",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=10,
    ))

    assert events == [
        SSEEvent(
            event="response.output_text.delta",
            data={"type": "response.output_text.delta", "delta": "Hello"},
        )
    ]
    assert mock_post.call_args.args[0] == "https://api.openai.com/v1/responses"
    assert mock_post.call_args.kwargs["json"] == {
        "model": "gpt-5",
        "input": [{"role": "user", "content": "Hi"}],
        "max_output_tokens": 10,
        "stream": True,
    }
    response.close.assert_called_once()


@pytest.mark.unit
def test_prepare_responses_payload_warns_when_ignoring_temperature_for_gpt5_nano(client):
    with pytest.warns(
        UserWarning,
        match="Parameter 'temperature' is not supported for model 'gpt-5-nano'",
    ):
        payload = client._prepare_responses_payload_for_model(
            "gpt-5-nano",
            {"temperature": 0},
        )

    assert "temperature" not in payload


@pytest.mark.unit
def test_prepare_responses_payload_preserves_temperature_for_other_gpt5_models(client):
    payload = client._prepare_responses_payload_for_model(
        "gpt-5",
        {"temperature": 0},
    )

    assert payload["temperature"] == 0


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_closes_response_when_consumer_stops_early(mock_post, client):
    response = StreamingResponse([
        'data: {"choices":[{"delta":{"content":"Hello"}}]}',
        "",
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        "",
    ])
    mock_post.return_value = response

    events = client.stream("gpt-4o", messages=[])
    next(events)
    events.close()

    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_maps_http_errors_through_openai_error_handler(mock_post, client):
    response = StreamingResponse(
        [],
        status_code=429,
        error_payload={"error": {"type": "rate_limit_exceeded", "message": "Slow down"}},
    )
    mock_post.return_value = response

    with pytest.raises(LLMAPIRateLimitError, match="Slow down"):
        list(client.stream("gpt-4o", messages=[]))

    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_maps_generic_error_events_through_openai_error_handler(mock_post, client):
    response = StreamingResponse([
        "event: error",
        'data: {"error":{"code":"rate_limit_exceeded","message":"Slow down"}}',
        "",
    ])
    mock_post.return_value = response

    with pytest.raises(LLMAPIRateLimitError, match="Slow down"):
        list(client.stream("gpt-4o", messages=[]))

    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_maps_responses_failed_events_through_openai_error_handler(mock_post, client):
    response = StreamingResponse([
        "event: response.failed",
        'data: {"type":"response.failed","response":{"error":{"code":"server_error","message":"Unavailable"}}}',
        "",
    ])
    mock_post.return_value = response

    with pytest.raises(LLMAPIServerError, match="Unavailable"):
        list(client.stream("gpt-5", messages=[]))

    response.close.assert_called_once()
