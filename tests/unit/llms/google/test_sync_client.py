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
from src.llm_api_adapter.llms.google.sync_client import GeminiSyncClient
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
    return GeminiSyncClient(api_key="test_api_key")

@pytest.fixture
def mock_post_success():
    mock_response = Mock()
    mock_response.json.return_value = {
        "candidates": [{"content": "Hello"}]
    }
    mock_response.raise_for_status = Mock()
    with patch(
        "src.llm_api_adapter.llms.google.sync_client.requests.post",
        return_value=mock_response
    ) as mock_post:
        yield mock_post, mock_response

@pytest.mark.unit
def test_chat_completion_success(client, mock_post_success):
    mock_post, _ = mock_post_success
    result = client.chat_completion(
        "gemini-1", prompt={"messages":[{"author":"user","content":"Hi"}]}
    )
    assert isinstance(result, dict)
    assert "candidates" in result
    candidate_content = result["candidates"][0]["content"]
    assert candidate_content == "Hello"
    mock_post.assert_called_once()
    assert mock_post.call_args.args[0] == (
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1:generateContent"
    )
    headers = mock_post.call_args[1]["headers"]
    assert headers["x-goog-api-key"] == "test_api_key"

@pytest.mark.parametrize("exception,expected_exception", [
    (requests.exceptions.Timeout("timeout"), LLMAPITimeoutError),
    (requests.exceptions.RequestException("generic error"), LLMAPIClientError),
])
@pytest.mark.unit
@patch("src.llm_api_adapter.llms.google.sync_client.requests.post")
def test_send_request_exceptions(
    mock_post, client, exception, expected_exception
):
    mock_post.side_effect = exception
    with pytest.raises(expected_exception):
        client._send_request("http://example.com", {})

@pytest.mark.parametrize("status_code,error_status,expected_exception", [
    (401, "UNAUTHENTICATED", LLMAPIAuthorizationError),
    (429, "RESOURCE_EXHAUSTED", LLMAPIRateLimitError),
    (400, "PERMISSION_DENIED", LLMAPIAuthorizationError),
    (500, "INTERNAL", LLMAPIServerError),
])
@pytest.mark.unit
@patch("src.llm_api_adapter.llms.google.sync_client.requests.post")
def test_send_request_http_errors(
    mock_post, client, status_code, error_status, expected_exception
):
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.json.return_value = {
        "error": {"status": error_status, "message": "Error message"}
    }
    http_err = requests.exceptions.HTTPError(response=mock_response)
    mock_post.side_effect = http_err
    with pytest.raises(expected_exception):
        client._send_request("http://example.com", {})

@pytest.mark.unit
@patch("src.llm_api_adapter.llms.google.sync_client.requests.post")
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
def test_stream_uses_stream_generate_content_and_yields_raw_chunks(mock_post, client):
    response = StreamingResponse([
        'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}',
        "",
        'data: {"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"city":"Tel Aviv"}}}]},"finishReason":"STOP"}],"usageMetadata":{"totalTokenCount":12}}',
        "",
    ])
    mock_post.return_value = response

    events = list(client.stream(
        "gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": "Hi"}]}],
        generationConfig={"maxOutputTokens": 64},
    ))

    assert events == [
        SSEEvent(
            event=None,
            data={"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},
        ),
        SSEEvent(
            event=None,
            data={
                "candidates": [{
                    "content": {"parts": [{"functionCall": {
                        "name": "get_weather", "args": {"city": "Tel Aviv"},
                    }}]},
                    "finishReason": "STOP",
                }],
                "usageMetadata": {"totalTokenCount": 12},
            },
        ),
    ]
    assert mock_post.call_args.args[0] == (
        "https://generativelanguage.googleapis.com/v1beta/"
        "models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    )
    assert mock_post.call_args.kwargs["json"] == {
        "model": "gemini-2.5-flash",
        "contents": [{"role": "user", "parts": [{"text": "Hi"}]}],
        "generationConfig": {},
    }
    assert mock_post.call_args.kwargs["stream"] is True
    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_preserves_adapter_resolved_thinking_payload(mock_post, client):
    response = StreamingResponse([])
    mock_post.return_value = response

    assert list(client.stream(
        "gemini-2.5-flash-lite",
        contents=[],
        generationConfig={"thinkingConfig": {"thinkingBudget": 1}},
    )) == []

    assert mock_post.call_args.kwargs["json"] == {
        "model": "gemini-2.5-flash-lite",
        "contents": [],
        "generationConfig": {"thinkingConfig": {"thinkingBudget": 1}},
    }


@pytest.mark.unit
def test_prepare_payload_allows_thought_summaries_without_budget(client):
    payload = client._prepare_chat_payload_for_model(
        "gemini-2.5-pro",
        {"generationConfig": {"thinkingConfig": {"includeThoughts": True}}},
    )

    assert payload["generationConfig"] == {
        "thinkingConfig": {"includeThoughts": True}
    }


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_closes_response_when_consumer_stops_early(mock_post, client):
    response = StreamingResponse([
        'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}]}}]}',
        "",
        'data: {"candidates":[{"content":{"parts":[{"text":" world"}]}}]}',
        "",
    ])
    mock_post.return_value = response

    events = client.stream("gemini-2.5-flash", contents=[])
    next(events)
    events.close()

    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_maps_http_errors_through_google_error_handler(mock_post, client):
    response = StreamingResponse(
        [],
        status_code=429,
        error_payload={"error": {"status": "RESOURCE_EXHAUSTED", "message": "Slow down"}},
    )
    mock_post.return_value = response

    with pytest.raises(LLMAPIRateLimitError, match="Slow down"):
        list(client.stream("gemini-2.5-flash", contents=[]))

    response.close.assert_called_once()


@pytest.mark.unit
@patch("src.llm_api_adapter.llms.streaming.requests.post")
def test_stream_maps_error_chunks_through_google_error_handler(mock_post, client):
    response = StreamingResponse([
        'data: {"error":{"status":"UNAVAILABLE","message":"Temporarily unavailable"}}',
        "",
    ])
    mock_post.return_value = response

    with pytest.raises(LLMAPIServerError, match="Temporarily unavailable"):
        list(client.stream("gemini-2.5-flash", contents=[]))

    response.close.assert_called_once()
