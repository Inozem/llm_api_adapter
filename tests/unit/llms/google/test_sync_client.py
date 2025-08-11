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
    headers = mock_post.call_args[1]["headers"]
    assert headers["x-goog-api-key"] == "test_api_key"

@pytest.mark.parametrize("exception,expected_exception", [
    (requests.exceptions.Timeout("timeout"), LLMAPITimeoutError),
    (requests.exceptions.RequestException("generic error"), LLMAPIClientError),
])
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
    (400, "INVALID_ARGUMENT", LLMAPIAuthorizationError),
    (500, "INTERNAL", LLMAPIServerError),
])
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

@patch("src.llm_api_adapter.llms.google.sync_client.requests.post")
def test_send_request_fallback_error_parsing(mock_post, client):
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.json.side_effect = ValueError("Invalid JSON")
    http_err = requests.exceptions.HTTPError(response=mock_response)
    mock_post.side_effect = http_err
    with pytest.raises(LLMAPIClientError):
        client._send_request("http://example.com", {})
