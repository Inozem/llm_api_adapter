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

def test_chat_completion_success(client, mock_post_success):
    mock_post, _ = mock_post_success
    result = client.chat_completion(
        "claude-v1", messages=[{"role": "user", "content": "Hi"}]
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

@patch("src.llm_api_adapter.llms.anthropic.sync_client.requests.post")
def test_send_request_fallback_error_parsing(mock_post, client):
    mock_response = Mock()
    mock_response.status_code = 400
    mock_response.json.side_effect = ValueError("Invalid JSON")
    http_err = requests.exceptions.HTTPError(response=mock_response)
    mock_post.side_effect = http_err
    with pytest.raises(LLMAPIClientError):
        client._send_request("http://example.com", {})
