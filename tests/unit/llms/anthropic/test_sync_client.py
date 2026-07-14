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

@pytest.mark.unit
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
def test_prepare_payload_is_adaptive_thinking_never_in_result(client):
    for flag in (True, False):
        kwargs = {"messages": [], "is_adaptive_thinking": flag}
        payload = client._prepare_chat_payload_for_model("claude-opus-4-8", kwargs)
        assert "is_adaptive_thinking" not in payload
