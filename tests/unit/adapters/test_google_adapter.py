from unittest.mock import patch

import pytest

from src.llm_api_adapter.adapters.google_adapter import GoogleAdapter
from src.llm_api_adapter.errors.llm_api_error import LLMAPIError
from src.llm_api_adapter.llms.google.sync_client import GeminiSyncClient
from src.llm_api_adapter.models.responses.chat_response import ChatResponse

@pytest.fixture
def adapter():
    return GoogleAdapter(
        api_key="test_api_key",
        model="gemini-2.5-pro"
    )

@pytest.fixture
def mock_chat_completion_success():
    mock_response = {
        "choices": [{"message": {"content": "This is a test completion."}}]
    }
    method = "chat_completion"
    with patch.object(
        GeminiSyncClient, method, return_value=mock_response
    ) as mock_chat_completion:
        yield mock_chat_completion, mock_response

def test_chat_success(adapter, mock_chat_completion_success):
    mock_chat, mock_response = mock_chat_completion_success
    messages = []
    if adapter.verified_models:
        adapter.model = next(iter(adapter.verified_models))
    else:
        adapter.model = "gemini-2.5-pro"
    messages = [
        type('Prompt', (), {'content': 'system prompt', 'role': 'system'})(),
        type('Message', (), {'content': 'hello', 'role': 'user'})()
    ]
    method = "from_google_response"
    with patch.object(
        ChatResponse, method, return_value=ChatResponse()
    ) as mock_from_response:
        response = adapter.generate_chat_answer(messages)
        mock_chat.assert_called_once()
        mock_from_response.assert_called_once_with(mock_response)
        assert isinstance(response, ChatResponse)

@pytest.mark.parametrize("temperature,max_tokens,top_p,valid", [
    (1.0, 256, 1.0, True),
    (-0.1, 256, 1.0, False),
    (2.1, 256, 1.0, False),
    (0.0, 256, 1.0, True),
    (1.0, 256, -0.1, False),
    (1.0, 256, 1.1, False),
])
def test_parameter_validation(adapter, temperature, max_tokens, top_p, valid):
    if valid:
        temp_result = adapter._validate_parameter(
            "temperature", temperature, 0, 2
        )
        top_p_result = adapter._validate_parameter("top_p", top_p, 0, 1)
        assert temp_result == temperature
        assert top_p_result == top_p
    else:
        if temperature < 0 or temperature > 2:
            with pytest.raises(ValueError):
                adapter._validate_parameter("temperature", temperature, 0, 2)
        if top_p < 0 or top_p > 1:
            with pytest.raises(ValueError):
                adapter._validate_parameter("top_p", top_p, 0, 1)

def test_chat_handles_llmapi_error(adapter):
    messages = [
        type("Prompt", (), {"content": "system prompt", "role": "system"})(),
        type("Message", (), {"content": "hello", "role": "user"})(),
    ]
    method = "chat_completion"
    with patch.object(
        GeminiSyncClient, method, side_effect=LLMAPIError("API error")
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.generate_chat_answer(messages)
        mock_handle_error.assert_called_once()

def test_chat_handles_generic_exception(adapter):
    messages = [
        type("Prompt", (), {"content": "system prompt", "role": "system"})(),
        type("Message", (), {"content": "hello", "role": "user"})(),
    ]
    method = "chat_completion"
    with patch.object(
        GeminiSyncClient, method, side_effect=Exception("Generic error")
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.generate_chat_answer(messages)
        mock_handle_error.assert_called_once()
