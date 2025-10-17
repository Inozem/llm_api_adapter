from unittest.mock import patch

import pytest

from src.llm_api_adapter.adapters.anthropic_adapter import AnthropicAdapter
from src.llm_api_adapter.errors.llm_api_error import LLMAPIError
from src.llm_api_adapter.adapters.anthropic_adapter import ClaudeSyncClient
from src.llm_api_adapter.models.responses.chat_response import ChatResponse

@pytest.fixture
def adapter():
    return AnthropicAdapter(
        api_key="test_api_key",
        model="claude-sonnet-4-5"
    )

@pytest.mark.parametrize("temperature,width,valid", [
    (1.0, 256, True),
    (-0.1, 256, False),
    (2.1, 256, False),
    (0.0, 256, True),
])
def test_temperature_validation(adapter, temperature, width, valid):
    if valid:
        result = adapter._validate_parameter("temperature", temperature, 0, 2)
        assert result == temperature
    else:
        with pytest.raises(ValueError):
            adapter._validate_parameter("temperature", temperature, 0, 2)

def test_chat_handles_llmapi_error(adapter):
    messages = [
        type("Prompt", (), {"content": "system prompt", "role": "system"})(),
        type("Message", (), {"content": "hello", "role": "user"})(),
    ]
    method = "chat_completion"
    with patch.object(
        ClaudeSyncClient, method, side_effect=LLMAPIError("API error")
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
        ClaudeSyncClient, method, side_effect=Exception("Generic error")
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.generate_chat_answer(messages)
        mock_handle_error.assert_called_once()

def test_pricing_is_applied_when_present(adapter):
    adapter.pricing = type("P", (), {
        "in_per_token": 0.001, "out_per_token": 0.002, "currency": "USD"
    })()
    fake_response = {"some": "anthropic response"}
    fake_chat_response = ChatResponse(content="fake")
    patch_chat_completion = patch.object(
        ClaudeSyncClient, "chat_completion", return_value=fake_response
    )
    patch_from_anthropic = patch.object(
        ChatResponse, "from_anthropic_response", return_value=fake_chat_response
    )
    patch_apply_pricing = patch.object(ChatResponse, "apply_pricing")
    with (
        patch_chat_completion,
        patch_from_anthropic as mock_from,
        patch_apply_pricing as mock_apply
    ):
        adapter.generate_chat_answer([type("Message", (), {
            "content": "hi", "role": "user"
        })()])
        mock_apply.assert_called_once_with(
            price_input_per_token=adapter.pricing.in_per_token,
            price_output_per_token=adapter.pricing.out_per_token,
            currency=adapter.pricing.currency,
        )
