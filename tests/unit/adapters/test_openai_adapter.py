from unittest.mock import patch

import pytest

from src.llm_api_adapter.adapters.openai_adapter import OpenAIAdapter
from src.llm_api_adapter.errors.llm_api_error import LLMAPIError
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.models.responses.chat_response import ChatResponse

@pytest.fixture
def adapter():
    return OpenAIAdapter(
        api_key="test_api_key",
        model="gpt-5"
    )

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
        OpenAISyncClient, method, side_effect=LLMAPIError("API error")
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
        OpenAISyncClient, method, side_effect=Exception("Generic error")
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.generate_chat_answer(messages)
        mock_handle_error.assert_called_once()

def test_pricing_is_applied_when_present(adapter):
    adapter.pricing = type("P", (), {
        "in_per_token": 0.001, "out_per_token": 0.002, "currency": "USD"
    })()
    fake_response = {"some": "openai response"}
    fake_chat_response = ChatResponse()
    patch_chat_completion = patch.object(
        OpenAISyncClient, "chat_completion", return_value=fake_response
    )
    patch_from_openai_response = patch.object(
        ChatResponse, "from_openai_response", return_value=fake_chat_response
    )
    patch_apply_pricing = patch.object(ChatResponse, "apply_pricing")
    with (
        patch_chat_completion as mock_client,
        patch_from_openai_response as mock_from,
        patch_apply_pricing as mock_apply
    ):
        result = adapter.generate_chat_answer([
            type("Message", (), {"content": "hi", "role": "user"})()
        ], max_tokens=10)
    mock_client.assert_called_once()
    mock_from.assert_called_once_with(fake_response)
    mock_apply.assert_called_once_with(
        price_input_per_token=adapter.pricing.in_per_token,
        price_output_per_token=adapter.pricing.out_per_token,
        currency=adapter.pricing.currency,
    )
    assert result is fake_chat_response
