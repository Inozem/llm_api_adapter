from unittest.mock import patch

import pytest

from src.llm_api_adapter.adapters.base_adapter import LLMAdapterBase
from src.llm_api_adapter.errors.llm_api_error import LLMAPIError
from src.llm_api_adapter.models.responses.chat_response import ChatResponse


class DummyClient:
    def chat_completion(self, messages, **kwargs):
        return {"choices": [{"message": {"content": "dummy response"}}]}

class _TestAdapter(LLMAdapterBase):
    def generate_chat_answer(self, messages, **kwargs):
        try:
            raw_response = self.client.chat_completion(messages, **kwargs)
            return ChatResponse(**raw_response)
        except (LLMAPIError, Exception) as e:
            return self.handle_error(e)

@pytest.fixture
def adapter():
    adapter_instance = _TestAdapter(
        company="openai",
        api_key="dummy_key",
        model="gpt-5",
    )
    adapter_instance.client = DummyClient()
    return adapter_instance

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
    with patch.object(
        DummyClient, "chat_completion", side_effect=LLMAPIError("API error")
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.generate_chat_answer(messages)
        mock_handle_error.assert_called_once()

def test_chat_handles_generic_exception(adapter):
    messages = [
        type("Prompt", (), {"content": "system prompt", "role": "system"})(),
        type("Message", (), {"content": "hello", "role": "user"})(),
    ]
    with patch.object(
        DummyClient, "chat_completion", side_effect=Exception("Generic error")
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.generate_chat_answer(messages)
        mock_handle_error.assert_called_once()
