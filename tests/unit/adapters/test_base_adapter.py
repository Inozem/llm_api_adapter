from unittest.mock import patch
from types import SimpleNamespace

import pytest

from src.llm_api_adapter.adapters.base_adapter import LLMAdapterBase
from src.llm_api_adapter.adapters import base_adapter as base_module
from src.llm_api_adapter.errors.llm_api_error import LLMAPIError
from src.llm_api_adapter.models.messages.chat_message import (
    Messages, Prompt, UserMessage
)
from src.llm_api_adapter.models.responses.chat_response import ChatResponse


class DummyClient:
    def chat_completion(self, messages, **kwargs):
        return {"choices": [{"message": {"content": "dummy response"}}]}


class _TestAdapter(LLMAdapterBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = DummyClient()

    def chat(self, messages, **kwargs):
        try:
            raw_response = self.client.chat_completion(messages, **kwargs)
            try:
                return ChatResponse(**raw_response)
            except Exception:
                return raw_response
        except LLMAPIError as e:
            return self.handle_error(e)
        except Exception as e:
            return self.handle_error(e)
    
    def _normalize_reasoning_level(self, reasoning_level):
        return reasoning_level


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
@pytest.mark.unit
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

@pytest.mark.unit
def test_chat_handles_llmapi_error(adapter):
    messages = [Prompt("system prompt"), UserMessage("hello")]
    with patch.object(
        DummyClient, "chat_completion", side_effect=LLMAPIError("API error")
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.chat(messages)
        mock_handle_error.assert_called_once()

@pytest.mark.unit
def test_chat_handles_generic_exception(adapter):
    messages = [Prompt("system prompt"), UserMessage("hello")]
    with patch.object(
        DummyClient, "chat_completion", side_effect=Exception("Generic error")
    ), patch.object(adapter, "handle_error") as mock_handle_error:
        adapter.chat(messages)
        mock_handle_error.assert_called_once()

@pytest.mark.unit
def test_init_with_empty_api_key_raises():
    with pytest.raises(ValueError):
        _TestAdapter(company="any", api_key="", model="m1")

@pytest.mark.unit
def test_unverified_model_warns_and_leaves_pricing_none(monkeypatch):
    monkeypatch.setattr(
        base_module,
        "LLM_REGISTRY",
        SimpleNamespace(providers={}),
        raising=False
    )
    with pytest.warns(UserWarning):
        a = _TestAdapter(company="missing", api_key="k", model="unknown-model")
    assert a.pricing is None

@pytest.mark.unit
def test_pricing_copied_from_registry(monkeypatch):
    base_pricing = {"cost_per_token": 0.001}
    provider = SimpleNamespace(
        models={"m-pro": SimpleNamespace(pricing=base_pricing)}
    )
    monkeypatch.setattr(
        base_module,
        "LLM_REGISTRY",
        SimpleNamespace(providers={"acme": provider}),
        raising=False
    )
    adapter_instance = _TestAdapter(company="acme", api_key="k", model="m-pro")
    assert adapter_instance.pricing == base_pricing
    assert adapter_instance.pricing is not base_pricing

@pytest.mark.unit
def test_normalize_messages_accepts_list_and_messages(adapter):
    raw_messages = [UserMessage("hi")]
    normalized = adapter._normalize_messages(raw_messages)
    assert isinstance(normalized, Messages)
    same = adapter._normalize_messages(normalized)
    assert same is normalized
    with pytest.raises(TypeError):
        adapter._normalize_messages(123)

@pytest.mark.unit
def test_generate_chat_answer_emits_deprecation_warning(adapter):
    with pytest.warns(DeprecationWarning):
        adapter.generate_chat_answer(messages=[UserMessage("hi")])

@pytest.mark.unit
def test_handle_error_reraises_and_logs(adapter, caplog):
    err = Exception("boom")
    with pytest.raises(Exception) as excinfo:
        adapter.handle_error(err, "some message")
    assert isinstance(excinfo.value, Exception)
    assert any(
        sub in str(excinfo.value)
        for sub in ("boom","No active exception to reraise")
    )
    assert adapter.company in caplog.text or adapter.model in caplog.text
