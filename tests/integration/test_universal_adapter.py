import pytest
import requests_mock

from src.llm_api_adapter.models.messages.chat_message import (
    Prompt, UserMessage, AIMessage
)
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

@pytest.fixture
def openai_client_mock():
    with requests_mock.Mocker() as mock:
        mock.post("https://api.openai.com/v1/chat/completions", json={
            "choices": [{"message": {"content": "Hello from mocked OpenAI!"}}]
        })
        yield mock

@pytest.fixture
def google_client_mock():
    with requests_mock.Mocker() as mock:
        mock_endpoint = "https://generativelanguage.googleapis.com/v1beta"
        mock.post(f"{mock_endpoint}/models/gemini-2.5-pro:generateContent",
            json={
                "candidates": [{
                    "content": {
                        "parts": [{"text": "Hello from mocked Google!"}]
                    },
                    "finishReason": "stop"
                }],
                "usageMetadata": {"totalTokenCount": 42}
            })
        yield mock

@pytest.fixture
def anthropic_client_mock():
    with requests_mock.Mocker() as mock:
        mock.post("https://api.anthropic.com/v1/messages", json={
            "model": "claude-sonnet-4-5",
            "id": "response123",
            "usage": {"input_tokens": 5, "output_tokens": 7},
            "content": [{"type": "text", "text": "Hello from mocked Anthropic!"}],
            "stop_reason": "stop"
        })
        yield mock

@pytest.mark.integration
def test_openai_chat(openai_client_mock):
    messages = [
        Prompt("You are an assistant."),
        UserMessage("Hi! Can you explain?"),
        AIMessage("Sure!"),
        UserMessage("How does AI learn?"),
    ]
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key="dummy_key"
    )
    response = adapter.chat(messages=messages)
    assert response.content == "Hello from mocked OpenAI!"

@pytest.mark.integration
def test_openai_chat_with_raw_dict_messages(openai_client_mock):
    messages = [
        {"role": "system", "content": "You are a friendly assistant who answers only yes or no."},
        {"role": "user", "content": "Do you know how AI learns?"},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "Can you explain it in one sentence?"}
    ]
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key="dummy_key"
    )
    resp = adapter.chat(messages=messages)
    assert resp.content == "Hello from mocked OpenAI!"

@pytest.mark.integration
def test_google_chat(google_client_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="google",
        model="gemini-2.5-pro",
        api_key="dummy_key"
    )
    messages = [
        Prompt("You are an assistant."),
        UserMessage("Hi! Can you explain?"),
        AIMessage("Sure!"),
        UserMessage("How does AI learn?"),
    ]
    resp = adapter.chat(messages=messages)
    assert resp.content == "Hello from mocked Google!"

@pytest.mark.integration
def test_google_chat_with_raw_dict_messages(google_client_mock):
    messages = [
        {"role": "system", "content": "You are a friendly assistant who answers only yes or no."},
        {"role": "user", "content": "Do you know how AI learns?"},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "Can you explain it in one sentence?"}
    ]
    adapter = UniversalLLMAPIAdapter(
        organization="google",
        model="gemini-2.5-pro",
        api_key="dummy_key"
    )
    resp = adapter.chat(messages=messages)
    assert resp.content == "Hello from mocked Google!"

@pytest.mark.integration
def test_anthropic_chat(anthropic_client_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-sonnet-4-5",
        api_key="dummy_key"
    )
    messages = [
        Prompt("You are an assistant."),
        UserMessage("Hi! Can you explain?"),
        AIMessage("Sure!"),
        UserMessage("How does AI learn?"),
    ]
    resp = adapter.chat(messages=messages, max_tokens=2000)
    assert resp.content == "Hello from mocked Anthropic!"

@pytest.mark.integration
def test_anthropic_chat_with_raw_dict_messages(anthropic_client_mock):
    messages = [
        {"role": "system", "content": "You are a friendly assistant who answers only yes or no."},
        {"role": "user", "content": "Do you know how AI learns?"},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "Can you explain it in one sentence?"}
    ]
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-sonnet-4-5",
        api_key="dummy_key"
    )
    resp = adapter.chat(messages=messages, max_tokens=2000)
    assert resp.content == "Hello from mocked Anthropic!"

@pytest.mark.integration
def test_google_usage_and_pricing(google_client_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="google",
        model="gemini-2.5-pro",
        api_key="dummy_key"
    )
    # Set pricing to zero so cost_total is deterministic (0) while usage.total_tokens should map from response
    adapter.pricing.set_in_per_1m(0)
    adapter.pricing.set_out_per_1m(0)
    adapter.pricing.set_currency("EUR")

    messages = [UserMessage("Hello")]
    resp = adapter.chat(messages=messages)

    # The mocked Google response provides totalTokenCount = 42
    assert resp.usage.total_tokens == 42
    assert resp.cost_total == 0
    assert resp.currency == "EUR"

@pytest.mark.integration
def test_anthropic_tokens_and_pricing(anthropic_client_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-sonnet-4-5",
        api_key="dummy_key"
    )
    # Set pricing so cost per token = 1 unit (per_1m = 1_000_000 -> 1_000_000/1_000_000 = 1)
    adapter.pricing.set_in_per_1m(1_000_000)
    adapter.pricing.set_out_per_1m(1_000_000)
    adapter.pricing.set_currency("USD")

    messages = [UserMessage("Hello")]
    resp = adapter.chat(messages=messages, max_tokens="2000")

    # The mocked Anthropic response provides input_tokens=5 and output_tokens=7
    assert resp.usage.input_tokens == 5
    assert resp.usage.output_tokens == 7
    assert resp.usage.total_tokens == 12

    # With per-token cost of 1 unit, costs should equal the token counts
    assert resp.cost_input == 5
    assert resp.cost_output == 7
    assert resp.cost_total == 12
    assert resp.currency == "USD"
