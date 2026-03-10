import pytest
import requests_mock

from src.llm_api_adapter.models.messages.chat_message import (
    Prompt,
    UserMessage,
    AIMessage,
)
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


@pytest.fixture
def openai_responses_mock():
    with requests_mock.Mocker() as mock:
        mock.post(
            "https://api.openai.com/v1/responses",
            json={
                "id": "resp_123",
                "object": "response",
                "model": "gpt-5",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Hello from mocked OpenAI!",
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 7,
                    "total_tokens": 12,
                },
            },
        )
        yield mock


@pytest.fixture
def openai_chat_completions_mock():
    with requests_mock.Mocker() as mock:
        mock.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "id": "chatcmpl_123",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello from mocked OpenAI legacy!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "total_tokens": 12,
                },
            },
        )
        yield mock


@pytest.fixture
def google_client_mock():
    with requests_mock.Mocker() as mock:
        mock_endpoint = "https://generativelanguage.googleapis.com/v1beta"
        mock.post(
            f"{mock_endpoint}/models/gemini-2.5-pro:generateContent",
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Hello from mocked Google!"}]
                        },
                        "finishReason": "stop",
                    }
                ],
                "usageMetadata": {"totalTokenCount": 42},
            },
        )
        yield mock


@pytest.fixture
def anthropic_client_mock():
    with requests_mock.Mocker() as mock:
        mock.post(
            "https://api.anthropic.com/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "id": "response123",
                "usage": {"input_tokens": 5, "output_tokens": 7},
                "content": [
                    {"type": "text", "text": "Hello from mocked Anthropic!"}
                ],
                "stop_reason": "stop",
            },
        )
        yield mock


@pytest.mark.integration
def test_openai_chat_gpt5_responses_api(openai_responses_mock):
    messages = [
        Prompt("You are an assistant."),
        UserMessage("Hi! Can you explain?"),
        AIMessage("Sure!"),
        UserMessage("How does AI learn?"),
    ]
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key="dummy_key",
    )

    response = adapter.chat(messages=messages)

    assert response.content == "Hello from mocked OpenAI!"


@pytest.mark.integration
def test_openai_chat_with_raw_dict_messages_gpt5_responses_api(
    openai_responses_mock,
):
    messages = [
        {
            "role": "system",
            "content": "You are a friendly assistant who answers only yes or no.",
        },
        {"role": "user", "content": "Do you know how AI learns?"},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "Can you explain it in one sentence?"},
    ]
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key="dummy_key",
    )

    resp = adapter.chat(messages=messages)

    assert resp.content == "Hello from mocked OpenAI!"


@pytest.mark.integration
def test_openai_chat_legacy_chat_completions(openai_chat_completions_mock):
    messages = [
        Prompt("You are an assistant."),
        UserMessage("Hi! Can you explain?"),
        AIMessage("Sure!"),
        UserMessage("How does AI learn?"),
    ]
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-4o",
        api_key="dummy_key",
    )

    response = adapter.chat(messages=messages)

    assert response.content == "Hello from mocked OpenAI legacy!"


@pytest.mark.integration
def test_google_chat(google_client_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="google",
        model="gemini-2.5-pro",
        api_key="dummy_key",
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
        {
            "role": "system",
            "content": "You are a friendly assistant who answers only yes or no.",
        },
        {"role": "user", "content": "Do you know how AI learns?"},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "Can you explain it in one sentence?"},
    ]
    adapter = UniversalLLMAPIAdapter(
        organization="google",
        model="gemini-2.5-pro",
        api_key="dummy_key",
    )

    resp = adapter.chat(messages=messages)

    assert resp.content == "Hello from mocked Google!"


@pytest.mark.integration
def test_anthropic_chat(anthropic_client_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-sonnet-4-5",
        api_key="dummy_key",
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
        {
            "role": "system",
            "content": "You are a friendly assistant who answers only yes or no.",
        },
        {"role": "user", "content": "Do you know how AI learns?"},
        {"role": "assistant", "content": "Yes."},
        {"role": "user", "content": "Can you explain it in one sentence?"},
    ]
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-sonnet-4-5",
        api_key="dummy_key",
    )

    resp = adapter.chat(messages=messages, max_tokens=2000)

    assert resp.content == "Hello from mocked Anthropic!"


@pytest.mark.integration
def test_google_usage_and_pricing(google_client_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="google",
        model="gemini-2.5-pro",
        api_key="dummy_key",
    )
    adapter.pricing.set_in_per_1m(0)
    adapter.pricing.set_out_per_1m(0)
    adapter.pricing.set_currency("EUR")

    messages = [UserMessage("Hello")]
    resp = adapter.chat(messages=messages)

    assert resp.usage.total_tokens == 42
    assert resp.cost_total == 0
    assert resp.currency == "EUR"


@pytest.mark.integration
def test_anthropic_tokens_and_pricing(anthropic_client_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-sonnet-4-5",
        api_key="dummy_key",
    )
    adapter.pricing.set_in_per_1m(1_000_000)
    adapter.pricing.set_out_per_1m(1_000_000)
    adapter.pricing.set_currency("USD")

    messages = [UserMessage("Hello")]
    resp = adapter.chat(messages=messages, max_tokens=2000)

    assert resp.usage.input_tokens == 5
    assert resp.usage.output_tokens == 7
    assert resp.usage.total_tokens == 12

    assert resp.cost_input == 5
    assert resp.cost_output == 7
    assert resp.cost_total == 12
    assert resp.currency == "USD"
