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
            "model": "claude-v1",
            "id": "response123",
            "usage": {"input_tokens": 5, "output_tokens": 7},
            "content": [{"text": "Hello from mocked Anthropic!"}],
            "stop_reason": "stop"
        })
        yield mock

def test_openai_chat(openai_client_mock):
    messages = [
        Prompt("You are an assistant."),
        UserMessage("Hi! Can you explain?"),
        AIMessage("Sure!"),
        UserMessage("How does AI learn?"),
    ]
    adapter = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-4",
        api_key="dummy_key"
    )
    response = adapter.chat(messages=messages)
    assert response.content == "Hello from mocked OpenAI!"

def test_google_chat(google_client_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="google",
        model="gemini-2.5-pro",
        api_key="dummy_key"
    )
    messages = [UserMessage("Hello")]
    resp = adapter.chat(messages=messages)
    assert resp.content == "Hello from mocked Google!"

def test_anthropic_chat(anthropic_client_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-opus-4-20250514",
        api_key="dummy_key"
    )
    messages = [UserMessage("Hello")]
    resp = adapter.chat(messages=messages)
    assert resp.content == "Hello from mocked Anthropic!"
