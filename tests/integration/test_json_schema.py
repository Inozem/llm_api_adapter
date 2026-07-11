import pytest
import requests_mock as requests_mock_module
from pydantic import BaseModel

from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


JSON_CONTENT = '{"name": "Alice", "age": 30}'

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}


class Person(BaseModel):
    name: str
    age: int


@pytest.fixture
def openai_responses_json_mock():
    with requests_mock_module.Mocker() as mock:
        mock.post(
            "https://api.openai.com/v1/responses",
            json={
                "id": "resp_json_1",
                "model": "gpt-5",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": JSON_CONTENT}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18},
            },
        )
        yield mock


@pytest.fixture
def openai_legacy_json_mock():
    with requests_mock_module.Mocker() as mock:
        mock.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "id": "chatcmpl_json_1",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": JSON_CONTENT},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
            },
        )
        yield mock


@pytest.fixture
def anthropic_json_mock():
    with requests_mock_module.Mocker() as mock:
        mock.post(
            "https://api.anthropic.com/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "id": "msg_json_1",
                "usage": {"input_tokens": 10, "output_tokens": 8},
                "content": [{"type": "text", "text": JSON_CONTENT}],
                "stop_reason": "end_turn",
            },
        )
        yield mock


@pytest.fixture
def google_json_mock():
    with requests_mock_module.Mocker() as mock:
        mock.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent",
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": JSON_CONTENT}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"totalTokenCount": 18},
            },
        )
        yield mock


# ---------------------------
# OpenAI Responses API (gpt-5)
# ---------------------------

@pytest.mark.integration
def test_openai_responses_api_sends_json_schema_in_text_param(openai_responses_json_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="openai", model="gpt-5", api_key="dummy_key"
    )

    resp = adapter.chat(messages=[UserMessage("hi")], json_schema=SCHEMA)

    payload = openai_responses_json_mock.request_history[0].json()
    assert "text" in payload
    fmt = payload["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["name"] == "response"
    assert fmt["strict"] is True
    assert fmt["schema"]["type"] == "object"
    assert fmt["schema"]["additionalProperties"] is False

    assert resp.content == JSON_CONTENT
    assert resp.parsed_json == {"name": "Alice", "age": 30}


@pytest.mark.integration
def test_openai_responses_api_sends_schema_and_returns_parsed_model(openai_responses_json_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="openai", model="gpt-5", api_key="dummy_key"
    )

    resp = adapter.chat(messages=[UserMessage("hi")], response_model=Person)

    payload = openai_responses_json_mock.request_history[0].json()
    fmt = payload["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"] == {**Person.model_json_schema(), "additionalProperties": False}

    assert resp.parsed_json == {"name": "Alice", "age": 30}
    assert isinstance(resp.parsed_model, Person)
    assert resp.parsed_model.name == "Alice"
    assert resp.parsed_model.age == 30


# ---------------------------
# OpenAI Legacy API (gpt-4o)
# ---------------------------

@pytest.mark.integration
def test_openai_legacy_api_sends_json_schema_in_response_format(openai_legacy_json_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="openai", model="gpt-4o", api_key="dummy_key"
    )

    resp = adapter.chat(messages=[UserMessage("hi")], json_schema=SCHEMA)

    payload = openai_legacy_json_mock.request_history[0].json()
    assert "response_format" in payload
    rf = payload["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "response"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["schema"]["type"] == "object"
    assert rf["json_schema"]["schema"]["additionalProperties"] is False

    assert resp.content == JSON_CONTENT
    assert resp.parsed_json == {"name": "Alice", "age": 30}


@pytest.mark.integration
def test_openai_legacy_api_sends_schema_and_returns_parsed_model(openai_legacy_json_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="openai", model="gpt-4o", api_key="dummy_key"
    )

    resp = adapter.chat(messages=[UserMessage("hi")], response_model=Person)

    payload = openai_legacy_json_mock.request_history[0].json()
    js = payload["response_format"]["json_schema"]
    assert js["schema"] == {**Person.model_json_schema(), "additionalProperties": False}

    assert resp.parsed_json == {"name": "Alice", "age": 30}
    assert isinstance(resp.parsed_model, Person)
    assert resp.parsed_model.name == "Alice"
    assert resp.parsed_model.age == 30


# ---------------------------
# Anthropic
# ---------------------------

@pytest.mark.integration
def test_anthropic_sends_output_config_for_json_schema(anthropic_json_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic", model="claude-sonnet-4-5", api_key="dummy_key"
    )

    resp = adapter.chat(
        messages=[UserMessage("hi")], max_tokens=256, json_schema=SCHEMA
    )

    payload = anthropic_json_mock.request_history[0].json()
    assert "output_config" in payload
    fmt = payload["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["type"] == "object"
    assert fmt["schema"]["additionalProperties"] is False

    assert resp.content == JSON_CONTENT
    assert resp.parsed_json == {"name": "Alice", "age": 30}


@pytest.mark.integration
def test_anthropic_sends_schema_and_returns_parsed_model(anthropic_json_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="anthropic", model="claude-sonnet-4-5", api_key="dummy_key"
    )

    resp = adapter.chat(messages=[UserMessage("hi")], max_tokens=256, response_model=Person)

    payload = anthropic_json_mock.request_history[0].json()
    fmt = payload["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"] == {**Person.model_json_schema(), "additionalProperties": False}

    assert resp.parsed_json == {"name": "Alice", "age": 30}
    assert isinstance(resp.parsed_model, Person)
    assert resp.parsed_model.name == "Alice"
    assert resp.parsed_model.age == 30


# ---------------------------
# Google
# ---------------------------

@pytest.mark.integration
def test_google_sends_response_mime_type_and_schema(google_json_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="google", model="gemini-2.5-pro", api_key="dummy_key"
    )

    resp = adapter.chat(messages=[UserMessage("hi")], json_schema=SCHEMA)

    payload = google_json_mock.request_history[0].json()
    gen_cfg = payload["generationConfig"]
    assert gen_cfg["responseMimeType"] == "application/json"
    assert "responseSchema" in gen_cfg
    assert gen_cfg["responseSchema"]["type"] == "OBJECT"
    assert "name" in gen_cfg["responseSchema"]["properties"]
    assert "age" in gen_cfg["responseSchema"]["properties"]
    assert "additionalProperties" not in gen_cfg["responseSchema"]

    assert resp.content == JSON_CONTENT
    assert resp.parsed_json == {"name": "Alice", "age": 30}


@pytest.mark.integration
def test_google_sends_schema_and_returns_parsed_model_for_response_model(google_json_mock):
    adapter = UniversalLLMAPIAdapter(
        organization="google", model="gemini-2.5-pro", api_key="dummy_key"
    )

    resp = adapter.chat(messages=[UserMessage("hi")], response_model=Person)

    payload = google_json_mock.request_history[0].json()
    gen_cfg = payload["generationConfig"]
    assert gen_cfg["responseMimeType"] == "application/json"
    assert "name" in gen_cfg["responseSchema"]["properties"]
    assert "age" in gen_cfg["responseSchema"]["properties"]

    assert resp.parsed_json == {"name": "Alice", "age": 30}
    assert isinstance(resp.parsed_model, Person)
    assert resp.parsed_model.name == "Alice"
    assert resp.parsed_model.age == 30
