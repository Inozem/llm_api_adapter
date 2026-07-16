import base64

import pytest
import requests_mock as req_mock

from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.models.messages.file_parts import ImagePart
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

IMAGE_URL = "https://example.com/photo.jpg"
IMAGE_BYTES = b"fakeimagebytes"
IMAGE_MEDIA_TYPE = "image/jpeg"

_OPENAI_CHAT_RESPONSE = {
    "id": "chatcmpl_123",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}

_OPENAI_RESPONSES_RESPONSE = {
    "id": "resp_123",
    "object": "response",
    "model": "gpt-5.5",
    "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}],
    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
}

_ANTHROPIC_RESPONSE = {
    "model": "claude-haiku-4-5",
    "id": "msg_123",
    "usage": {"input_tokens": 5, "output_tokens": 3},
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
}

_GOOGLE_RESPONSE = {
    "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"totalTokenCount": 8},
}


# ---------------------------------------------------------------------------
# OpenAI Chat Completions
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_openai_chat_completions_vision_url():
    with req_mock.Mocker() as m:
        m.post("https://api.openai.com/v1/chat/completions", json=_OPENAI_CHAT_RESPONSE)
        adapter = UniversalLLMAPIAdapter(organization="openai", model="gpt-4o", api_key="dummy")
        adapter.chat(messages=[UserMessage("look", files=[ImagePart(url=IMAGE_URL)])])

        body = m.last_request.json()
        image_part = body["messages"][0]["content"][1]
        assert image_part["type"] == "image_url"
        assert image_part["image_url"] == {"url": IMAGE_URL}


@pytest.mark.integration
def test_openai_chat_completions_vision_bytes_produces_data_uri():
    with req_mock.Mocker() as m:
        m.post("https://api.openai.com/v1/chat/completions", json=_OPENAI_CHAT_RESPONSE)
        adapter = UniversalLLMAPIAdapter(organization="openai", model="gpt-4o", api_key="dummy")
        adapter.chat(messages=[UserMessage("look", files=[ImagePart(data=IMAGE_BYTES, media_type=IMAGE_MEDIA_TYPE)])])

        body = m.last_request.json()
        image_url = body["messages"][0]["content"][1]["image_url"]["url"]
        assert image_url.startswith("data:image/jpeg;base64,")


@pytest.mark.integration
def test_openai_chat_completions_vision_two_images():
    with req_mock.Mocker() as m:
        m.post("https://api.openai.com/v1/chat/completions", json=_OPENAI_CHAT_RESPONSE)
        adapter = UniversalLLMAPIAdapter(organization="openai", model="gpt-4o", api_key="dummy")
        adapter.chat(messages=[UserMessage("look", files=[
            ImagePart(url="https://example.com/a.jpg"),
            ImagePart(url="https://example.com/b.png"),
        ])])

        content = m.last_request.json()["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[2]["type"] == "image_url"


# ---------------------------------------------------------------------------
# OpenAI Responses API
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_openai_responses_api_vision_url_flat_string():
    with req_mock.Mocker() as m:
        m.post("https://api.openai.com/v1/responses", json=_OPENAI_RESPONSES_RESPONSE)
        adapter = UniversalLLMAPIAdapter(organization="openai", model="gpt-5.5", api_key="dummy")
        adapter.chat(messages=[UserMessage("look", files=[ImagePart(url=IMAGE_URL)])])

        body = m.last_request.json()
        image_part = body["input"][0]["content"][1]
        assert image_part["type"] == "input_image"
        assert image_part["image_url"] == IMAGE_URL  # flat string, not nested dict


@pytest.mark.integration
def test_openai_responses_api_vision_bytes():
    with req_mock.Mocker() as m:
        m.post("https://api.openai.com/v1/responses", json=_OPENAI_RESPONSES_RESPONSE)
        adapter = UniversalLLMAPIAdapter(organization="openai", model="gpt-5.5", api_key="dummy")
        adapter.chat(messages=[UserMessage("look", files=[ImagePart(data=IMAGE_BYTES, media_type=IMAGE_MEDIA_TYPE)])])

        body = m.last_request.json()
        image_url = body["input"][0]["content"][1]["image_url"]
        assert isinstance(image_url, str)
        assert image_url.startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_anthropic_vision_url():
    with req_mock.Mocker() as m:
        m.post("https://api.anthropic.com/v1/messages", json=_ANTHROPIC_RESPONSE)
        adapter = UniversalLLMAPIAdapter(organization="anthropic", model="claude-haiku-4-5", api_key="dummy")
        adapter.chat(messages=[UserMessage("look", files=[ImagePart(url=IMAGE_URL)])], max_tokens=100)

        content = m.last_request.json()["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "look"}
        assert content[1] == {"type": "image", "source": {"type": "url", "url": IMAGE_URL}}


@pytest.mark.integration
def test_anthropic_vision_bytes():
    with req_mock.Mocker() as m:
        m.post("https://api.anthropic.com/v1/messages", json=_ANTHROPIC_RESPONSE)
        adapter = UniversalLLMAPIAdapter(organization="anthropic", model="claude-haiku-4-5", api_key="dummy")
        adapter.chat(messages=[UserMessage("look", files=[ImagePart(data=IMAGE_BYTES, media_type=IMAGE_MEDIA_TYPE)])], max_tokens=100)

        source = m.last_request.json()["messages"][0]["content"][1]["source"]
        assert source["type"] == "base64"
        assert source["media_type"] == IMAGE_MEDIA_TYPE
        assert source["data"] == base64.b64encode(IMAGE_BYTES).decode()


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_google_vision_url_camel_case():
    with req_mock.Mocker() as m:
        m.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            json=_GOOGLE_RESPONSE,
        )
        adapter = UniversalLLMAPIAdapter(organization="google", model="gemini-2.5-flash", api_key="dummy")
        adapter.chat(messages=[UserMessage("look", files=[ImagePart(url=IMAGE_URL)])])

        parts = m.last_request.json()["contents"][0]["parts"]
        assert parts[0] == {"text": "look"}
        assert parts[1] == {"fileData": {"mimeType": "image/jpeg", "fileUri": IMAGE_URL}}


@pytest.mark.integration
def test_google_vision_bytes_camel_case():
    with req_mock.Mocker() as m:
        m.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            json=_GOOGLE_RESPONSE,
        )
        adapter = UniversalLLMAPIAdapter(organization="google", model="gemini-2.5-flash", api_key="dummy")
        adapter.chat(messages=[UserMessage("look", files=[ImagePart(data=IMAGE_BYTES, media_type=IMAGE_MEDIA_TYPE)])])

        parts = m.last_request.json()["contents"][0]["parts"]
        assert parts[1] == {
            "inlineData": {
                "mimeType": IMAGE_MEDIA_TYPE,
                "data": base64.b64encode(IMAGE_BYTES).decode(),
            }
        }
