import base64

import pytest
import requests_mock as req_mock

from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.models.messages.file_parts import DocumentPart
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

PDF_URL = "https://example.com/report.pdf"
PDF_BYTES = b"%PDF-1.4 test document"
PDF_MEDIA_TYPE = "application/pdf"

_OPENAI_CHAT_RESPONSE = {
    "id": "chatcmpl_123",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "ok"},
        "finish_reason": "stop",
    }],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}

_OPENAI_RESPONSES_RESPONSE = {
    "id": "resp_123",
    "object": "response",
    "model": "gpt-5.5",
    "output": [{
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "ok"}],
    }],
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
# Anthropic
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_anthropic_document_url():
    with req_mock.Mocker() as m:
        m.post("https://api.anthropic.com/v1/messages", json=_ANTHROPIC_RESPONSE)
        adapter = UniversalLLMAPIAdapter(
            organization="anthropic",
            model="claude-haiku-4-5",
            api_key="dummy",
        )
        adapter.chat(
            messages=[UserMessage("summarize", files=[DocumentPart(url=PDF_URL)])],
            max_tokens=100,
        )

        document = m.last_request.json()["messages"][0]["content"][1]
        assert document == {
            "type": "document",
            "source": {"type": "url", "url": PDF_URL},
        }


@pytest.mark.integration
def test_anthropic_document_bytes():
    with req_mock.Mocker() as m:
        m.post("https://api.anthropic.com/v1/messages", json=_ANTHROPIC_RESPONSE)
        adapter = UniversalLLMAPIAdapter(
            organization="anthropic",
            model="claude-haiku-4-5",
            api_key="dummy",
        )
        adapter.chat(
            messages=[
                UserMessage(
                    "summarize",
                    files=[DocumentPart(data=PDF_BYTES, media_type=PDF_MEDIA_TYPE)],
                )
            ],
            max_tokens=100,
        )

        source = m.last_request.json()["messages"][0]["content"][1]["source"]
        assert source["type"] == "base64"
        assert source["media_type"] == PDF_MEDIA_TYPE
        assert source["data"] == base64.b64encode(PDF_BYTES).decode()


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_google_document_url():
    with req_mock.Mocker() as m:
        m.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            json=_GOOGLE_RESPONSE,
        )
        adapter = UniversalLLMAPIAdapter(
            organization="google",
            model="gemini-2.5-flash",
            api_key="dummy",
        )
        adapter.chat(
            messages=[UserMessage("summarize", files=[DocumentPart(url=PDF_URL)])],
        )

        document = m.last_request.json()["contents"][0]["parts"][1]
        assert document == {
            "fileData": {"mimeType": PDF_MEDIA_TYPE, "fileUri": PDF_URL},
        }


@pytest.mark.integration
def test_google_document_bytes():
    with req_mock.Mocker() as m:
        m.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            json=_GOOGLE_RESPONSE,
        )
        adapter = UniversalLLMAPIAdapter(
            organization="google",
            model="gemini-2.5-flash",
            api_key="dummy",
        )
        adapter.chat(
            messages=[
                UserMessage(
                    "summarize",
                    files=[DocumentPart(data=PDF_BYTES, media_type=PDF_MEDIA_TYPE)],
                )
            ],
        )

        document = m.last_request.json()["contents"][0]["parts"][1]
        assert document == {
            "inlineData": {
                "mimeType": PDF_MEDIA_TYPE,
                "data": base64.b64encode(PDF_BYTES).decode(),
            },
        }


# ---------------------------------------------------------------------------
# OpenAI Responses API
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_openai_responses_document_url():
    with req_mock.Mocker() as m:
        m.post("https://api.openai.com/v1/responses", json=_OPENAI_RESPONSES_RESPONSE)
        adapter = UniversalLLMAPIAdapter(
            organization="openai",
            model="gpt-5.5",
            api_key="dummy",
        )
        adapter.chat(
            messages=[UserMessage("summarize", files=[DocumentPart(url=PDF_URL)])],
        )

        document = m.last_request.json()["input"][0]["content"][1]
        assert document == {"type": "input_file", "file_url": PDF_URL}


@pytest.mark.integration
def test_openai_responses_document_bytes():
    with req_mock.Mocker() as m:
        m.post("https://api.openai.com/v1/responses", json=_OPENAI_RESPONSES_RESPONSE)
        adapter = UniversalLLMAPIAdapter(
            organization="openai",
            model="gpt-5.5",
            api_key="dummy",
        )
        adapter.chat(
            messages=[
                UserMessage(
                    "summarize",
                    files=[DocumentPart(data=PDF_BYTES, media_type=PDF_MEDIA_TYPE)],
                )
            ],
        )

        document = m.last_request.json()["input"][0]["content"][1]
        assert document == {
            "type": "input_file",
            "filename": "document.pdf",
            "file_data": f"data:{PDF_MEDIA_TYPE};base64,{base64.b64encode(PDF_BYTES).decode()}",
        }


# ---------------------------------------------------------------------------
# OpenAI Chat Completions
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_openai_chat_document_bytes():
    with req_mock.Mocker() as m:
        m.post("https://api.openai.com/v1/chat/completions", json=_OPENAI_CHAT_RESPONSE)
        adapter = UniversalLLMAPIAdapter(
            organization="openai",
            model="gpt-4o",
            api_key="dummy",
        )
        adapter.chat(
            messages=[
                UserMessage(
                    "summarize",
                    files=[DocumentPart(data=PDF_BYTES, media_type=PDF_MEDIA_TYPE)],
                )
            ],
        )

        document = m.last_request.json()["messages"][0]["content"][1]
        assert document == {
            "type": "file",
            "file": {
                "filename": "document.pdf",
                "file_data": f"data:{PDF_MEDIA_TYPE};base64,{base64.b64encode(PDF_BYTES).decode()}",
            },
        }


@pytest.mark.integration
def test_openai_chat_document_url_raises_before_request():
    with req_mock.Mocker() as m:
        m.post("https://api.openai.com/v1/chat/completions", json=_OPENAI_CHAT_RESPONSE)
        adapter = UniversalLLMAPIAdapter(
            organization="openai",
            model="gpt-4o",
            api_key="dummy",
        )

        with pytest.raises(ValueError, match="does not support DocumentPart URL"):
            adapter.chat(
                messages=[UserMessage("summarize", files=[DocumentPart(url=PDF_URL)])],
            )
        assert not m.called
