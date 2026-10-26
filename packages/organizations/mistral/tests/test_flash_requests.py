"""Deterministic request and preflight tests for hosted GLM Flash on Mistral."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (str(PACKAGE_ROOT / "src"), str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
import llm_api_adapter_mistral.clients.async_client as mistral_async_client_module
from llm_api_adapter.errors.llm_api_error import (
    JSONSchemaError,
    ToolChoiceError,
)
from llm_api_adapter.errors import LLMReasoningLevelError
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.httpx_transport import HttpxSyncTransport
from llm_api_adapter.llms.requests_transport import RequestsSyncTransport
from llm_api_adapter.llms.transports import JSONResponse
from llm_api_adapter.models.messages.chat_message import ToolMessage, UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart, ImagePart
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_mistral.plugin import PLUGIN
from tests.fixtures.structured_output import (
    INLINE_NESTED_OBJECT_SCHEMA,
    NestedPydanticResponse,
)


MODEL = "glm-5.3-flash"
WIRE_MODEL = "zai-glm-5-3-flash"
CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
OCR_URL = "https://api.mistral.ai/v1/ocr"


class RecordingSyncTransport:
    """A URL-aware transport that records OCR and Chat independently."""

    def __init__(self, *, response=None, stream_events=()):
        self.response = response or {
            "id": "flash-chat",
            "model": WIRE_MODEL,
            "choices": [{"message": {"role": "assistant", "content": "Done."}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
        self.stream_events = list(stream_events)
        self.ocr_requests = []
        self.chat_requests = []
        self.stream_requests = []

    def post_json(self, request, *, http_error_handler=None):
        if request.url == OCR_URL:
            self.ocr_requests.append(request)
            return JSONResponse({
                "model": "mistral-ocr-4-1",
                "pages": [{"markdown": "# OCR text"}],
                "usage_info": {"pages_processed": 1},
            })
        self.chat_requests.append(request)
        return JSONResponse(self.response)

    def post_sse(self, request, *, http_error_handler=None, stream_error_handler=None):
        if request.url == OCR_URL:
            self.ocr_requests.append(request)
            return iter(())
        self.stream_requests.append(request)
        return iter(self.stream_events)


@pytest.fixture
def mistral_runtime(monkeypatch):
    registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert registry.register_organization_metadata(PLUGIN.model_metadata)
    providers = ServiceProviderRegistry()
    PLUGIN.register(providers)
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", providers)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)
    return registry


def new_adapter(transport="requests"):
    return UniversalLLMAPIAdapter(
        organization="zai",
        service_provider="mistral",
        model=MODEL,
        api_key="mistral-test-key",
        transport=transport,
    )


def set_sync_recorder(adapter, transport):
    expected = RequestsSyncTransport if adapter.adapter.transport == "requests" else HttpxSyncTransport
    assert isinstance(adapter.adapter._sync_transport, expected)
    adapter.adapter._sync_transport = transport


def run_mode(adapter, mode, messages, **kwargs):
    if mode == "chat":
        return adapter.chat(messages, **kwargs)
    if mode == "stream_chat":
        return list(adapter.stream_chat(messages, **kwargs))
    if mode == "achat":
        return asyncio.run(adapter.achat(messages, **kwargs))
    async def collect():
        return [item async for item in adapter.astream_chat(messages, **kwargs)]
    return asyncio.run(collect())


def configure_async(monkeypatch, response=None, events=()):
    calls = {"ocr": [], "chat": [], "stream": []}
    response = response or {
        "id": "flash-async",
        "model": WIRE_MODEL,
        "choices": [{"message": {"role": "assistant", "content": "Done."}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }

    async def request(url, **kwargs):
        bucket = "ocr" if url == OCR_URL else "chat"
        calls[bucket].append((url, kwargs))
        if bucket == "ocr":
            return {"model": "mistral-ocr-4-1", "pages": [{"markdown": "# OCR text"}], "usage_info": {"pages_processed": 1}}
        return response

    def stream_request(url, **kwargs):
        calls["stream"].append((url, kwargs))
        async def iterator():
            for event in events:
                yield event
        return iterator()

    monkeypatch.setattr(mistral_async_client_module, "async_request", request)
    monkeypatch.setattr(mistral_async_client_module, "async_stream_request", stream_request)
    return calls


@pytest.mark.integration
def test_flash_mixed_typed_and_dict_history_and_application_owned_tool_loop(mistral_runtime):
    tool = ToolSpec(name="lookup", description="Look up a value", json_schema={
        "type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]
    })
    tool_reply = {
        "id": "flash-tool-only",
        "model": WIRE_MODEL,
        "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": '{"key":"alpha"}'}
        }]}, "finish_reason": "tool_calls"}],
    }
    adapter = new_adapter()
    transport = RecordingSyncTransport(response=tool_reply)
    set_sync_recorder(adapter, transport)
    messages = [UserMessage("Find alpha"), {"role": "assistant", "content": "Checking."}]

    expected_choices = {
        "auto": "auto",
        "none": "none",
        "any": "any",
        "lookup": {"type": "function", "function": {"name": "lookup"}},
    }
    for choice, expected_wire_choice in expected_choices.items():
        response = adapter.chat(messages, tools=[tool], tool_choice=choice)
        assert response.tool_calls and response.tool_calls[0].arguments == {"key": "alpha"}
        assert response.content in (None, "")
        assert transport.chat_requests[-1].payload["tool_choice"] == expected_wire_choice
    assert len(transport.chat_requests) == 4
    assert len(transport.ocr_requests) == 0
    # The application receives the call. No local tool implementation is invoked.

    followup = [
        *messages,
        {
            "role": "assistant",
            "content": "",
            "index": 0,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"key":"alpha"}'},
            }],
        },
        ToolMessage(content="Found it", tool_call_id="call-1"),
    ]
    transport.response = {
        "id": "flash-final", "model": WIRE_MODEL,
        "choices": [{"message": {"role": "assistant", "content": "The value is found."}, "finish_reason": "stop"}],
    }
    response = adapter.chat(followup)
    assert response.content == "The value is found."
    serialized = transport.chat_requests[-1].payload["messages"]
    assert serialized[-2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"key": "alpha"}'},
        }],
    }
    assert serialized[-1] == {"role": "tool", "content": "Found it", "tool_call_id": "call-1"}
    assert "index" not in serialized[-2]


@pytest.mark.integration
def test_flash_portable_nested_schema_and_pydantic_response_model(mistral_runtime):
    content = '{"contact":{"name":"Ada"}}'
    response_payload = {
        "id": "flash-structured", "model": WIRE_MODEL,
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }
    adapter = new_adapter()
    transport = RecordingSyncTransport(response=response_payload)
    set_sync_recorder(adapter, transport)
    response = adapter.chat([{"role": "user", "content": "Return contact"}], json_schema=INLINE_NESTED_OBJECT_SCHEMA)
    schema_payload = transport.chat_requests[-1].payload["response_format"]["json_schema"]["schema"]
    assert schema_payload == INLINE_NESTED_OBJECT_SCHEMA
    assert response.parsed_json == {"contact": {"name": "Ada"}}

    response = adapter.chat([{"role": "user", "content": "Return contact"}], response_model=NestedPydanticResponse)
    model_schema = transport.chat_requests[-1].payload["response_format"]["json_schema"]["schema"]
    assert "$defs" not in repr(model_schema) and "$ref" not in repr(model_schema)
    assert response.parsed_json == {"contact": {"name": "Ada"}}
    assert isinstance(response.parsed_model, NestedPydanticResponse)
    assert response.parsed_model.contact.name == "Ada"


@pytest.mark.integration
@pytest.mark.parametrize(("finish_reason", "extra"), [
    ("stop", {"refusal": "I cannot help with that request."}),
    ("length", {}),
])
def test_flash_refusal_and_incomplete_leave_structured_output_unset(mistral_runtime, finish_reason, extra):
    adapter = new_adapter()
    payload = {"model": WIRE_MODEL, "choices": [{"message": {"content": None, **extra}, "finish_reason": finish_reason}]}
    transport = RecordingSyncTransport(response=payload)
    set_sync_recorder(adapter, transport)
    response = adapter.chat([{"role": "user", "content": "Return JSON"}], json_schema=INLINE_NESTED_OBJECT_SCHEMA)
    assert response.parsed_json is None
    assert response.parsed_model is None
    if finish_reason == "stop":
        assert response.refusal == "I cannot help with that request."
        assert response.incomplete_reason is None
    else:
        assert response.refusal is None
        assert response.incomplete_reason == "length"


@pytest.mark.integration
def test_flash_serializes_image_url_bytes_and_data_url(mistral_runtime):
    adapter = new_adapter()
    transport = RecordingSyncTransport()
    set_sync_recorder(adapter, transport)
    adapter.chat([UserMessage("Inspect", files=[
        ImagePart(url="https://example.com/image.png"),
        ImagePart(data=b"png", media_type="image/png"),
        ImagePart(url="data:image/jpeg;base64,anBn"),
    ])])
    content = transport.chat_requests[0].payload["messages"][0]["content"]
    assert content == [
        {"type": "text", "text": "Inspect"},
        {"type": "image_url", "image_url": "https://example.com/image.png"},
        {"type": "image_url", "image_url": "data:image/png;base64,cG5n"},
        {"type": "image_url", "image_url": "data:image/jpeg;base64,anBn"},
    ]


@pytest.mark.integration
def test_flash_pdf_url_and_bytes_are_ocr_preprocessed_before_chat(mistral_runtime):
    adapter = new_adapter()
    transport = RecordingSyncTransport()
    set_sync_recorder(adapter, transport)
    adapter.chat([UserMessage("Summarize", files=[
        DocumentPart(url="https://example.com/report.pdf"),
        DocumentPart(data=b"%PDF", media_type="application/pdf"),
    ])])
    assert len(transport.ocr_requests) == 2
    assert len(transport.chat_requests) == 1
    assert [request.payload["document"] for request in transport.ocr_requests] == [
        {"type": "document_url", "document_url": "https://example.com/report.pdf"},
        {"type": "document_url", "document_url": "data:application/pdf;base64,JVBERg=="},
    ]
    chat_messages = transport.chat_requests[0].payload["messages"]
    assert "# OCR text" in chat_messages[0]["content"]
    assert "<document index=\"1\">" in chat_messages[0]["content"]
    assert "<document index=\"2\">" in chat_messages[0]["content"]


@pytest.mark.integration
@pytest.mark.parametrize(
    ("mode", "transport", "invalid_case"),
    [
        (mode, transport, invalid_case)
        for mode in ("chat", "achat", "stream_chat", "astream_chat")
        for transport in (("requests", "httpx") if mode in ("chat", "stream_chat") else ("requests",))
        for invalid_case in (("schema", "response_model", "tool_choice", "reasoning", "sampling", "buffer_chars") if "stream" in mode else ("schema", "response_model", "tool_choice", "reasoning", "sampling"))
    ],
)
@pytest.mark.parametrize("pdf_form", ["url", "bytes"])
def test_flash_pdf_invalid_options_fail_before_ocr_or_chat(mistral_runtime, monkeypatch, mode, transport, invalid_case, pdf_form):
    """Invalid calls must preflight before OCR provider I/O and Chat provider I/O."""
    adapter = new_adapter(transport)
    sync = RecordingSyncTransport()
    set_sync_recorder(adapter, sync)
    async_calls = configure_async(monkeypatch)
    pdf = DocumentPart(url="https://example.com/report.pdf") if pdf_form == "url" else DocumentPart(data=b"%PDF", media_type="application/pdf")
    messages = [UserMessage("Summarize", files=[pdf])]
    kwargs = {}
    expected_error = ValueError
    if invalid_case == "schema":
        kwargs["json_schema"] = {"type": "string"}
        expected_error = JSONSchemaError
    elif invalid_case == "response_model":
        kwargs["response_model"] = object
        expected_error = JSONSchemaError
    elif invalid_case == "tool_choice":
        kwargs["tool_choice"] = "required"
        expected_error = ToolChoiceError
    elif invalid_case == "reasoning":
        kwargs["reasoning_level"] = "unrecognized"
        expected_error = LLMReasoningLevelError
    elif invalid_case == "sampling":
        kwargs["temperature"] = 3.0
    else:
        kwargs["buffer_chars"] = 0

    with pytest.raises(expected_error):
        run_mode(adapter, mode, messages, **kwargs)

    if mode in ("chat", "stream_chat"):
        assert sync.ocr_requests == [], "preflight must reject invalid input before any OCR provider I/O"
        assert sync.chat_requests == []
        assert sync.stream_requests == []
        assert async_calls["ocr"] == []
        assert async_calls["chat"] == []
        assert async_calls["stream"] == []
    else:
        assert sync.ocr_requests == []
        assert sync.chat_requests == []
        assert sync.stream_requests == []
        assert async_calls["ocr"] == [], "preflight must reject invalid input before any OCR provider I/O"
        assert async_calls["chat"] == []
        assert async_calls["stream"] == []


@pytest.mark.integration
def test_flash_async_pdf_url_and_bytes_use_separate_ocr_and_chat_recorders(mistral_runtime, monkeypatch):
    adapter = new_adapter()
    calls = configure_async(monkeypatch)
    response = asyncio.run(adapter.achat([UserMessage("Summarize", files=[
        DocumentPart(url="https://example.com/one.pdf"), DocumentPart(data=b"%PDF", media_type="application/pdf")
    ])]))
    assert response.content == "Done."
    assert [url for url, _ in calls["ocr"]] == [OCR_URL, OCR_URL]
    assert [url for url, _ in calls["chat"]] == [CHAT_URL]
    assert [kwargs["payload"]["document"] for _, kwargs in calls["ocr"]] == [
        {"type": "document_url", "document_url": "https://example.com/one.pdf"},
        {"type": "document_url", "document_url": "data:application/pdf;base64,JVBERg=="},
    ]
    chat_messages = calls["chat"][0][1]["payload"]["messages"]
    assert "# OCR text" in chat_messages[0]["content"]
    assert "<document index=\"1\">" in chat_messages[0]["content"]
    assert "<document index=\"2\">" in chat_messages[0]["content"]
