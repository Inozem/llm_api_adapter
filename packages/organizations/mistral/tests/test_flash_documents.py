"""OCR ownership and accounting tests for hosted GLM Flash on Mistral."""

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
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.httpx_transport import HttpxSyncTransport
from llm_api_adapter.llms.requests_transport import RequestsSyncTransport
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_mistral.plugin import PLUGIN


MODEL = "glm-5.3-flash"
WIRE_MODEL = "zai-glm-5-3-flash"
CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
OCR_URL = "https://api.mistral.ai/v1/ocr"
TOKEN_USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 10,
    "total_tokens": 110,
    "prompt_tokens_details": {"cached_tokens": 0},
}


def chat_payload():
    return {
        "id": "flash-ocr-chat",
        "model": WIRE_MODEL,
        "choices": [{"message": {"role": "assistant", "content": "PDF summary."}}],
        "usage": dict(TOKEN_USAGE),
    }


def ocr_payload(pages_processed=1):
    result = {
        "model": "mistral-ocr-4-1",
        "pages": [{"markdown": "# OCR text"}],
    }
    if pages_processed is not ...:
        result["usage_info"] = {"pages_processed": pages_processed}
    return result


class RecordingTransport:
    def __init__(self, *, page_counts=(1,), response=None, stream_events=()):
        self.page_counts = list(page_counts)
        self.response = response or chat_payload()
        self.stream_events = list(stream_events)
        self.ocr_requests = []
        self.chat_requests = []
        self.stream_requests = []

    def post_json(self, request, *, http_error_handler=None):
        if request.url == OCR_URL:
            self.ocr_requests.append(request)
            return JSONResponse(ocr_payload(self.page_counts.pop(0)))
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
    """Register only Mistral so hosted Flash cannot rely on Z.ai metadata."""
    registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert registry.register_organization_metadata(PLUGIN.model_metadata)
    providers = ServiceProviderRegistry()
    PLUGIN.register(providers)
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", providers)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)
    return registry


def new_adapter(transport="requests", *, organization="zai", model=MODEL):
    return UniversalLLMAPIAdapter(
        organization=organization,
        service_provider="mistral",
        model=model,
        api_key="mistral-test-key",
        transport=transport,
    )


def set_sync_recorder(adapter, recorder):
    expected = RequestsSyncTransport if adapter.adapter.transport == "requests" else HttpxSyncTransport
    assert isinstance(adapter.adapter._sync_transport, expected)
    adapter.adapter._sync_transport = recorder


def pdf_messages():
    return [UserMessage("Summarize these PDFs.", files=[
        DocumentPart(url="https://example.com/first.pdf"),
        DocumentPart(data=b"%PDF second", media_type="application/pdf"),
    ])]


def stream_event():
    return SSEEvent(event=None, data={
        "id": "flash-ocr-chat",
        "model": WIRE_MODEL,
        "choices": [{"index": 0, "delta": {"content": "PDF summary."}, "finish_reason": "stop"}],
        "usage": dict(TOKEN_USAGE),
    })


def configure_async(monkeypatch, *, page_counts=(1,), response=None, events=()):
    calls = {"ocr": [], "chat": [], "stream": []}
    remaining_counts = list(page_counts)
    response = response or chat_payload()

    async def request(url, **kwargs):
        bucket = "ocr" if url == OCR_URL else "chat"
        calls[bucket].append((url, kwargs))
        if bucket == "ocr":
            return ocr_payload(remaining_counts.pop(0))
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


async def collect_async_stream(adapter, messages, **kwargs):
    return [item async for item in adapter.astream_chat(messages, **kwargs)]


@pytest.mark.integration
@pytest.mark.parametrize("api_kind", ["sync", "async", "sync-stream", "async-stream"])
def test_hosted_flash_ocr_is_mistral_owned_and_accounted_once(mistral_runtime, monkeypatch, api_kind):
    """Mistral owns both calls, credentials, and the separate per-PDF OCR costs."""
    adapter = new_adapter()
    sync = RecordingTransport(page_counts=(2, 5), stream_events=(stream_event(),))
    set_sync_recorder(adapter, sync)
    async_calls = configure_async(monkeypatch, page_counts=(2, 5), events=(stream_event(),))
    messages = pdf_messages()
    completed = []

    if api_kind == "sync":
        response = adapter.chat(messages)
    elif api_kind == "async":
        response = asyncio.run(adapter.achat(messages))
    elif api_kind == "sync-stream":
        list(adapter.stream_chat(messages, on_done=completed.append))
        response = completed[0]
    else:
        asyncio.run(collect_async_stream(adapter, messages, on_done=completed.append))
        response = completed[0]

    if api_kind.startswith("sync"):
        ocr_requests = sync.ocr_requests
        chat_requests = sync.chat_requests
        stream_requests = sync.stream_requests
        assert [request.url for request in ocr_requests] == [OCR_URL, OCR_URL]
        assert [request.url for request in (chat_requests or stream_requests)] == [CHAT_URL]
        calls = [*ocr_requests, *(chat_requests or stream_requests)]
        assert all(request.headers.get("Authorization") == "Bearer mistral-test-key" for request in calls)
    else:
        assert [url for url, _ in async_calls["ocr"]] == [OCR_URL, OCR_URL]
        last_calls = async_calls["stream"] if api_kind == "async-stream" else async_calls["chat"]
        assert [url for url, _ in last_calls] == [CHAT_URL]
        calls = [*async_calls["ocr"], *last_calls]
        assert all(
            kwargs.get("headers", {}).get("Authorization") == "Bearer mistral-test-key"
            for _, kwargs in calls
        )

    assert response.content == "PDF summary."
    assert len(sync.ocr_requests) == (2 if api_kind.startswith("sync") else 0)
    assert len(async_calls["ocr"]) == (0 if api_kind.startswith("sync") else 2)
    assert response.cost_input is not None and response.cost_input > 0
    assert response.cost_output is not None and response.cost_output > 0
    assert response.cost_breakdown is not None
    ocr_lines = [item for item in response.cost_breakdown if item.operation == "ocr"]
    assert [
        (item.model, item.unit, item.quantity, item.rate, item.cost)
        for item in ocr_lines
    ] == [
        ("mistral-ocr-4-1", "page", 2.0, 0.004, 0.008),
        ("mistral-ocr-4-1", "page", 5.0, 0.004, 0.020),
    ]
    token_total = response.cost_input + response.cost_output
    assert response.cost_total == pytest.approx(token_total + 0.028)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("bad_usage_info", "expected_known_quantities"),
    [(None, [2.0]), ({"pages_processed": "unknown"}, [2.0])],
    ids=["missing-page-count", "malformed-page-count"],
)
def test_hosted_flash_unknown_page_counts_do_not_estimate_or_hide_known_costs(
    mistral_runtime, bad_usage_info, expected_known_quantities
):
    adapter = new_adapter()
    payloads = [ocr_payload(2), ocr_payload()]
    if bad_usage_info is not None:
        payloads[1]["usage_info"] = bad_usage_info

    class PayloadTransport(RecordingTransport):
        def post_json(self, request, *, http_error_handler=None):
            if request.url == OCR_URL:
                self.ocr_requests.append(request)
                return JSONResponse(payloads.pop(0))
            self.chat_requests.append(request)
            return JSONResponse(chat_payload())

    recorder = PayloadTransport()
    set_sync_recorder(adapter, recorder)
    response = adapter.chat(pdf_messages())

    assert len(recorder.ocr_requests) == 2
    assert len(recorder.chat_requests) == 1
    assert response.cost_input is not None and response.cost_input > 0
    assert response.cost_output is not None and response.cost_output > 0
    assert [item.quantity for item in response.cost_breakdown if item.operation == "ocr"] == expected_known_quantities
    assert response.cost_total is None


@pytest.mark.integration
def test_hosted_flash_unavailable_ocr_meter_keeps_tokens_but_not_total(mistral_runtime, monkeypatch):
    adapter = new_adapter()
    monkeypatch.setattr(adapter.adapter, "_ocr_meter", lambda: None)
    recorder = RecordingTransport(page_counts=(3, 4))
    set_sync_recorder(adapter, recorder)

    response = adapter.chat(pdf_messages())

    assert len(recorder.ocr_requests) == 2
    assert len(recorder.chat_requests) == 1
    assert response.cost_input is not None and response.cost_input > 0
    assert response.cost_output is not None and response.cost_output > 0
    assert response.cost_breakdown == []
    assert response.cost_total is None


@pytest.mark.integration
def test_native_mistral_pdf_ocr_retains_native_processing_and_accounting(mistral_runtime):
    adapter = new_adapter(organization="mistral", model="mistral-large-4")
    recorder = RecordingTransport(page_counts=(3, 4), response={
        **chat_payload(), "model": "mistral-large-4",
    })
    set_sync_recorder(adapter, recorder)

    response = adapter.chat(pdf_messages())

    assert len(recorder.ocr_requests) == 2
    assert len(recorder.chat_requests) == 1
    assert response.cost_breakdown is not None
    assert [item.quantity for item in response.cost_breakdown if item.operation == "ocr"] == [3.0, 4.0]
    assert response.cost_total == pytest.approx(
        response.cost_input + response.cost_output + 0.028
    )
