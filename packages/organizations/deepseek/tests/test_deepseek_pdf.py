"""Controlled request tests for the direct DeepSeek PDF-bytes route."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, replace
import json
from types import SimpleNamespace
from pathlib import Path
import sys
from typing import Any

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (str(PACKAGE_ROOT / "src"), str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIServerError,
)
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.transports import SSEEvent
from llm_api_adapter.models.messages.chat_message import AIMessage, UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart, ImagePart
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.llms import async_streaming, httpx_transport
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_deepseek.plugin import PLUGIN
import llm_api_adapter_deepseek.documents as document_module
import llm_api_adapter_deepseek.adapter as adapter_module


PDF_A = b"%PDF-a-private"
PDF_B = b"%PDF-b-private"
@pytest.fixture
def deepseek_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    models = RegistrySpec()
    assert models.register_organization_metadata(PLUGIN.model_metadata)
    providers = ServiceProviderRegistry()
    PLUGIN.register(providers)
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", models)
    monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", providers)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", models)


def _adapter() -> UniversalLLMAPIAdapter:
    return UniversalLLMAPIAdapter(
        organization="deepseek",
        model="deepseek-flash",
        api_key="private-test-key",
    )


def _http_adapter(transport: str) -> UniversalLLMAPIAdapter:
    return UniversalLLMAPIAdapter(
        organization="deepseek",
        model="deepseek-flash",
        api_key="private-test-key",
        transport=transport,
    )


def _response() -> dict[str, Any]:
    return {
        "object": "response",
        "id": "response-pdf-test",
        "model": "deepseek-flash",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "PDF read."}],
        }],
        "usage": {
            "input_tokens": 12,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 4,
            "total_tokens": 16,
        },
    }


class _FakeHttpStatusError(Exception):
    def __init__(self, request: Any, response: Any) -> None:
        super().__init__("controlled HTTP failure")
        self.request = request
        self.response = response


class _ControlledHTTPResponse:
    def __init__(
        self,
        provider: "_ControlledProvider",
        request: Any,
        payload: Any,
        *,
        status_code: int = 200,
        events: list[tuple[str, Any]] | None = None,
    ) -> None:
        self.provider = provider
        self.request = request
        self.payload = payload
        self.status_code = status_code
        self.events = events or []
        self.text = json.dumps(payload) if not isinstance(payload, str) else payload
        self.closed = False

    def json(self) -> Any:
        if isinstance(self.payload, str):
            return json.loads(self.payload)
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return
        if self.provider.backend == "httpx":
            raise _FakeHttpStatusError(self.request, self)
        import requests
        raise requests.HTTPError("controlled HTTP failure", response=self)

    def iter_lines(self, decode_unicode: bool = False):
        del decode_unicode
        for event_name, data in self.events:
            yield f"event: {event_name}"
            yield "data: " + json.dumps(data)
            yield ""

    async def aiter_lines(self):
        for line in self.iter_lines():
            yield line

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


class _ControlledProvider:
    """In-memory HTTP endpoint that records upload/generation ordering."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.calls: list[dict[str, Any]] = []
        self.next_file = 0
        self.uploaded: list[tuple[bytes, str]] = []
        self.http_error = False
        self.stream_failure = False
        self.metadata_echo = True
        self.reasoning_echo = False
        self.async_upload_gate: asyncio.Event | None = None
        self.async_upload_started: asyncio.Event | None = None
        self.async_upload_arrivals = 0
        self.async_clients_closed = 0

    def post(self, url: str, **kwargs: Any) -> _ControlledHTTPResponse:
        request = {"method": "POST", "url": url, **kwargs}
        self.calls.append(request)
        if url.endswith("/files"):
            self.next_file += 1
            files = kwargs.get("files")
            content = (
                files[0][1][1]
                if isinstance(files, list) and files
                else kwargs.get("content", b"")
            )
            pdf = next((candidate for candidate in (PDF_A, PDF_B) if candidate in content), b"")
            receipt = f"receipt-{self.next_file}"
            request["receipt"] = receipt
            request["uploaded_pdf"] = pdf
            self.uploaded.append((pdf, receipt))
            return _ControlledHTTPResponse(
                self,
                request,
                {"id": receipt},
            )
        payload = kwargs.get("json") or {}
        result = _response()
        if self.reasoning_echo:
            result["output"].extend([
                {
                    "type": "reasoning",
                    "content": [{
                        "type": "reasoning_text",
                        "text": f"receipt-{self.next_file} {PDF_A.decode()}",
                    }],
                },
                {
                    "type": "reasoning",
                    "content": [{
                        "type": "reasoning_text",
                        "text": "ordinary replay survives",
                    }],
                },
            ])
        if self.metadata_echo:
            result["metadata"] = {
                "file_id": f"receipt-{self.next_file}",
                "file_data": PDF_A.decode(),
                "echo": f"receipt-{self.next_file}",
            }
        if self.http_error:
            return _ControlledHTTPResponse(
                self,
                request,
                {"error": {"type": "invalid_request_error", "message": f"echo receipt-{self.next_file} {PDF_A!r}"}},
                status_code=400,
            )
        if kwargs.get("stream"):
            if self.stream_failure:
                events = [(
                    "response.failed",
                    {"type": "response.failed", "response": {
                        "object": "response",
                        "error": {"message": f"echo receipt-{self.next_file} {PDF_A!r}"},
                    }},
                )]
            else:
                response = dict(result)
                response["input"] = payload.get("input")
                events = [
                    ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "PDF read."}),
                    ("response.completed", {"type": "response.completed", "response": response}),
                ]
            return _ControlledHTTPResponse(self, request, None, events=events)
        return _ControlledHTTPResponse(self, request, result)


def _fake_httpx(provider: _ControlledProvider) -> Any:
    class Client:
        def post(self, url: str, **kwargs: Any):
            return provider.post(url, **kwargs)

        def build_request(self, method: str, url: str, **kwargs: Any):
            return {"method": method, "url": url, **kwargs}

        def send(self, request: dict[str, Any], *, stream: bool = False):
            return provider.post(request["url"], **{
                key: value for key, value in request.items() if key not in {"method", "url"}
            }, stream=stream)

        def close(self) -> None:
            pass

    class AsyncClient:
        async def post(self, url: str, **kwargs: Any):
            if url.endswith("/files") and provider.async_upload_gate is not None:
                provider.async_upload_arrivals += 1
                if provider.async_upload_started is not None:
                    provider.async_upload_started.set()
                await provider.async_upload_gate.wait()
            return provider.post(url, **kwargs)

        def build_request(self, method: str, url: str, **kwargs: Any):
            return {"method": method, "url": url, **kwargs}

        async def send(self, request: dict[str, Any], *, stream: bool = False):
            return provider.post(request["url"], **{
                key: value for key, value in request.items() if key not in {"method", "url"}
            }, stream=stream)

        async def aclose(self) -> None:
            provider.async_clients_closed += 1

    return SimpleNamespace(
        Client=Client,
        AsyncClient=AsyncClient,
        Request=__import__("httpx").Request,
        TimeoutException=TimeoutError,
        HTTPStatusError=_FakeHttpStatusError,
        RequestError=OSError,
    )


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["requests", "httpx"])
@pytest.mark.parametrize("mode", ["chat", "stream_chat"])
def test_real_sync_transport_orders_multipart_before_generation(
    deepseek_runtime,
    monkeypatch,
    backend: str,
    mode: str,
) -> None:
    provider = _ControlledProvider(backend)
    if backend == "requests":
        import requests
        monkeypatch.setattr(requests, "post", provider.post)
    else:
        monkeypatch.setattr(httpx_transport, "_require_httpx", lambda: _fake_httpx(provider))
    adapter = _http_adapter(backend)
    message = UserMessage("Read PDF", files=[DocumentPart(data=PDF_A, media_type="application/pdf")])
    completed = []
    if mode == "chat":
        response = adapter.chat([message])
    else:
        list(adapter.stream_chat([message], on_done=completed.append))
        response = completed[0]

    assert [call["url"].rsplit("/", 1)[-1] for call in provider.calls] == ["files", "responses"]
    upload, generation = provider.calls
    upload_fields = (
        list(upload["data"].items())
        if backend == "httpx"
        else upload["data"]
    )
    assert upload_fields == [
        ("purpose", "rich_layout_recognition"),
        ("expires_after[anchor]", "created_at"),
        ("expires_after[seconds]", "3600"),
    ]
    assert upload["files"][0][1][1:] == (PDF_A, "application/pdf")
    assert generation["json"]["input"][0]["content"] == [
        {"type": "input_text", "text": "Read PDF"},
        {"type": "input_file", "file_id": "receipt-1"},
    ]
    assert "receipt-1" not in repr(response)
    assert PDF_A.decode() not in repr(response)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["chat", "stream_chat"])
def test_pdf_replay_export_redacts_receipt_and_document_but_keeps_safe_replay(
    deepseek_runtime,
    monkeypatch,
    mode: str,
) -> None:
    provider = _ControlledProvider("requests")
    provider.reasoning_echo = True
    import requests
    monkeypatch.setattr(requests, "post", provider.post)
    adapter = _http_adapter("requests")
    message = UserMessage("Read PDF", files=[DocumentPart(data=PDF_A, media_type="application/pdf")])

    if mode == "chat":
        response = adapter.chat([message])
    else:
        completed = []
        assert list(adapter.stream_chat([message], on_done=completed.append)) == ["PDF read."]
        response = completed[0]

    assert response.content == "PDF read."
    replay = response.provider_data["deepseek.reasoning_replay"]["items"]
    replay_text = [part["text"] for item in replay for part in item["content"]]
    assert "ordinary replay survives" in replay_text
    assert "[DeepSeek PDF content redacted]" in replay_text
    assert all("receipt-1" not in text and PDF_A.decode() not in text for text in replay_text)


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["achat", "astream_chat"])
def test_real_async_multipart_helper_orders_upload_before_generation(
    deepseek_runtime,
    monkeypatch,
    mode: str,
) -> None:
    provider = _ControlledProvider("async")
    monkeypatch.setattr(async_streaming, "_require_httpx", lambda: _fake_httpx(provider))
    adapter = _adapter()
    message = UserMessage("Read PDF", files=[DocumentPart(data=PDF_A, media_type="application/pdf")])

    async def run() -> ChatResponse:
        completed = []
        if mode == "achat":
            return await adapter.achat([message])
        async for _ in adapter.astream_chat([message], on_done=completed.append):
            pass
        return completed[0]

    response = asyncio.run(run())
    assert [call["url"].rsplit("/", 1)[-1] for call in provider.calls] == ["files", "responses"]
    upload, generation = provider.calls
    assert b"rich_layout_recognition" in upload["content"]
    assert upload["content"].index(b"rich_layout_recognition") < upload["content"].index(b'name="file"')
    assert PDF_A in upload["content"]
    assert generation["json"]["input"][0]["content"] == [
        {"type": "input_text", "text": "Read PDF"},
        {"type": "input_file", "file_id": "receipt-1"},
    ]
    assert "receipt-1" not in repr(response)
    assert PDF_A.decode() not in repr(response)


@pytest.mark.unit
def test_concurrent_async_pdf_requests_keep_receipts_isolated(deepseek_runtime, monkeypatch) -> None:
    provider = _ControlledProvider("async")
    provider.async_upload_gate = asyncio.Event()
    provider.async_upload_started = asyncio.Event()
    provider.metadata_echo = False
    monkeypatch.setattr(async_streaming, "_require_httpx", lambda: _fake_httpx(provider))
    adapter = _adapter()

    async def run() -> None:
        first = asyncio.create_task(adapter.achat([
            UserMessage("request A", files=[DocumentPart(data=PDF_A, media_type="application/pdf")]),
        ]))
        second = asyncio.create_task(adapter.achat([
            UserMessage("request B", files=[DocumentPart(data=PDF_B, media_type="application/pdf")]),
        ]))
        await provider.async_upload_started.wait()
        while provider.async_upload_arrivals < 2:
            await asyncio.sleep(0)
        provider.async_upload_gate.set()
        await asyncio.gather(first, second)

    asyncio.run(run())
    uploads = {pdf: receipt for pdf, receipt in provider.uploaded}
    generations = [call for call in provider.calls if call["url"].endswith("/responses")]
    assert len(uploads) == 2
    assert len(generations) == 2
    by_prompt = {
        call["json"]["input"][0]["content"][0]["text"]:
        call["json"]["input"][0]["content"][1]["file_id"]
        for call in generations
    }
    assert by_prompt == {"request A": uploads[PDF_A], "request B": uploads[PDF_B]}
    for index, call in enumerate(provider.calls):
        if not call["url"].endswith("/responses"):
            continue
        receipt = call["json"]["input"][0]["content"][1]["file_id"]
        assert any(
            previous["url"].endswith("/files") and previous.get("receipt") == receipt
            for previous in provider.calls[:index]
        )


@pytest.mark.unit
def test_async_upload_cancellation_preserves_cancel_and_skips_generation(
    deepseek_runtime,
    monkeypatch,
) -> None:
    provider = _ControlledProvider("async")
    provider.async_upload_gate = asyncio.Event()
    provider.async_upload_started = asyncio.Event()
    monkeypatch.setattr(async_streaming, "_require_httpx", lambda: _fake_httpx(provider))
    adapter = _adapter()

    async def run() -> None:
        task = asyncio.create_task(adapter.achat([
            UserMessage("cancel", files=[DocumentPart(data=PDF_A, media_type="application/pdf")]),
        ]))
        await provider.async_upload_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert not any(call["url"].endswith("/responses") for call in provider.calls)
    assert provider.async_clients_closed == 1


@pytest.mark.unit
@pytest.mark.parametrize("failure_mode", ["http", "sse"])
def test_provider_failures_redact_pdf_material_before_sdk_logs(
    deepseek_runtime,
    monkeypatch,
    caplog,
    failure_mode: str,
) -> None:
    import requests

    provider = _ControlledProvider("requests")
    monkeypatch.setattr(requests, "post", provider.post)
    adapter = _http_adapter("requests")
    provider.http_error = failure_mode == "http"
    provider.stream_failure = failure_mode == "sse"
    message = [UserMessage("Read", files=[DocumentPart(data=PDF_A, media_type="application/pdf")])]
    completed = []
    if failure_mode == "http":
        with pytest.raises(LLMAPIClientError) as caught:
            adapter.chat(message)
    else:
        with pytest.raises(LLMAPIClientError) as caught:
            list(adapter.stream_chat(message, on_done=completed.append))
        assert completed == []
    assert "DeepSeek Responses request failed" in caught.value.detail
    assert "receipt-1" not in str(caught.value)
    assert PDF_A.decode() not in caplog.text
    assert "receipt-1" not in caplog.text


@pytest.mark.unit
def test_pdf_stream_callback_error_identity_is_preserved(deepseek_runtime, monkeypatch) -> None:
    import requests

    provider = _ControlledProvider("requests")
    monkeypatch.setattr(requests, "post", provider.post)
    adapter = _http_adapter("requests")
    callback_error = LLMAPIClientError(detail="caller callback failed")

    def fail_callback(_delta: str) -> None:
        raise callback_error

    with pytest.raises(LLMAPIClientError) as caught:
        list(adapter.stream_chat(
            [UserMessage("Read", files=[DocumentPart(data=PDF_A, media_type="application/pdf")])],
            on_delta=fail_callback,
        ))
    assert caught.value is callback_error


@pytest.mark.unit
def test_no_pdf_generation_error_keeps_original_identity_and_cause(deepseek_runtime) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    cause = ValueError("underlying provider cause")
    provider_error = LLMAPIClientError(detail="ordinary no-PDF failure")

    def fail_generation(**kwargs: Any) -> Any:
        raise provider_error from cause

    adapter.adapter._client.create = fail_generation
    with pytest.raises(LLMAPIClientError) as caught:
        adapter.chat([UserMessage("No files")])
    assert caught.value is provider_error
    assert caught.value.__cause__ is cause
    assert [kind for kind, _ in calls] == []


def _configure_sync(adapter: UniversalLLMAPIAdapter, calls: list[tuple[str, Any]]) -> None:
    client = adapter.adapter._client

    def upload(content: bytes, timeout: float | None = None) -> str:
        calls.append(("upload", (content, timeout)))
        return f"receipt-{sum(kind == 'upload' for kind, _ in calls)}"

    def create(*, model: str, timeout: float | None = None, **parameters: Any) -> dict[str, Any]:
        calls.append(("generate", parameters))
        return _response()

    def stream(*, model: str, timeout: float | None = None, **parameters: Any):
        calls.append(("generate", parameters))
        response = _response()
        return iter([SSEEvent(event="response.completed", data={"type": "response.completed", "response": response})])

    client._upload_pdf = upload
    client.create = create
    client.stream = stream


def _configure_async(adapter: UniversalLLMAPIAdapter, calls: list[tuple[str, Any]]) -> None:
    client = adapter.adapter._async_client

    async def upload(content: bytes, timeout: float | None = None) -> str:
        calls.append(("upload", (content, timeout)))
        return f"receipt-{sum(kind == 'upload' for kind, _ in calls)}"

    async def create(*, model: str, timeout: float | None = None, **parameters: Any) -> dict[str, Any]:
        calls.append(("generate", parameters))
        return _response()

    async def stream(*, model: str, timeout: float | None = None, **parameters: Any):
        calls.append(("generate", parameters))
        response = _response()
        yield SSEEvent(event="response.completed", data={"type": "response.completed", "response": response})

    client._upload_pdf = upload
    client.create = create
    client.stream = stream


def _request_input(generated: tuple[str, Any]) -> list[dict[str, Any]]:
    assert generated[0] == "generate"
    return generated[1]["input"]


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["chat", "achat", "stream_chat", "astream_chat"])
def test_pdf_upload_precedes_generation_and_replaces_ordered_slots(
    deepseek_runtime,
    mode: str,
) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    _configure_async(adapter, calls)
    message = UserMessage(
        "Read these files",
        files=[
            DocumentPart(data=PDF_A, media_type="application/pdf"),
            ImagePart(data=b"image", media_type="image/png"),
            DocumentPart(data=PDF_B, media_type="application/pdf"),
        ],
    )

    if mode == "chat":
        response = adapter.chat([message], timeout_s=19)
    elif mode == "achat":
        response = asyncio.run(adapter.achat([message], timeout_s=19))
    elif mode == "stream_chat":
        completed = []
        stream = adapter.stream_chat([message], timeout_s=19, on_done=completed.append)
        list(stream)
        response = completed[0]
    else:
        async def collect():
            completed = []
            stream = adapter.astream_chat([message], timeout_s=19, on_done=completed.append)
            values = [part async for part in stream]
            return completed[0], values
        response, values = asyncio.run(collect())

    assert [kind for kind, _ in calls] == ["upload", "upload", "generate"]
    assert [item[1][0] for item in calls[:2]] == [PDF_A, PDF_B]
    assert [item[1][1] for item in calls[:2]] == [19, 19]
    content = _request_input(calls[2])[0]["content"]
    assert content == [
        {"type": "input_text", "text": "Read these files"},
        {"type": "input_file", "file_id": "receipt-1"},
        {"type": "input_image", "image_url": "data:image/png;base64,aW1hZ2U="},
        {"type": "input_file", "file_id": "receipt-2"},
    ]
    assert response.cost_input is not None
    assert response.cost_output is not None
    assert response.cost_total is None
    assert "receipt-" not in repr(response)


@pytest.mark.unit
def test_pdf_preflight_rejects_late_invalid_items_before_any_upload(deepseek_runtime) -> None:
    cases = [
        [UserMessage("read", files=[
            DocumentPart(data=PDF_A, media_type="application/pdf"),
            DocumentPart(url="https://example.test/x.pdf"),
        ])],
        [{"role": "user", "content": [
            {"type": "input_file", "file_data": f"data:application/pdf;base64,{base64.b64encode(PDF_A).decode()}"},
            {"type": "input_file", "file_data": "data:application/pdf;base64,%%bad"},
        ]}],
        [{"role": "user", "content": [
            {"type": "input_file", "file_data": f"data:application/pdf;base64,{base64.b64encode(PDF_A).decode()}", "file_url": "https://example.test/a.pdf"},
        ]}],
        [{"role": "user", "content": [{"type": "input_file", "file_data": 7}]}],
        [{"role": "assistant", "content": [{"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "cGRm"}}]}],
    ]
    for messages in cases:
        adapter = _adapter()
        calls: list[tuple[str, Any]] = []
        _configure_sync(adapter, calls)
        with pytest.raises(ValueError):
            adapter.chat(messages)
        assert calls == []


@pytest.mark.unit
def test_pdf_size_and_count_bounds_are_checked_before_upload(deepseek_runtime, monkeypatch) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    monkeypatch.setattr(document_module, "MAX_PDF_BYTES", len(PDF_A) - 1)
    with pytest.raises(ValueError, match="100 MiB"):
        adapter.chat([UserMessage("Read", files=[DocumentPart(data=PDF_A, media_type="application/pdf")])])
    assert calls == []
    monkeypatch.setattr(document_module, "MAX_PDF_BYTES", 100 * 1024 * 1024)
    monkeypatch.setattr(document_module, "MAX_PDFS_PER_REQUEST", 1)
    with pytest.raises(ValueError, match="at most 20 PDFs"):
        adapter.chat([UserMessage("Read", files=[
            DocumentPart(data=PDF_A, media_type="application/pdf"),
            DocumentPart(data=PDF_B, media_type="application/pdf"),
        ])])
    assert calls == []


@pytest.mark.unit
def test_typed_pdf_copy_preserves_subclass_and_invalid_file_collection(deepseek_runtime) -> None:
    @dataclass
    class ExtendedUserMessage(UserMessage):
        label: str = "preserve"

    uri = "data:application/pdf;base64," + base64.b64encode(PDF_A).decode()
    original = ExtendedUserMessage(
        "Read",
        files=[DocumentPart(url=uri)],
        label="caller value",
    )
    prepared = document_module.validate_raw_document_forms([original])[0]
    assert isinstance(prepared, ExtendedUserMessage)
    assert prepared.label == "caller value"
    assert original.files[0].url == uri
    assert prepared.files[0].data == PDF_A

    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    malformed = UserMessage("Read", files=(DocumentPart(data=PDF_A, media_type="application/pdf"),))
    with pytest.raises(ValueError, match="files must be a list"):
        adapter.chat([malformed])
    assert calls == []


@pytest.mark.unit
def test_valid_inline_pdf_data_uri_uploads_decoded_bytes(deepseek_runtime) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    uri = "data:application/pdf;base64," + base64.b64encode(PDF_A).decode()
    adapter.chat([{"role": "user", "content": [
        {"type": "text", "text": "Read this PDF"},
        {"type": "input_file", "file_data": uri},
    ]}])
    assert calls[0] == ("upload", (PDF_A, None))


@pytest.mark.unit
@pytest.mark.parametrize("form", ["typed", "input_file", "anthropic"])
def test_inline_pdf_base64_is_decoded_once_per_attachment(
    deepseek_runtime,
    monkeypatch,
    form,
) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    encoded = base64.b64encode(PDF_A).decode()
    uri = f"data:application/pdf;base64,{encoded}"
    if form == "typed":
        messages = [UserMessage("Read", files=[DocumentPart(url=uri)])]
    elif form == "input_file":
        messages = [{"role": "user", "content": [{"type": "input_file", "file_data": uri}]}]
    else:
        messages = [{"role": "user", "content": [{
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": encoded},
        }]}]

    original_decode = base64.b64decode
    decoded_inputs: list[Any] = []

    def counted_decode(value, *args, **kwargs):
        decoded_inputs.append(value)
        return original_decode(value, *args, **kwargs)

    monkeypatch.setattr(document_module.base64, "b64decode", counted_decode)
    adapter.chat(messages)

    assert len(decoded_inputs) == 1
    assert calls[0] == ("upload", (PDF_A, None))
    content = _request_input(next(item for item in calls if item[0] == "generate"))[0]["content"]
    assert content[-1] == {"type": "input_file", "file_id": "receipt-1"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "part",
    [
        lambda uri: {"type": "input_file", "file_data": uri, "file_url": ""},
        lambda uri: {"type": "file", "file": {"file_data": uri, "file_url": "https://ignored.test/file"}},
    ],
)
def test_supported_inline_pdf_forms_keep_existing_url_field_precedence(
    deepseek_runtime,
    part,
) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    uri = "data:application/pdf;base64," + base64.b64encode(PDF_A).decode()

    adapter.chat([{"role": "user", "content": [part(uri)]}])

    assert calls[0] == ("upload", (PDF_A, None))


@pytest.mark.unit
@pytest.mark.parametrize(
    ("message", "expected_text"),
    [
        pytest.param(
            lambda uri: {"role": "user", "content": "Read this", "files": [
                {"type": "input_file", "file_data": uri},
            ]},
            "Read this",
            id="string-content-with-files",
        ),
        pytest.param(
            lambda uri: {"role": "user", "content": [
                {"type": "input_text", "text": "Read"},
                {"type": "input_file", "file_data": uri},
                {"type": "input_text", "text": "this"},
            ]},
            "Read this",
            id="input-text-array",
        ),
        pytest.param(
            lambda uri: {"role": "user", "content": [
                {"type": "input_file", "file_data": uri},
            ]},
            "",
            id="attachment-only-array",
        ),
    ],
)
def test_raw_message_shapes_keep_pdf_and_text(deepseek_runtime, message, expected_text) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    uri = "data:application/pdf;base64," + base64.b64encode(PDF_A).decode()
    adapter.chat([message(uri)])
    content = _request_input(next(item for item in calls if item[0] == "generate"))[0]["content"]
    assert content == [
        {"type": "input_text", "text": expected_text},
        {"type": "input_file", "file_id": "receipt-1"},
    ]
    assert calls[0] == ("upload", (PDF_A, None))


@pytest.mark.unit
def test_pdf_support_requires_exact_registered_pass_behavior(deepseek_runtime) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    model_spec = adapter.adapter.model_spec
    adapter.adapter.model_spec = replace(
        model_spec,
        capability_exceptions=tuple(
            item
            for item in model_spec.require_capability_profile()
            if item.capability_id != "pdf_bytes"
        ),
    )
    with pytest.raises(NotImplementedError, match="not registered for PDF byte inputs"):
        adapter.chat([UserMessage("Read", files=[DocumentPart(data=PDF_A, media_type="application/pdf")])])
    assert calls == []


@pytest.mark.unit
def test_prepared_payload_does_not_reread_mutated_history_or_replay(deepseek_runtime) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    user = UserMessage("Original prompt", files=[DocumentPart(data=PDF_A, media_type="application/pdf")])
    assistant = AIMessage("Earlier answer")
    previous = ChatResponse(
        model="deepseek-flash",
        response_id="prior-response",
        provider_data={"deepseek.reasoning_replay": {
            "model": "deepseek-flash",
            "response_id": "prior-response",
            "items": [{"type": "reasoning", "content": [
                {"type": "reasoning_text", "text": "prepared replay"},
            ]}],
        }},
    )

    def mutate_after_preflight(content: bytes, timeout: float | None = None) -> str:
        previous.provider_data.clear()
        assistant.content = "mutated history"
        user.content = "mutated prompt"
        user.files.clear()
        calls.append(("upload", (content, timeout)))
        return "receipt-1"

    adapter.adapter._client._upload_pdf = mutate_after_preflight
    adapter.chat([assistant, user], previous_response=previous)
    payload = next(value for kind, value in calls if kind == "generate")
    assert payload["input"][0] == {
        "type": "reasoning",
        "content": [{"type": "reasoning_text", "text": "prepared replay"}],
    }
    assert payload["input"][1] == {"role": "assistant", "content": "Earlier answer"}
    assert payload["input"][2]["content"] == [
        {"type": "input_text", "text": "Original prompt"},
        {"type": "input_file", "file_id": "receipt-1"},
    ]


@pytest.mark.unit
def test_repeated_pdf_uploads_again_and_custom_endpoint_stays_rejected(deepseek_runtime) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    message = UserMessage("Read", files=[DocumentPart(data=PDF_A, media_type="application/pdf")])
    adapter.chat([message])
    adapter.chat([message])
    assert [kind for kind, _ in calls] == ["upload", "generate", "upload", "generate"]

    adapter.adapter.endpoint = "https://proxy.example/responses"
    calls.clear()
    with pytest.raises(ValueError, match="official Responses endpoint"):
        adapter.chat([message])
    assert calls == []


@pytest.mark.unit
def test_pdf_generation_error_redacts_receipt_and_bytes(deepseek_runtime, caplog) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    client = adapter.adapter._client
    client._upload_pdf = lambda content, timeout=None: "echoed-receipt"
    error = LLMAPIServerError(detail=f"provider echoed {PDF_A!r} echoed-receipt")
    client.create = lambda **kwargs: (_ for _ in ()).throw(error)
    with pytest.raises(LLMAPIServerError) as caught:
        adapter.chat([UserMessage("Read", files=[DocumentPart(data=PDF_A, media_type="application/pdf")])])
    assert "DeepSeek Responses request failed" in caught.value.detail
    assert "echoed-receipt" not in str(caught.value)
    assert "private" not in caplog.text


@pytest.mark.unit
def test_upload_failure_stops_generation_and_sanitizes_previous_receipt(deepseek_runtime) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    client = adapter.adapter._client
    def upload(content: bytes, timeout: float | None = None) -> str:
        calls.append(("upload", content))
        if content == PDF_B:
            raise LLMAPIClientError(detail="failed receipt-1")
        return "receipt-1"
    client._upload_pdf = upload
    message = UserMessage("Read", files=[
        DocumentPart(data=PDF_A, media_type="application/pdf"),
        DocumentPart(data=PDF_B, media_type="application/pdf"),
    ])
    with pytest.raises(LLMAPIClientError) as caught:
        adapter.chat([message])
    assert caught.value.detail == "DeepSeek Responses request failed"
    assert "receipt-1" not in str(caught.value)
    assert all(kind != "generate" for kind, _ in calls)


@pytest.mark.unit
def test_failed_pdf_stream_redacts_provider_echo_and_skips_on_done(deepseek_runtime) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    client = adapter.adapter._client
    client._upload_pdf = lambda content, timeout=None: "receipt-1"
    client.stream = lambda **kwargs: iter([SSEEvent(
        event="response.failed",
        data={"type": "response.failed", "response": {
            "object": "response",
            "error": {"message": f"provider echoed receipt-1 and {PDF_A!r}"},
        }},
    )])
    completed = []
    stream = adapter.stream_chat(
        [UserMessage("Read", files=[DocumentPart(data=PDF_A, media_type="application/pdf")])],
        on_done=completed.append,
    )
    with pytest.raises(LLMAPIClientError) as caught:
        list(stream)
    assert "DeepSeek Responses request failed" in caught.value.detail
    assert "receipt-1" not in str(caught.value)
    assert completed == []


@pytest.mark.unit
def test_successful_sse_scrubbing_preserves_output_and_only_cleans_metadata() -> None:
    receipt = "receipt-private"
    event = SSEEvent(
        event="response.completed",
        data={
            "type": "response.completed",
            "response": {
                "metadata": {"file_id": receipt, "note": receipt},
                "output": [{
                    "type": "function_call",
                    "arguments": json.dumps({"file_id": receipt, "file_data": PDF_A.decode()}),
                }],
                "output_text": f"visible {receipt} {PDF_A.decode()}",
            },
        },
    )
    cleaned = adapter_module.DeepSeekAdapter._redact_pdf_event(event, (receipt,))
    response = cleaned.data["response"]
    assert response["metadata"] == {"note": "[redacted]"}
    assert response["output"] == event.data["response"]["output"]
    assert response["output_text"] == event.data["response"]["output_text"]


@pytest.mark.unit
def test_pdf_stream_close_closes_native_sync_source_immediately(deepseek_runtime) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)

    class Source:
        closed = False

        def __iter__(self):
            yield SSEEvent(
                event="response.output_text.delta",
                data={"type": "response.output_text.delta", "delta": "visible"},
            )
            yield SSEEvent(
                event="response.completed",
                data={"type": "response.completed", "response": _response()},
            )

        def close(self) -> None:
            self.closed = True

    source = Source()
    adapter.adapter._client.stream = lambda **kwargs: source
    completed = []
    stream = adapter.stream_chat(
        [UserMessage("Read", files=[DocumentPart(data=PDF_A, media_type="application/pdf")])],
        on_done=completed.append,
    )
    assert next(stream) == "visible"
    stream.close()
    assert source.closed is True
    assert completed == []


@pytest.mark.unit
def test_pdf_stream_close_closes_native_async_source_immediately(deepseek_runtime) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_async(adapter, calls)

    class Source:
        closed = False

        def __aiter__(self):
            self._events = iter([
                SSEEvent(
                    event="response.output_text.delta",
                    data={"type": "response.output_text.delta", "delta": "visible"},
                ),
                SSEEvent(
                    event="response.completed",
                    data={"type": "response.completed", "response": _response()},
                ),
            ])
            return self

        async def __anext__(self):
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

        async def aclose(self) -> None:
            self.closed = True

    source = Source()
    adapter.adapter._async_client.stream = lambda **kwargs: source

    async def run() -> None:
        completed = []
        stream = adapter.astream_chat(
            [UserMessage("Read", files=[DocumentPart(data=PDF_A, media_type="application/pdf")])],
            on_done=completed.append,
        )
        assert await stream.__anext__() == "visible"
        await stream.aclose()
        assert source.closed is True
        assert completed == []

    asyncio.run(run())


@pytest.mark.unit
def test_stream_is_lazy_before_consumption_and_early_close_skips_generation(deepseek_runtime) -> None:
    adapter = _adapter()
    calls: list[tuple[str, Any]] = []
    _configure_sync(adapter, calls)
    stream = adapter.stream_chat([UserMessage("Read", files=[DocumentPart(data=PDF_A, media_type="application/pdf")])])
    assert calls == []
    stream.close()
    assert calls == []
