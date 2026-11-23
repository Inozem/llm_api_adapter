"""Same-adapter Mistral Responses/OCR invocation isolation tests."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Iterator, Mapping
import ipaddress
from pathlib import Path
import socket
import sys
from typing import Any

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (str(PACKAGE_ROOT / "src"), str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter_mistral.clients.async_client as async_client_module
from llm_api_adapter.errors.llm_api_error import LLMAPIRateLimitError
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.transports import (
    HTTPErrorHandler,
    JSONResponse,
    MultipartForm,
    SSEEvent,
    StreamErrorHandler,
    SyncTransport,
    TransportRequest,
)
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.models.responses.chat_response import ChatResponse, CostLineItem
from llm_api_adapter_mistral.adapter import MistralAdapter
from llm_api_adapter_mistral.documents import PreparedOCROperation
from llm_api_adapter_mistral.plugin import PLUGIN


MODEL = "mistral-large-4"
RESPONSES_URL = "https://api.mistral.ai/v1/responses"
SECOND_RESPONSES_URL = "https://mistral.example/v1/responses/session-b"
OCR_URL = "https://api.mistral.ai/v1/ocr"
USAGE = {
    "input_tokens": 10,
    "output_tokens": 2,
    "total_tokens": 12,
    "input_tokens_details": {"cached_tokens": 0},
}


@pytest.fixture
def mistral_registry(monkeypatch: pytest.MonkeyPatch) -> RegistrySpec:
    registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert registry.register_organization_metadata(PLUGIN.model_metadata)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)
    return registry


@pytest.fixture(autouse=True)
def deny_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect

    def is_loopback(host: Any) -> bool:
        if host in {"localhost", b"localhost"}:
            return True
        if isinstance(host, bytes):
            host = host.decode("ascii", "ignore")
        try:
            return ipaddress.ip_address(host).is_loopback
        except (ValueError, TypeError):
            return False

    def guarded_getaddrinfo(
        host: Any, *args: Any, **kwargs: Any
    ) -> list[tuple[Any, ...]]:
        if not is_loopback(host):
            raise AssertionError("isolation test attempted external DNS")
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(connection: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and address and not is_loopback(address[0]):
            raise AssertionError("isolation test attempted external network")
        original_connect(connection, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def document_message(label: str) -> list[UserMessage]:
    return [UserMessage(
        f"Summarize {label}.",
        files=[DocumentPart(data=f"PDF:{label}".encode(), media_type="application/pdf")],
    )]


def _ocr_label(payload: Mapping[str, Any]) -> str:
    document = payload.get("document")
    source = document.get("document_url") if isinstance(document, Mapping) else None
    assert isinstance(source, str)
    return next(
        label for label in ("A", "B", "FAIL", "CLOSE", "CANCEL")
        if base64.b64encode(f"PDF:{label}".encode()).decode() in source
    )


def _prepared_label(payload: Mapping[str, Any]) -> str:
    serialized_input = str(payload.get("input"))
    return next(
        (label for label in ("A", "B", "FAIL", "CLOSE", "CANCEL")
         if f"OCR:{label}" in serialized_input),
        "plain",
    )


def _completed_response(label: str, effort: str) -> dict[str, Any]:
    return {
        "object": "response",
        "id": f"response-{label}",
        "model": MODEL,
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": effort}]},
            {"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": f"visible:{label}"}]},
        ],
        "usage": dict(USAGE),
    }


def _expected_total(adapter: MistralAdapter, registry: RegistrySpec, pages: int) -> float:
    assert adapter.model_spec is not None
    tier = adapter.model_spec.pricing_tiers.tier_for_prompt_tokens(10)
    meter = registry.organizations["mistral"].metered_operations["ocr"]
    return 10 * tier.in_per_token + 2 * tier.out_per_token + pages * meter.rate


class IsolationTransport(SyncTransport):
    """Return controlled OCR and Responses values through native sync seams."""

    def __init__(self) -> None:
        self.requests: list[TransportRequest] = []
        self.closed: list[str] = []

    def _response(self, request: TransportRequest) -> dict[str, Any]:
        label = _prepared_label(request.payload)
        reasoning = request.payload.get("reasoning", {})
        effort = reasoning.get("effort", "none") if isinstance(reasoning, Mapping) else "none"
        return _completed_response(label, str(effort))

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        self.requests.append(request)
        if request.url == OCR_URL:
            label = _ocr_label(request.payload)
            pages = {"A": 2, "B": 5, "FAIL": 3, "CLOSE": 7, "CANCEL": 11}[label]
            return JSONResponse({
                "pages": [{"markdown": f"OCR:{label}"}],
                "usage_info": {"pages_processed": pages},
            })
        return JSONResponse(self._response(request))

    def post_multipart(
        self,
        request: TransportRequest,
        form: MultipartForm,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        raise AssertionError("Mistral OCR and Responses use their existing JSON seam")

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> Iterator[SSEEvent]:
        self.requests.append(request)
        label = _prepared_label(request.payload)
        reasoning = request.payload.get("reasoning", {})
        effort = reasoning.get("effort", "none") if isinstance(reasoning, Mapping) else "none"

        def events() -> Iterator[SSEEvent]:
            try:
                yield SSEEvent(event="response.output_text.delta", data={"delta": f"visible:{label}"})
                if label == "FAIL":
                    yield SSEEvent(event="response.failed", data={
                        "response": {"error": {"type": "rate_limit_error", "message": "controlled failure"}}
                    })
                else:
                    yield SSEEvent(event="response.completed", data={
                        "response": _completed_response(label, str(effort))
                    })
            finally:
                self.closed.append(label)

        return events()


@pytest.mark.integration
def test_interleaved_streams_keep_ocr_finalizer_and_native_request_state_local(
    mistral_registry: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MistralAdapter(api_key="isolated-key-a", model=MODEL, company="mistral")
    transport = IsolationTransport()
    adapter._sync_transport = transport
    completed: list[tuple[str, ChatResponse]] = []
    finalized: list[tuple[ChatResponse, tuple[PreparedOCROperation, ...]]] = []
    inherited_apply_ocr_costs = adapter._apply_ocr_costs

    def observe_apply_ocr_costs(
        response: ChatResponse,
        operations: tuple[PreparedOCROperation, ...],
    ) -> None:
        finalized.append((response, operations))
        assert adapter.model_spec is not None
        tier = adapter.model_spec.pricing_tiers.tier_for_prompt_tokens(10)
        assert response.cost_input is not None
        assert response.cost_output is not None
        assert response.cost_input == pytest.approx(10 * tier.in_per_token)
        assert response.cost_output == pytest.approx(2 * tier.out_per_token)
        assert response.cost_breakdown in (None, [])
        inherited_apply_ocr_costs(response, operations)

    monkeypatch.setattr(adapter, "_apply_ocr_costs", observe_apply_ocr_costs)

    def record_done(label: str, pages: int, response: ChatResponse) -> None:
        meter = mistral_registry.organizations["mistral"].metered_operations["ocr"]
        assert response.cost_total == pytest.approx(
            _expected_total(adapter, mistral_registry, pages)
        )
        assert response.cost_breakdown == [CostLineItem(
            operation=meter.name, model=meter.model, unit=meter.unit,
            quantity=pages, rate=meter.rate, currency=meter.currency,
        )]
        completed.append((label, response))

    def on_done_a(response: ChatResponse) -> None:
        record_done("A", 2, response)

    def on_done_b(response: ChatResponse) -> None:
        record_done("B", 5, response)

    stream_a = adapter.stream_chat(
        document_message("A"), reasoning_level="low", capture_reasoning=True,
        on_done=on_done_a,
    )
    assert next(stream_a) == "visible:A"
    adapter.api_key = "isolated-key-b"
    adapter.endpoint = SECOND_RESPONSES_URL
    stream_b = adapter.stream_chat(
        document_message("B"), reasoning_level="high", capture_reasoning=True,
        on_done=on_done_b,
    )
    assert next(stream_b) == "visible:B"

    list(stream_a)
    list(stream_b)

    response_requests = [request for request in transport.requests if request.url != OCR_URL]
    assert [request.url for request in response_requests] == [RESPONSES_URL, SECOND_RESPONSES_URL]
    assert [request.headers["Authorization"] for request in response_requests] == [
        "Bearer isolated-key-a", "Bearer isolated-key-b"
    ]
    assert [request.headers["Authorization"] for request in transport.requests if request.url == OCR_URL] == [
        "Bearer isolated-key-a", "Bearer isolated-key-b"
    ]
    assert [request.payload["reasoning"]["effort"] for request in response_requests] == ["low", "high"]
    assert ["OCR:A" in str(request.payload["input"]) for request in response_requests] == [True, False]
    assert ["OCR:B" in str(request.payload["input"]) for request in response_requests] == [False, True]
    assert [label for label, _ in completed] == ["A", "B"]
    assert [response.reasoning_events[0].text for _, response in completed] == ["low", "high"]
    assert len(finalized) == 2
    assert finalized[0][0] is completed[0][1]
    assert finalized[1][0] is completed[1][1]
    assert finalized[0][1] is not finalized[1][1]
    assert [len(operations) for _, operations in finalized] == [1, 1]
    assert all(isinstance(operations[0], PreparedOCROperation) for _, operations in finalized)
    assert [operations[0].pages_processed for _, operations in finalized] == [2, 5]
    meter = mistral_registry.organizations["mistral"].metered_operations["ocr"]
    for pages, (_, response) in zip((2, 5), completed):
        assert response.usage is not None
        assert response.cost_input is not None
        assert response.cost_output is not None
        assert response.cost_total == pytest.approx(_expected_total(adapter, mistral_registry, pages))
        assert response.cost_breakdown == [CostLineItem(
            operation=meter.name, model=meter.model, unit=meter.unit,
            quantity=pages, rate=meter.rate, currency=meter.currency,
        )]
    assert [label for label in transport.closed if label in {"A", "B"}] == ["A", "B"]


@pytest.mark.integration
def test_sync_failure_close_and_later_no_pdf_do_not_reuse_ocr_state(
    mistral_registry: RegistrySpec,
) -> None:
    adapter = MistralAdapter(api_key="isolated-key", model=MODEL, company="mistral")
    transport = IsolationTransport()
    adapter._sync_transport = transport
    completed: list[ChatResponse] = []

    with pytest.raises(LLMAPIRateLimitError):
        list(adapter.stream_chat(document_message("FAIL"), on_done=completed.append))
    assert completed == []
    assert "FAIL" in transport.closed

    closable = adapter.stream_chat(document_message("CLOSE"), on_done=completed.append)
    assert next(closable) == "visible:CLOSE"
    closable.close()
    assert completed == []
    assert "CLOSE" in transport.closed

    plain = adapter.chat([UserMessage("No PDF in this request.")])
    assert plain.cost_breakdown in (None, [])
    assert plain.cost_input is not None, (plain.usage, plain.cost_output)
    assert plain.cost_output is not None
    assert plain.cost_total == pytest.approx(plain.cost_input + plain.cost_output)
    response_requests = [request for request in transport.requests if request.url == RESPONSES_URL]
    assert [request.url for request in response_requests] == [RESPONSES_URL, RESPONSES_URL, RESPONSES_URL]
    assert all(request.headers["Authorization"] == "Bearer isolated-key" for request in response_requests)


@pytest.mark.integration
def test_overlapping_async_calls_and_cancelled_stream_keep_ocr_local(
    mistral_registry: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = MistralAdapter(api_key="isolated-async-key", model=MODEL, company="mistral")
    response_requests: list[TransportRequest] = []
    ocr_requests: list[TransportRequest] = []
    stream_requests: list[TransportRequest] = []
    first_response_started = asyncio.Event()
    both_responses_started = asyncio.Event()
    started_count = 0
    pages = {"A": 2, "B": 5, "CANCEL": 11}

    async def request(
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        payload: Any = None,
        timeout: float | None = None,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> dict[str, Any]:
        nonlocal started_count
        captured = TransportRequest(url=url, headers=headers or {}, payload=payload, timeout=timeout)
        if url == OCR_URL:
            ocr_requests.append(captured)
            label = _ocr_label(payload)
            return {"pages": [{"markdown": f"OCR:{label}"}], "usage_info": {"pages_processed": pages[label]}}
        response_requests.append(captured)
        started_count += 1
        if started_count == 1:
            first_response_started.set()
        if started_count == 2:
            both_responses_started.set()
        if started_count <= 2:
            await asyncio.wait_for(both_responses_started.wait(), timeout=2)
        return _completed_response(_prepared_label(payload), str(payload.get("reasoning", {}).get("effort", "none")))

    monkeypatch.setattr(async_client_module, "async_request", request)

    async def cancel_stream() -> tuple[list[str], list[bool], list[ChatResponse]]:
        source_started = asyncio.Event()
        source_closed: list[bool] = []
        completed: list[ChatResponse] = []
        yielded: list[str] = []

        def stream_request(
            url: str,
            *,
            headers: Mapping[str, str] | None = None,
            payload: Any = None,
            timeout: float | None = None,
            http_error_handler: HTTPErrorHandler | None = None,
            stream_error_handler: StreamErrorHandler | None = None,
        ) -> AsyncIterator[SSEEvent]:
            stream_requests.append(TransportRequest(
                url=url, headers=headers or {}, payload=payload, timeout=timeout
            ))

            async def events() -> AsyncIterator[SSEEvent]:
                try:
                    yield SSEEvent(event="response.output_text.delta", data={"delta": "partial"})
                    source_started.set()
                    await asyncio.Event().wait()
                finally:
                    source_closed.append(True)

            return events()

        monkeypatch.setattr(async_client_module, "async_stream_request", stream_request)

        async def consume() -> None:
            async for text in adapter.astream_chat(
                document_message("CANCEL"), buffer_chars=64, on_done=completed.append
            ):
                yielded.append(text)

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(source_started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        assert yielded == []
        assert source_closed == [True]
        assert completed == []
        return yielded, source_closed, completed

    async def exercise() -> tuple[ChatResponse, ChatResponse, list[bool], list[ChatResponse]]:
        adapter.api_key = "isolated-async-key-a"
        adapter.endpoint = RESPONSES_URL
        first = asyncio.create_task(adapter.achat(
            document_message("A"), reasoning_level="low", capture_reasoning=True
        ))
        tasks = [first]
        try:
            await asyncio.wait_for(first_response_started.wait(), timeout=2)
            adapter.api_key = "isolated-async-key-b"
            adapter.endpoint = SECOND_RESPONSES_URL
            second = asyncio.create_task(adapter.achat(
                document_message("B"), reasoning_level="high", capture_reasoning=False
            ))
            tasks.append(second)
            result_a, result_b = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        assert result_a.reasoning_events[0].text == "low"
        assert result_b.reasoning_events == []
        for pages, result in zip((2, 5), (result_a, result_b)):
            assert result.usage is not None
            assert result.cost_input is not None
            assert result.cost_output is not None
            assert result.cost_total == pytest.approx(
                _expected_total(adapter, mistral_registry, pages)
            )
        _, source_closed, cancelled_done = await cancel_stream()
        return result_a, result_b, source_closed, cancelled_done

    result_a, result_b, source_closed, cancelled_done = asyncio.run(exercise())
    assert len(ocr_requests) == 3
    assert len(response_requests) == 2
    assert [request.headers["Authorization"] for request in response_requests] == [
        "Bearer isolated-async-key-a", "Bearer isolated-async-key-b"
    ]
    assert [request.url for request in response_requests] == [RESPONSES_URL, SECOND_RESPONSES_URL]
    assert [request.headers["Authorization"] for request in ocr_requests] == [
        "Bearer isolated-async-key-a", "Bearer isolated-async-key-b", "Bearer isolated-async-key-b"
    ]
    assert [request.payload["reasoning"]["effort"] for request in response_requests] == ["low", "high"]
    assert len(stream_requests) == 1
    assert stream_requests[0].url == SECOND_RESPONSES_URL
    assert stream_requests[0].headers["Authorization"] == "Bearer isolated-async-key-b"
    assert "OCR:CANCEL" in str(stream_requests[0].payload["input"])
    assert [len(result.cost_breakdown or []) for result in (result_a, result_b)] == [1, 1]
    assert [result.cost_breakdown[0].quantity for result in (result_a, result_b)] == [2, 5]
    assert source_closed == [True]
    assert cancelled_done == []

    plain = asyncio.run(adapter.achat([UserMessage("No PDF after cancellation.")]))
    assert plain.cost_breakdown in (None, [])
    assert plain.cost_input is not None, (plain.usage, plain.cost_output)
    assert plain.cost_output is not None
    assert plain.cost_total == pytest.approx(plain.cost_input + plain.cost_output)
    assert len(ocr_requests) == 3
