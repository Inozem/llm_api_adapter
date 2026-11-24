"""Native Responses accounting at return and stream completion boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
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
from llm_api_adapter.adapters.base_adapter import AsyncOnDone
import llm_api_adapter_mistral.clients.async_client as async_client_module
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
OCR_URL = "https://api.mistral.ai/v1/ocr"
METHODS = ("chat", "achat", "stream_chat", "astream_chat")
MESSAGES = [UserMessage("Summarize this document.", files=[
    DocumentPart(data=b"%PDF synthetic", media_type="application/pdf")
])]
USAGE = {
    "input_tokens": 10,
    "output_tokens": 2,
    "total_tokens": 12,
    "input_tokens_details": {"cached_tokens": 0},
}
MISSING_PAGES = object()


@dataclass(frozen=True)
class AccountingSnapshot:
    """Immutable response and OCR state observed at one accounting boundary."""

    parsed_json: dict[str, Any] | None
    reasoning_text: tuple[str, ...]
    usage_counts: tuple[int | None, int | None, int | None] | None
    token_costs: tuple[float | None, float | None]
    cost_total: float | None
    cost_items: tuple[CostLineItem, ...]
    operations: tuple[PreparedOCROperation, ...] = ()


def snapshot_response(
    response: ChatResponse,
    operations: tuple[PreparedOCROperation, ...] = (),
) -> AccountingSnapshot:
    usage_counts = (
        None
        if response.usage is None
        else (
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens,
        )
    )
    return AccountingSnapshot(
        parsed_json=response.parsed_json,
        reasoning_text=tuple(event.text for event in response.reasoning_events),
        usage_counts=usage_counts,
        token_costs=(response.cost_input, response.cost_output),
        cost_total=response.cost_total,
        cost_items=tuple(response.cost_breakdown or ()),
        operations=operations,
    )


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

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        if not is_loopback(host):
            raise AssertionError("accounting test attempted external DNS")
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(connection: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and address and not is_loopback(address[0]):
            raise AssertionError("accounting test attempted external network")
        original_connect(connection, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def completed_response(*, usage: Mapping[str, Any] = USAGE) -> dict[str, Any]:
    return {
        "object": "response",
        "id": "accounting-response",
        "model": MODEL,
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": [
                {"type": "summary_text", "text": "Checked the document."}
            ]},
            {"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": '{"answer":"42"}'}]},
        ],
        "usage": dict(usage),
    }


class AccountingTransport(SyncTransport):
    """Serve controlled OCR and Responses results through native transport hooks."""

    def __init__(self, *, pages: object = 2) -> None:
        self.requests: list[TransportRequest] = []
        self.pages = pages
        self.ocr_results: list[dict[str, Any]] = []
        self.response = completed_response()
        self.events = (
            SSEEvent(event="response.output_text.delta", data={"delta": '{"answer":"42"}'}),
            SSEEvent(event="response.completed", data={"response": self.response}),
        )

    def _result(self, request: TransportRequest) -> JSONResponse:
        self.requests.append(request)
        if request.url == OCR_URL:
            result: dict[str, Any] = {
                "model": "mistral-ocr-4-1",
                "pages": [{"markdown": "Synthetic extracted text."}],
            }
            if self.pages is not MISSING_PAGES:
                result["usage_info"] = {"pages_processed": self.pages}
            self.ocr_results.append(result)
            return JSONResponse(result)
        return JSONResponse(self.response)

    def post_json(
        self, request: TransportRequest, *, http_error_handler: HTTPErrorHandler | None = None
    ) -> JSONResponse:
        return self._result(request)

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> Iterator[SSEEvent]:
        self.requests.append(request)
        return iter(self.events)

    def post_multipart(
        self, request: TransportRequest, form: MultipartForm,
        *, http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        raise AssertionError("Responses accounting must use JSON")


def install_async_transport(
    monkeypatch: pytest.MonkeyPatch, transport: AccountingTransport
) -> list[TransportRequest]:
    requests: list[TransportRequest] = []

    async def request(
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        payload: Any = None,
        timeout: float | None = None,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> dict[str, Any]:
        captured = TransportRequest(url=url, headers=headers or {}, payload=payload, timeout=timeout)
        requests.append(captured)
        return transport._result(captured).json()

    def stream_request(
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        payload: Any = None,
        timeout: float | None = None,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> AsyncIterator[SSEEvent]:
        requests.append(TransportRequest(url=url, headers=headers or {}, payload=payload, timeout=timeout))

        async def events() -> AsyncIterator[SSEEvent]:
            for event in transport.events:
                yield event

        return events()

    monkeypatch.setattr(async_client_module, "async_request", request)
    monkeypatch.setattr(async_client_module, "async_stream_request", stream_request)
    return requests


async def _collect_async_stream(
    adapter: MistralAdapter, on_done: AsyncOnDone
) -> list[str]:
    return [chunk async for chunk in adapter.astream_chat(
        MESSAGES,
        json_schema={"type": "object", "properties": {"answer": {"type": "string"}},
                     "required": ["answer"], "additionalProperties": False},
        capture_reasoning=True,
        on_done=on_done,
    )]


@pytest.mark.integration
@pytest.mark.parametrize("method", METHODS)
def test_native_responses_finalizes_schema_reasoning_tokens_then_ocr_once_before_done(
    mistral_registry: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral")
    caller_pricing = adapter.pricing
    assert caller_pricing is not None
    transport = AccountingTransport()
    adapter._sync_transport = transport
    async_requests = install_async_transport(monkeypatch, transport)
    callbacks: list[AccountingSnapshot] = []
    pre_ocr_snapshots: list[AccountingSnapshot] = []

    original_apply_ocr_costs = MistralAdapter._apply_ocr_costs

    def observe_ocr_costs(
        response: ChatResponse,
        operations: tuple[PreparedOCROperation, ...],
    ) -> None:
        snapshot = snapshot_response(response, operations)
        pre_ocr_snapshots.append(snapshot)
        assert snapshot.parsed_json == {"answer": "42"}
        assert snapshot.reasoning_text == ("Checked the document.",)
        assert snapshot.usage_counts == (10, 2, 12)
        tier = caller_pricing.tier_for_prompt_tokens(10)
        assert snapshot.token_costs == pytest.approx((
            10 * tier.in_per_token, 2 * tier.out_per_token,
        ))
        assert snapshot.cost_items == ()
        assert snapshot.operations == operations
        assert len(operations) == 1
        operation = operations[0]
        assert isinstance(operation, PreparedOCROperation)
        assert operation.pages_processed == 2
        assert operation.meter is not None
        original_apply_ocr_costs(response, operations)

    monkeypatch.setattr(
        MistralAdapter, "_apply_ocr_costs", staticmethod(observe_ocr_costs)
    )

    def record_done(result: ChatResponse) -> None:
        callbacks.append(snapshot_response(result))

    kwargs = {
        "json_schema": {"type": "object", "properties": {"answer": {"type": "string"}},
                        "required": ["answer"], "additionalProperties": False},
        "capture_reasoning": True,
    }
    if method == "chat":
        result = adapter.chat(MESSAGES, **kwargs)
    elif method == "achat":
        result = asyncio.run(adapter.achat(MESSAGES, **kwargs))
    elif method == "stream_chat":
        list(adapter.stream_chat(MESSAGES, on_done=record_done, **kwargs))
        result = None
    else:
        asyncio.run(_collect_async_stream(adapter, record_done))
        result = None

    requests = async_requests if method in {"achat", "astream_chat"} else transport.requests
    assert [request.url for request in requests] == [OCR_URL, RESPONSES_URL]
    assert requests[1].payload["input"][0]["content"].startswith(
        "Summarize this document."
    )

    assert adapter.model_spec is not None
    tier = adapter.model_spec.pricing_tiers.tier_for_prompt_tokens(10)
    meter = mistral_registry.organizations["mistral"].metered_operations["ocr"]
    expected_input = 10 * tier.in_per_token
    expected_output = 2 * tier.out_per_token
    expected_total = expected_input + expected_output + 2 * meter.rate
    expected_line = CostLineItem(
        operation=meter.name, model=meter.model, unit=meter.unit,
        quantity=2, rate=meter.rate, currency=meter.currency,
    )
    assert len(pre_ocr_snapshots) == 1
    assert len(pre_ocr_snapshots[0].operations) == 1
    assert pre_ocr_snapshots[0].operations[0].meter == meter
    if result is not None:
        record_done(result)
    assert len(callbacks) == 1
    finalized = callbacks[0]
    assert finalized.parsed_json == {"answer": "42"}
    assert finalized.reasoning_text == ("Checked the document.",)
    assert finalized.usage_counts == (10, 2, 12)
    assert finalized.token_costs == pytest.approx((expected_input, expected_output))
    assert finalized.cost_total == pytest.approx(expected_total)
    assert finalized.cost_items == (expected_line,)


@pytest.mark.integration
def test_native_caller_tariff_is_used_without_repricing_ocr(
    mistral_registry: RegistrySpec,
) -> None:
    adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral")
    caller_pricing = adapter.pricing
    assert caller_pricing is not None
    baseline_tier = caller_pricing.tier_for_prompt_tokens(10)
    transport = AccountingTransport()
    adapter._sync_transport = transport
    caller_pricing.set_in_per_1m(2.0)
    caller_pricing.set_out_per_1m(3.0)

    result = adapter.chat(MESSAGES)

    meter = mistral_registry.organizations["mistral"].metered_operations["ocr"]
    another_adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral")
    assert another_adapter.pricing is not None
    assert another_adapter.pricing.tier_for_prompt_tokens(10) == baseline_tier
    assert mistral_registry.organizations["mistral"].models[MODEL].pricing_tiers.tier_for_prompt_tokens(10) == baseline_tier
    assert result.cost_input == pytest.approx(10 * 2 / 1_000_000)
    assert result.cost_output == pytest.approx(2 * 3 / 1_000_000)
    assert result.cost_breakdown is not None and len(result.cost_breakdown) == 1
    assert result.cost_breakdown[0].rate == meter.rate
    assert result.cost_total == pytest.approx(
        result.cost_input + result.cost_output + 2 * meter.rate
    )


@pytest.mark.integration
def test_native_ocr_currency_mismatch_keeps_known_costs_but_total_unknown(
    mistral_registry: RegistrySpec,
) -> None:
    adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral")
    transport = AccountingTransport()
    adapter._sync_transport = transport
    caller_pricing = adapter.pricing
    assert caller_pricing is not None
    caller_pricing.set_currency("EUR")

    result = adapter.chat(MESSAGES)

    assert result.currency == "EUR"
    assert result.cost_input is not None and result.cost_output is not None
    assert result.cost_breakdown is not None and len(result.cost_breakdown) == 1
    assert result.cost_breakdown[0].currency == mistral_registry.organizations["mistral"].metered_operations["ocr"].currency
    assert result.cost_total is None


@pytest.mark.integration
@pytest.mark.parametrize(
    ("pages", "expected_reported_usage"),
    [
        pytest.param(MISSING_PAGES, None, id="omitted-count"),
        pytest.param(True, True, id="bool-count"),
        pytest.param(-1, -1, id="negative-count"),
        pytest.param("unknown", "unknown", id="invalid-count"),
        pytest.param(None, None, id="explicit-none"),
        pytest.param(0, 0, id="reported-zero"),
    ],
)
def test_native_invalid_ocr_pages_preserve_tokens_without_complete_total(
    mistral_registry: RegistrySpec,
    pages: object,
    expected_reported_usage: object,
) -> None:
    adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral")
    transport = AccountingTransport(pages=pages)
    adapter._sync_transport = transport

    result = adapter.chat(MESSAGES)

    assert result.usage is not None and result.usage.input_tokens == 10
    assert result.cost_input is not None and result.cost_output is not None
    reported = transport.ocr_results[0]
    if pages is MISSING_PAGES:
        assert "pages_processed" not in reported.get("usage_info", {})
        assert result.cost_breakdown == []
        assert result.cost_total is None
    else:
        assert reported["usage_info"]["pages_processed"] == expected_reported_usage
    if pages == 0 and not isinstance(pages, bool):
        assert result.cost_breakdown is not None and len(result.cost_breakdown) == 1
        line = result.cost_breakdown[0]
        meter = mistral_registry.organizations["mistral"].metered_operations["ocr"]
        assert (line.operation, line.model, line.unit, line.quantity, line.rate, line.currency, line.cost) == (
            meter.name, meter.model, meter.unit, 0.0, meter.rate, meter.currency, 0.0
        )
        assert result.cost_total == pytest.approx(result.cost_input + result.cost_output)
    else:
        assert result.cost_breakdown == []
        assert result.cost_total is None
