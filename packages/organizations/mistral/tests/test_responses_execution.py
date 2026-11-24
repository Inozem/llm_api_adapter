"""Exercise the discovered native Mistral adapter over both sync transports."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from importlib.metadata import EntryPoint
import ipaddress
import json
from pathlib import Path
import socket
import sys
from typing import Any, Optional

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (str(PACKAGE_ROOT / "src"), str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

from llm_api_adapter.adapters.base_adapter import LLMAdapterBase
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec, resolve_model_spec
from llm_api_adapter.llms.httpx_transport import HttpxSyncTransport
from llm_api_adapter.llms.requests_transport import RequestsSyncTransport
from llm_api_adapter.llms.transports import (
    HTTPErrorHandler,
    SSEEvent,
    StreamErrorHandler,
    TransportRequest,
)
from llm_api_adapter.models.messages.chat_message import AIMessage, Message, UserMessage
from llm_api_adapter.organization_registry import (
    ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP,
    OrganizationPluginDiscovery,
)
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.organization_registry as organization_registry_module
import llm_api_adapter.universal_adapter as universal_module
import llm_api_adapter.llms.requests_transport as requests_transport_module
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
import llm_api_adapter_mistral.clients.async_client as async_client_module
from llm_api_adapter_mistral.adapter import MistralAdapter
from tests.external_organization_metadata import read_project_metadata


MODEL_NAMES = (
    "mistral-large-4",
    "mistral-small-2603",
    "mistral-medium-3-5",
    "mistral-large-2512",
    "glm-5.3-flash",
    "deepseek-flash",
)
TRANSPORTS = ("requests", "httpx")
HISTORY_KINDS = ("typed", "dictionary", "mixed")
METHODS = ("chat", "stream_chat", "achat", "astream_chat")


def _completed_response(response_id: str) -> dict[str, Any]:
    return {
        "object": "response",
        "id": response_id,
        "status": "completed",
        "model": "mistral-large-4",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "native response"}],
            }
        ],
    }


def _stream_events() -> tuple[SSEEvent, ...]:
    return (
        SSEEvent(event="response.output_text.delta", data={"delta": "native response"}),
        SSEEvent(
            event="response.completed",
            data={"response": _completed_response("resp_native_stream")},
        ),
    )


def _expected_history(
    kind: str,
) -> tuple[list[Message | dict[str, Any]], list[dict[str, Any]]]:
    turns: dict[str, list[Message | dict[str, Any]]] = {
        "typed": [UserMessage("typed user"), AIMessage("typed assistant"), UserMessage("typed follow-up")],
        "dictionary": [
            {"role": "user", "content": "typed user"},
            {"role": "assistant", "content": "typed assistant"},
            {"role": "user", "content": "typed follow-up"},
        ],
        "mixed": [
            UserMessage("typed user"),
            {"role": "assistant", "content": "typed assistant"},
            UserMessage("typed follow-up"),
        ],
    }
    expected: list[dict[str, Any]] = [
        {"role": "user", "content": "typed user"},
        {"role": "assistant", "content": "typed assistant"},
        {"role": "user", "content": "typed follow-up"},
    ]
    return turns[kind], expected


def _is_loopback(host: Any) -> bool:
    if host in {"localhost", b"localhost"}:
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", "ignore")
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False


@pytest.fixture(autouse=True)
def deny_external_network(monkeypatch: pytest.MonkeyPatch) -> None:
    original_getaddrinfo = socket.getaddrinfo
    original_connect = socket.socket.connect

    def guard_getaddrinfo(
        host: Any,
        *args: Any,
        **kwargs: Any,
    ) -> list[tuple[int, int, int, str, tuple[Any, ...]]]:
        if not _is_loopback(host):
            raise AssertionError("Mistral execution test attempted external DNS")
        return original_getaddrinfo(host, *args, **kwargs)

    def guard_connect(connection: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and address and not _is_loopback(address[0]):
            raise AssertionError("Mistral execution test attempted external network")
        original_connect(connection, address)

    monkeypatch.setattr(socket, "getaddrinfo", guard_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guard_connect)


@pytest.fixture
def discovered_mistral_runtime(monkeypatch: pytest.MonkeyPatch) -> RegistrySpec:
    metadata = read_project_metadata(PACKAGE_ROOT / "pyproject.toml")
    entry_point_value = metadata["entry-points"][
        ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP
    ]["mistral"]
    entry_point = EntryPoint(
        name="mistral",
        value=entry_point_value,
        group=ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP,
    )
    service_registry = ServiceProviderRegistry()
    model_registry = RegistrySpec()
    discovery = OrganizationPluginDiscovery()

    def controlled_entry_points(*, group: str) -> tuple[EntryPoint, ...]:
        assert group == ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP
        return (entry_point,)

    monkeypatch.setattr(
        organization_registry_module, "entry_points", controlled_entry_points
    )
    discovery.discover(service_registry, model_registry)
    assert discovery.failures == ()
    assert service_registry.service_providers == ("mistral",)
    assert "mistral" in model_registry.organizations
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", model_registry)
    monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", service_registry)
    monkeypatch.setattr(universal_module, "ORGANIZATION_PLUGIN_DISCOVERY", discovery)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", model_registry)
    return model_registry


class FakeHTTPResponse:
    payload: dict[str, Any]
    lines: tuple[str, ...]
    status_code: int
    closed: bool
    streaming: bool

    def __init__(self, *, streaming: bool) -> None:
        self.payload = _completed_response("resp_native_chat")
        self.lines = tuple(
            line
            for event in _stream_events()
            for line in (
                f"event: {event.event}",
                f"data: {json.dumps(event.data)}",
                "",
            )
        ) + ("data: [DONE]", "")
        self.status_code = 200
        self.closed = False
        self.streaming = streaming

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload

    def iter_lines(self, decode_unicode: bool = True) -> Iterator[str]:
        yield from self.lines

    def close(self) -> None:
        self.closed = True


class FakeHTTPXRequest:
    method: str
    url: str
    headers: dict[str, str]
    json_payload: dict[str, Any]
    timeout: Optional[float]

    def __init__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        json_payload: dict[str, Any],
        timeout: Optional[float],
    ) -> None:
        self.method = method
        self.url = url
        self.headers = headers
        self.json_payload = json_payload
        self.timeout = timeout


class FakeHTTPXClient:
    calls: list[tuple[str, dict[str, str], dict[str, Any], Optional[float], bool]]
    closed: bool

    def __init__(
        self,
        calls: list[
            tuple[str, dict[str, str], dict[str, Any], Optional[float], bool]
        ],
    ) -> None:
        self.calls = calls
        self.closed = False

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: Optional[float],
    ) -> FakeHTTPResponse:
        self.calls.append((url, headers, json, timeout, False))
        return FakeHTTPResponse(streaming=False)

    def build_request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: Optional[float],
    ) -> FakeHTTPXRequest:
        return FakeHTTPXRequest(method, url, headers, json, timeout)

    def send(self, request: FakeHTTPXRequest, *, stream: bool) -> FakeHTTPResponse:
        self.calls.append(
            (request.url, request.headers, request.json_payload, request.timeout, stream)
        )
        return FakeHTTPResponse(streaming=stream)

    def close(self) -> None:
        self.closed = True


@pytest.mark.unit
@pytest.mark.parametrize("model", MODEL_NAMES)
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize("history_kind", HISTORY_KINDS)
@pytest.mark.parametrize("method", METHODS)
def test_discovered_native_adapter_executes_responses_for_all_profiles(
    monkeypatch: pytest.MonkeyPatch,
    discovered_mistral_runtime: RegistrySpec,
    model: str,
    transport: str,
    history_kind: str,
    method: str,
) -> None:
    calls: list[tuple[str, dict[str, str], dict[str, Any], Optional[float], bool]] = []
    async_calls: list[tuple[str, dict[str, str], dict[str, Any], Optional[float]]] = []

    def fake_requests_post(
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: Optional[float],
        stream: bool = False,
    ) -> FakeHTTPResponse:
        calls.append((url, headers, json, timeout, stream))
        return FakeHTTPResponse(streaming=stream)

    monkeypatch.setattr(requests_transport_module.requests, "post", fake_requests_post)
    import httpx

    def create_fake_httpx_client() -> FakeHTTPXClient:
        return FakeHTTPXClient(calls)

    monkeypatch.setattr(httpx, "Client", create_fake_httpx_client)

    async def fake_async_request(
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: Optional[float],
        http_error_handler: Optional[HTTPErrorHandler] = None,
    ) -> dict[str, Any]:
        async_calls.append((url, headers, payload, timeout))
        return _completed_response("resp_native_achat")

    def fake_async_stream_request(
        url: str,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: Optional[float],
        http_error_handler: Optional[HTTPErrorHandler] = None,
        stream_error_handler: Optional[StreamErrorHandler] = None,
    ) -> AsyncIterator[SSEEvent]:
        async_calls.append((url, headers, payload, timeout))

        async def events() -> AsyncIterator[SSEEvent]:
            for event in _stream_events():
                yield event

        return events()

    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", fake_async_stream_request)
    organization = {"glm-5.3-flash": "zai", "deepseek-flash": "deepseek"}.get(model, "mistral")
    facade = UniversalLLMAPIAdapter(
        organization=organization,
        service_provider="mistral",
        model=model,
        api_key="synthetic-mistral-key",
        transport=transport,
    )
    assert isinstance(facade.adapter, LLMAdapterBase)
    assert facade.organization == organization
    assert facade.service_provider == "mistral"
    assert isinstance(facade.adapter, MistralAdapter)
    assert facade.adapter.service_provider == "mistral"
    assert facade.adapter.model_spec is resolve_model_spec(
        discovered_mistral_runtime, "mistral", model
    )
    expected_transport_type = (
        RequestsSyncTransport if transport == "requests" else HttpxSyncTransport
    )
    assert isinstance(facade.adapter._sync_transport, expected_transport_type)
    turns, expected_history = _expected_history(history_kind)
    execution_options: dict[str, Any] = {
        "max_tokens": 23,
        "temperature": 1.25,
        "top_p": 0.8,
        "timeout_s": 7.0,
    }

    if method == "chat":
        response = facade.chat(turns, **execution_options)
        assert response.content == "native response"
        assert len(calls) == 1 and len(async_calls) == 0
        url, headers, payload, timeout, is_stream = calls[0]
        assert not is_stream
    elif method == "stream_chat":
        assert "".join(facade.stream_chat(turns, **execution_options)) == "native response"
        assert len(calls) == 1 and len(async_calls) == 0
        url, headers, payload, timeout, is_stream = calls[0]
        assert is_stream
    elif method == "achat":
        response = asyncio.run(facade.achat(turns, **execution_options))
        assert response.content == "native response"
        assert len(calls) == 0 and len(async_calls) == 1
        url, headers, payload, timeout = async_calls[0]
    else:
        async def collect() -> str:
            return "".join(
                [part async for part in facade.astream_chat(turns, **execution_options)]
            )

        assert asyncio.run(collect()) == "native response"
        assert len(calls) == 0 and len(async_calls) == 1
        url, headers, payload, timeout = async_calls[0]

    expected_spec = resolve_model_spec(discovered_mistral_runtime, "mistral", model)
    assert expected_spec is not None
    expected_wire_model = expected_spec.provider_model or model
    assert facade.adapter.model_spec is expected_spec
    assert url == "https://api.mistral.ai/v1/responses"
    assert headers["Authorization"] == "Bearer synthetic-mistral-key"
    assert payload["model"] == expected_wire_model
    assert payload["max_output_tokens"] == 23
    assert payload["temperature"] == 1.25
    assert payload["top_p"] == 0.8
    assert payload["input"] == expected_history
    assert timeout == 7.0
