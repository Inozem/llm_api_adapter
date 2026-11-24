"""Structured output contract for the real native Mistral Responses adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping
import ipaddress
from pathlib import Path
import socket
import sys
from typing import Any

import pytest
from pydantic import BaseModel


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (str(PACKAGE_ROOT / "src"), str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter_mistral.clients.async_client as async_client_module
from llm_api_adapter.errors.llm_api_error import JSONSchemaError
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
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools import ToolSpec
from llm_api_adapter_mistral.adapter import MistralAdapter
from llm_api_adapter_mistral.plugin import PLUGIN


MODEL = "mistral-large-4"
METHODS = ("chat", "stream_chat", "achat", "astream_chat")
PORTABLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class StructuredAnswer(BaseModel):
    answer: str

    model_config = {"extra": "forbid"}


class RecordingTransport(SyncTransport):
    """Typed transport seam that captures actual native adapter submissions."""

    def __init__(self, response: dict[str, Any], events: tuple[SSEEvent, ...]) -> None:
        self.response = response
        self.events = events
        self.requests: list[TransportRequest] = []
        self.stream_requests: list[TransportRequest] = []

    def post_multipart(
        self,
        request: TransportRequest,
        form: MultipartForm,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        raise AssertionError("structured-output tests do not submit multipart requests")

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        self.requests.append(request)
        return JSONResponse(self.response)

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> Iterator[SSEEvent]:
        self.stream_requests.append(request)
        return iter(self.events)


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
        host: Any,
        *args: Any,
        **kwargs: Any,
    ) -> list[tuple[int, int, int, str, tuple[Any, ...]]]:
        if not is_loopback(host):
            raise AssertionError("structured-output test attempted external DNS")
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(connection: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and address and not is_loopback(address[0]):
            raise AssertionError("structured-output test attempted external network")
        original_connect(connection, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def _response(
    content: str | None = '{"answer":"ready"}',
    *,
    status: str = "completed",
    refusal: str | None = None,
) -> dict[str, Any]:
    parts = (
        [{"type": "refusal", "refusal": refusal}]
        if refusal is not None
        else ([{"type": "output_text", "text": content}] if content is not None else [])
    )
    result: dict[str, Any] = {
        "object": "response",
        "id": "resp_structured",
        "model": MODEL,
        "status": status,
        "output": [{"type": "message", "role": "assistant", "status": "completed", "content": parts}],
    }
    if status == "incomplete":
        result["incomplete_details"] = {"reason": "max_output_tokens"}
    return result


def _events(response: dict[str, Any], content: str | None = '{"answer":"ready"}') -> tuple[SSEEvent, ...]:
    values: list[SSEEvent] = []
    if content:
        values.append(SSEEvent(event="response.output_text.delta", data={"delta": content}))
    terminal = "response.incomplete" if response["status"] == "incomplete" else "response.completed"
    values.append(SSEEvent(event=terminal, data={"response": response}))
    return tuple(values)


def _make_adapter() -> MistralAdapter:
    return MistralAdapter(api_key="synthetic-mistral-key", model=MODEL, company="mistral")


def _run(
    adapter: MistralAdapter,
    method: str,
    kwargs: dict[str, Any],
    *,
    completion_capture: list[ChatResponse] | None = None,
    expected_visible: str | None = '{"answer":"ready"}',
) -> ChatResponse:
    completed = completion_capture if completion_capture is not None else []
    if method == "chat":
        return adapter.chat([{"role": "user", "content": "Return the answer."}], **kwargs)
    if method == "stream_chat":
        visible = list(adapter.stream_chat(
            [{"role": "user", "content": "Return the answer."}],
            on_done=completed.append,
            **kwargs,
        ))
        assert len(completed) == 1
        assert visible == ([expected_visible] if expected_visible else [])
        return completed[0]
    if method == "achat":
        return asyncio.run(adapter.achat(
            [{"role": "user", "content": "Return the answer."}], **kwargs
        ))

    async def collect() -> ChatResponse:
        visible = [item async for item in adapter.astream_chat(
            [{"role": "user", "content": "Return the answer."}],
            on_done=completed.append,
            **kwargs,
        )]
        assert len(completed) == 1
        assert visible == ([expected_visible] if expected_visible else [])
        return completed[0]

    return asyncio.run(collect())


def _install_async(
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, Any],
    events: tuple[SSEEvent, ...],
) -> dict[str, list[TransportRequest]]:
    calls: dict[str, list[TransportRequest]] = {"chat": [], "stream": []}

    async def async_request(
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        payload: Any = None,
        timeout: float | None = None,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> dict[str, Any]:
        calls["chat"].append(TransportRequest(
            url=url, headers=headers or {}, payload=payload, timeout=timeout
        ))
        return response

    def async_stream_request(
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        payload: Any = None,
        timeout: float | None = None,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> AsyncIterator[SSEEvent]:
        calls["stream"].append(TransportRequest(
            url=url, headers=headers or {}, payload=payload, timeout=timeout
        ))

        async def source() -> AsyncIterator[SSEEvent]:
            for event in events:
                yield event

        return source()

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream_request)
    return calls


def _request_payload(
    method: str,
    recorder: RecordingTransport,
    async_calls: dict[str, list[TransportRequest]],
) -> dict[str, Any]:
    if method == "chat":
        return recorder.requests[0].payload
    if method == "stream_chat":
        return recorder.stream_requests[0].payload
    bucket = "chat" if method == "achat" else "stream"
    return async_calls[bucket][0].payload


@pytest.mark.integration
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("schema_kind", ("portable", "response_model"))
def test_real_native_methods_send_strict_schema_and_parse_results(
    mistral_registry: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    schema_kind: str,
) -> None:
    response = _response()
    events = _events(response)
    adapter = _make_adapter()
    recorder = RecordingTransport(response, events)
    adapter._sync_transport = recorder
    async_calls = _install_async(monkeypatch, response, events)
    kwargs: dict[str, Any] = (
        {"json_schema": PORTABLE_SCHEMA}
        if schema_kind == "portable"
        else {"response_model": StructuredAnswer}
    )

    result = _run(adapter, method, kwargs)
    format_spec = _request_payload(method, recorder, async_calls)["text"]["format"]

    assert format_spec["type"] == "json_schema"
    assert format_spec["strict"] is True
    assert format_spec["name"] == "response"
    if schema_kind == "portable":
        assert format_spec["schema"] == PORTABLE_SCHEMA
        assert result.parsed_json == {"answer": "ready"}
        assert result.parsed_model is None
    else:
        assert format_spec["schema"]["properties"] == {
            "answer": {"title": "Answer", "type": "string"}
        }
        assert format_spec["schema"]["required"] == ["answer"]
        assert format_spec["schema"]["additionalProperties"] is False
        assert result.parsed_json == {"answer": "ready"}
        assert result.parsed_model == StructuredAnswer(answer="ready")


@pytest.mark.integration
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("response", "error_match"),
    (
        (_response(content="not valid JSON"), "not valid JSON"),
        (_response(content='{"answer":42}'), "Pydantic validation"),
    ),
)
def test_invalid_completed_structured_output_raises_public_schema_error(
    mistral_registry: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    response: dict[str, Any],
    error_match: str,
) -> None:
    events = _events(response, response["output"][0]["content"][0].get("text"))
    adapter = _make_adapter()
    adapter._sync_transport = RecordingTransport(response, events)
    _install_async(monkeypatch, response, events)
    kwargs = (
        {"json_schema": PORTABLE_SCHEMA}
        if error_match == "not valid JSON"
        else {"response_model": StructuredAnswer}
    )

    completed: list[ChatResponse] = []
    with pytest.raises(JSONSchemaError, match=error_match):
        _run(adapter, method, kwargs, completion_capture=completed)
    if method in {"stream_chat", "astream_chat"}:
        assert completed == []


@pytest.mark.integration
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("outcome", ("refusal", "incomplete"))
def test_refusal_and_incomplete_leave_native_parsed_output_unavailable(
    mistral_registry: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    outcome: str,
) -> None:
    partial_text = '{"answer":'
    response = (
        _response(content=None, refusal="Cannot provide that answer.")
        if outcome == "refusal"
        else _response(content=partial_text, status="incomplete")
    )
    events = _events(response, content=None if outcome == "refusal" else partial_text)
    adapter = _make_adapter()
    adapter._sync_transport = RecordingTransport(response, events)
    _install_async(monkeypatch, response, events)

    completed: list[ChatResponse] = []
    result = _run(
        adapter,
        method,
        {"response_model": StructuredAnswer},
        completion_capture=completed,
        expected_visible=None if outcome == "refusal" else partial_text,
    )
    assert result.parsed_json is None
    assert result.parsed_model is None
    if method in {"stream_chat", "astream_chat"}:
        assert len(completed) == 1
    if outcome == "refusal":
        assert result.finish_reason == "content_filter"
        assert result.refusal == "Cannot provide that answer."
    else:
        assert result.finish_reason == "length"
        assert result.incomplete_reason == "length"
        assert result.content == partial_text


@pytest.mark.integration
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    (
        ({"json_schema": PORTABLE_SCHEMA, "response_model": StructuredAnswer}, "cannot be used together"),
        ({"json_schema": PORTABLE_SCHEMA, "tools": [ToolSpec(name="lookup", json_schema={"type": "object"})]}, "cannot be used together"),
        ({"response_model": StructuredAnswer, "tools": [ToolSpec(name="lookup", json_schema={"type": "object"})]}, "cannot be used together"),
        ({"json_schema": {"type": "string"}}, "root object schema"),
        ({"response_model": object}, "Pydantic BaseModel subclass"),
    ),
)
def test_incompatible_structured_options_fail_before_transport(
    mistral_registry: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    kwargs: dict[str, Any],
    error_match: str,
) -> None:
    response = _response()
    events = _events(response)
    adapter = _make_adapter()
    recorder = RecordingTransport(response, events)
    adapter._sync_transport = recorder
    async_calls = _install_async(monkeypatch, response, events)

    with pytest.raises(JSONSchemaError, match=error_match):
        _run(adapter, method, kwargs)

    assert recorder.requests == []
    assert recorder.stream_requests == []
    assert async_calls == {"chat": [], "stream": []}
