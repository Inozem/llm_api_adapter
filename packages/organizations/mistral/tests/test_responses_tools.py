"""Native Mistral Responses tool behavior through the real adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping
import json
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
import llm_api_adapter.llms.requests_transport as requests_transport_module
import llm_api_adapter_mistral.clients.async_client as async_client_module
from llm_api_adapter.errors.llm_api_error import (
    InvalidToolArgumentsError,
    InvalidToolSchemaError,
    ToolChoiceError,
)
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
from llm_api_adapter.models.tools.tool_call import ToolCall
from llm_api_adapter.models.tools.tool_spec import ToolSpec
from llm_api_adapter_mistral.adapter import MistralAdapter
from llm_api_adapter_mistral.plugin import PLUGIN


MODEL = "mistral-large-4"
TOOL = ToolSpec(
    name="lookup",
    description="Look up a record",
    json_schema={"type": "object", "properties": {"key": {"type": "string"}}},
)
CHOICES: tuple[str | None, ...] = (None, "auto", "none", "any", "lookup")
PARALLEL: tuple[bool | None, ...] = (None, False, True)
EXECUTION_PATHS = (
    ("requests", "chat"), ("requests", "stream_chat"),
    ("httpx", "chat"), ("httpx", "stream_chat"),
    ("async", "achat"), ("async", "astream_chat"),
)
METHODS = ("chat", "stream_chat", "achat", "astream_chat")


class ToolTransport(SyncTransport):
    """Typed Core transport seam for focused request and result assertions."""

    def __init__(self, response: dict[str, Any], events: tuple[SSEEvent, ...]) -> None:
        self.response = response
        self.events = events
        self.requests: list[TransportRequest] = []

    def post_json(
        self, request: TransportRequest, *, http_error_handler: HTTPErrorHandler | None = None
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
        self.requests.append(request)
        return iter(self.events)

    def post_multipart(
        self,
        request: TransportRequest,
        form: MultipartForm,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        raise AssertionError("tool tests do not submit multipart requests")


class FakeResponse:
    def __init__(self, body: dict[str, Any], lines: tuple[str, ...] = ()) -> None:
        self.body = body
        self.lines = lines

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.body

    def iter_lines(self, decode_unicode: bool = True) -> Iterator[str]:
        yield from self.lines

    def close(self) -> None:
        return None


class FakeHTTPXRequest:
    def __init__(self, url: str, payload: dict[str, Any]) -> None:
        self.url = url
        self.payload = payload


class FakeHTTPXClient:
    def __init__(self, response: FakeResponse, captured: list[TransportRequest]) -> None:
        self.response = response
        self.captured = captured

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float | None) -> FakeResponse:
        self.captured.append(TransportRequest(url=url, headers=headers, payload=json, timeout=timeout))
        return self.response

    def build_request(
        self, method: str, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float | None
    ) -> FakeHTTPXRequest:
        self.captured.append(TransportRequest(url=url, headers=headers, payload=json, timeout=timeout))
        return FakeHTTPXRequest(url, json)

    def send(self, request: FakeHTTPXRequest, *, stream: bool) -> FakeResponse:
        return self.response

    def close(self) -> None:
        return None


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

    def guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> list[tuple[int, int, int, str, tuple[Any, ...]]]:
        if not is_loopback(host):
            raise AssertionError("tool test attempted external DNS")
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_connect(connection: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and address and not is_loopback(address[0]):
            raise AssertionError("tool test attempted external network")
        original_connect(connection, address)

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


def _tool_output(call_id: str = "call_lookup", arguments: str = '{"key":"record-7"}') -> dict[str, Any]:
    return {
        "type": "function_call", "call_id": call_id, "name": "lookup",
        "arguments": arguments, "status": "completed",
    }


def _response(output: list[dict[str, Any]]) -> dict[str, Any]:
    return {"object": "response", "id": "resp_tool", "model": MODEL,
            "status": "completed", "output": output}


def _fragment_events(output: dict[str, Any]) -> tuple[SSEEvent, ...]:
    return (
        SSEEvent(event="response.output_item.added", data={"output_index": 0, "item": {
            "id": "fc_item", "type": "function_call", "call_id": output["call_id"],
            "name": output["name"], "arguments": "",
        }}),
        SSEEvent(event="response.function_call_arguments.delta", data={
            "item_id": "fc_item", "output_index": 0, "call_id": output["call_id"],
            "name": output["name"], "delta": '{"key":',
        }),
        SSEEvent(event="response.function_call_arguments.delta", data={
            "item_id": "fc_item", "output_index": 0, "call_id": output["call_id"],
            "name": output["name"], "delta": '"record-7"}',
        }),
        SSEEvent(event="response.output_item.done", data={"output_index": 0, "item": output}),
        SSEEvent(event="response.completed", data={"response": _response([output])}),
    )


def _install_async(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, Any], events: tuple[SSEEvent, ...]
) -> list[TransportRequest]:
    captured: list[TransportRequest] = []

    async def request(
        url: str, *, headers: Mapping[str, str] | None = None, payload: Any = None,
        timeout: float | None = None, http_error_handler: HTTPErrorHandler | None = None,
    ) -> dict[str, Any]:
        captured.append(TransportRequest(url=url, headers=headers or {}, payload=payload, timeout=timeout))
        return response

    def stream_request(
        url: str, *, headers: Mapping[str, str] | None = None, payload: Any = None,
        timeout: float | None = None, http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> AsyncIterator[SSEEvent]:
        captured.append(TransportRequest(url=url, headers=headers or {}, payload=payload, timeout=timeout))

        async def source() -> AsyncIterator[SSEEvent]:
            for event in events:
                yield event

        return source()

    monkeypatch.setattr(async_client_module, "async_request", request)
    monkeypatch.setattr(async_client_module, "async_stream_request", stream_request)
    return captured


def _install_sync_transport(
    monkeypatch: pytest.MonkeyPatch,
    transport: str,
    response: dict[str, Any],
    events: tuple[SSEEvent, ...],
) -> list[TransportRequest]:
    captured: list[TransportRequest] = []
    lines = tuple(
        line
        for event in events
        for line in (f"event: {event.event}", f"data: {json.dumps(event.data)}", "")
    )
    json_response = FakeResponse(response)
    stream_response = FakeResponse(response, lines)
    if transport == "requests":
        def post(
            url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float | None,
            stream: bool = False,
        ) -> FakeResponse:
            captured.append(TransportRequest(url=url, headers=headers, payload=json, timeout=timeout))
            return stream_response if stream else json_response

        monkeypatch.setattr(requests_transport_module.requests, "post", post)
    else:
        import httpx

        def create_client() -> FakeHTTPXClient:
            return FakeHTTPXClient(stream_response, captured)

        monkeypatch.setattr(httpx, "Client", create_client)
    return captured


def _run(
    adapter: MistralAdapter, method: str, *, tools: list[ToolSpec], choice: str | None,
    parallel: bool | None, completion: list[ChatResponse],
    history: list[dict[str, Any]] | None = None,
) -> ChatResponse:
    request_history = history or [{"role": "user", "content": "Look up the record."}]
    options: dict[str, Any] = {"tools": tools, "tool_choice": choice}
    if parallel is not None:
        options["parallel_tool_calls"] = parallel
    if method == "chat":
        return adapter.chat(request_history, **options)
    if method == "stream_chat":
        list(adapter.stream_chat(request_history, on_done=completion.append, **options))
        return completion[0]
    if method == "achat":
        return asyncio.run(adapter.achat(request_history, **options))

    async def collect() -> ChatResponse:
        return await _collect_stream(adapter, request_history, completion, options)

    return asyncio.run(collect())


def _invoke_with_choice(
    adapter: MistralAdapter, method: str, choice: Any
) -> None:
    history = [{"role": "user", "content": "Look up the record."}]
    if method == "chat":
        adapter.chat(history, tools=[TOOL], tool_choice=choice)
    elif method == "stream_chat":
        _ = list(adapter.stream_chat(history, tools=[TOOL], tool_choice=choice))
    elif method == "achat":
        asyncio.run(adapter.achat(history, tools=[TOOL], tool_choice=choice))
    else:
        async def collect() -> None:
            _ = [part async for part in adapter.astream_chat(history, tools=[TOOL], tool_choice=choice)]
        asyncio.run(collect())


async def _collect_stream(
    adapter: MistralAdapter, history: list[dict[str, Any]], completion: list[ChatResponse],
    options: dict[str, Any],
) -> ChatResponse:
    _ = [chunk async for chunk in adapter.astream_chat(history, on_done=completion.append, **options)]
    return completion[0]


def _assert_choice(payload: dict[str, Any], choice: str | None, parallel: bool | None) -> None:
    expected: dict[str, Any] = {
        "auto": "auto", "none": "none", "any": "required", "lookup": {"type": "function", "name": "lookup"},
    }
    if choice is None:
        assert "tool_choice" not in payload
    else:
        assert payload["tool_choice"] == expected[choice]
    assert payload["tools"] == [{"type": "function", "name": "lookup", "description": TOOL.description,
                                  "parameters": TOOL.json_schema}]
    if parallel is None:
        assert "parallel_tool_calls" not in payload
    else:
        assert payload["parallel_tool_calls"] is parallel


@pytest.mark.integration
@pytest.mark.parametrize(("path", "method"), EXECUTION_PATHS)
@pytest.mark.parametrize("choice", CHOICES)
@pytest.mark.parametrize("parallel", PARALLEL)
def test_native_tool_choices_and_parallel_control_reach_real_responses_payload(
    mistral_registry: RegistrySpec, monkeypatch: pytest.MonkeyPatch,
    method: str, choice: str | None, path: str, parallel: bool | None,
) -> None:
    call = _tool_output()
    response = _response([call])
    events = _fragment_events(call)
    transport = path if path in {"requests", "httpx"} else "requests"
    adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral", transport=transport)
    sync_requests = (
        _install_sync_transport(monkeypatch, path, response, events)
        if path in {"requests", "httpx"} else []
    )
    if path == "async":
        recorder: ToolTransport | None = ToolTransport(response, events)
        adapter._sync_transport = recorder
    else:
        recorder = None
    async_requests = _install_async(monkeypatch, response, events)
    completed: list[ChatResponse] = []
    result = _run(adapter, method, tools=[TOOL], choice=choice, parallel=parallel, completion=completed)
    if path == "async":
        payload = async_requests[0].payload
        request = async_requests[0]
        assert sync_requests == []
        assert recorder is not None and recorder.requests == []
    else:
        payload = sync_requests[0].payload
        request = sync_requests[0]
        assert async_requests == []
    assert len(sync_requests if path != "async" else async_requests) == 1
    assert request.url == "https://api.mistral.ai/v1/responses"
    _assert_choice(payload, choice, parallel)
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls == [
        ToolCall(name="lookup", arguments={"key": "record-7"}, call_id="call_lookup")
    ]


@pytest.mark.integration
@pytest.mark.parametrize("method", ("stream_chat", "astream_chat"))
def test_native_tool_only_fragmented_stream_emits_one_typed_call_before_done(
    mistral_registry: RegistrySpec, monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    call = _tool_output()
    response = _response([call])
    events = _fragment_events(call)
    adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral")
    adapter._sync_transport = ToolTransport(response, events)
    _install_async(monkeypatch, response, events)
    callbacks: list[ToolCall] = []
    order: list[str] = []
    completed: list[ChatResponse] = []
    def callback(value: ToolCall) -> None:
        callbacks.append(value)
        order.append("tool")

    def done(value: ChatResponse) -> None:
        completed.append(value)
        order.append("done")
    history = [{"role": "user", "content": "Look up the record."}]
    if method == "stream_chat":
        visible = list(adapter.stream_chat(history, tools=[TOOL], tool_choice="any", on_tool_call=callback, on_done=done))
    elif method == "astream_chat":
        async def collect() -> list[str]:
            return [part async for part in adapter.astream_chat(history, tools=[TOOL], tool_choice="any", on_tool_call=callback, on_done=done)]
        visible = asyncio.run(collect())
    assert visible == []
    expected = ToolCall(name="lookup", arguments={"key": "record-7"}, call_id="call_lookup")
    assert callbacks == [expected]
    assert completed[0].tool_calls == [expected]
    assert order == ["tool", "done"]


@pytest.mark.integration
@pytest.mark.parametrize("method", METHODS)
def test_tool_result_history_keeps_call_id_and_arguments_association(
    mistral_registry: RegistrySpec, monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    initial_call = _tool_output("call_from_initial_turn", '{"key":"record-7"}')
    initial_response = _response([initial_call])
    adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral")
    first_recorder = ToolTransport(initial_response, _fragment_events(initial_call))
    adapter._sync_transport = first_recorder
    first_async_requests = _install_async(
        monkeypatch, initial_response, _fragment_events(initial_call)
    )
    first_completion: list[ChatResponse] = []
    first_result = _run(
        adapter, method, tools=[TOOL], choice="any", parallel=None,
        completion=first_completion,
    )
    initial_calls = first_result.tool_calls
    assert initial_calls is not None and len(initial_calls) == 1
    returned_call = initial_calls[0]
    assert returned_call.call_id == "call_from_initial_turn"
    assert returned_call.name == "lookup"
    assert returned_call.arguments == {"key": "record-7"}

    result_text = "record 7 is ready"
    final_response = _response([{
        "type": "message", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": result_text}],
    }])
    final_events = (
        SSEEvent(event="response.output_text.delta", data={"delta": result_text}),
        SSEEvent(event="response.completed", data={"response": final_response}),
    )
    second_recorder = ToolTransport(final_response, final_events)
    adapter._sync_transport = second_recorder
    second_async_requests = _install_async(monkeypatch, final_response, final_events)
    tool_result = "application lookup result"
    history = [
        {"role": "user", "content": "Look up the record."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": returned_call.call_id, "type": "function", "function": {
                "name": returned_call.name,
                "arguments": json.dumps(returned_call.arguments, separators=(",", ":")),
            },
        }]},
        {"role": "tool", "tool_call_id": returned_call.call_id, "content": tool_result},
    ]
    second_completion: list[ChatResponse] = []
    followup = _run(
        adapter, method, tools=[TOOL], choice=None, parallel=None,
        completion=second_completion, history=history,
    )
    payload = (
        second_async_requests[0].payload
        if method in {"achat", "astream_chat"}
        else second_recorder.requests[0].payload
    )
    assert payload["input"][0] == {"role": "user", "content": "Look up the record."}
    function_call, function_output = payload["input"][1:]
    assert function_call["type"] == "function_call"
    assert function_call["call_id"] == returned_call.call_id
    assert function_call["name"] == returned_call.name
    assert json.loads(function_call["arguments"]) == returned_call.arguments
    assert function_output == {
        "type": "function_call_output", "call_id": returned_call.call_id,
        "output": tool_result,
    }
    assert followup.content == result_text
    if method in {"chat", "stream_chat"}:
        assert len(first_recorder.requests) == len(second_recorder.requests) == 1
        assert first_async_requests == second_async_requests == []
    else:
        assert first_recorder.requests == second_recorder.requests == []
        assert len(first_async_requests) == len(second_async_requests) == 1


@pytest.mark.integration
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("choice", "message"),
    (("missing", "Unknown tool_choice"), (42, "Invalid tool_choice type")),
)
def test_invalid_tool_choice_keeps_public_pre_io_error(
    mistral_registry: RegistrySpec, monkeypatch: pytest.MonkeyPatch, method: str,
    choice: Any, message: str,
) -> None:
    adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral")
    recorder = ToolTransport(_response([]), ())
    adapter._sync_transport = recorder
    async_requests = _install_async(monkeypatch, _response([]), ())
    with pytest.raises(ToolChoiceError, match=message):
        _invoke_with_choice(adapter, method, choice)
    assert recorder.requests == []
    assert async_requests == []


@pytest.mark.integration
@pytest.mark.parametrize("method", METHODS)
def test_invalid_tool_schema_keeps_public_pre_io_error(
    mistral_registry: RegistrySpec, monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral")
    recorder = ToolTransport(_response([]), ())
    adapter._sync_transport = recorder
    async_requests = _install_async(monkeypatch, _response([]), ())
    with pytest.raises(InvalidToolSchemaError, match="Duplicate tool name"):
        _run(adapter, method, tools=[TOOL, TOOL], choice=None, parallel=None, completion=[])
    assert recorder.requests == []
    assert async_requests == []


@pytest.mark.integration
@pytest.mark.parametrize("method", METHODS)
def test_malformed_completed_tool_arguments_keep_public_error_and_no_stream_completion(
    mistral_registry: RegistrySpec, monkeypatch: pytest.MonkeyPatch, method: str,
) -> None:
    call = _tool_output(arguments="not-json")
    response = _response([call])
    adapter = MistralAdapter(api_key="synthetic-key", model=MODEL, company="mistral")
    recorder = ToolTransport(response, _fragment_events(call))
    adapter._sync_transport = recorder
    async_requests = _install_async(monkeypatch, response, _fragment_events(call))
    completed: list[ChatResponse] = []
    with pytest.raises(InvalidToolArgumentsError):
        _run(adapter, method, tools=[TOOL], choice="auto", parallel=None, completion=completed)
    if method in {"achat", "astream_chat"}:
        assert len(async_requests) == 1
    else:
        assert len(recorder.requests) == 1
    if method in {"stream_chat", "astream_chat"}:
        assert completed == []
