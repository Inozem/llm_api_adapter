"""Credential-free native and shared-engine contracts for Z.ai clients."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field, fields as dataclass_fields
import inspect
import ipaddress
import os
from pathlib import Path
import subprocess
import socket
import sys
import urllib.request
import warnings
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[4]
CORE_SOURCE = ROOT / "src"
PACKAGE_SOURCE = ROOT / "packages/organizations/zai/src"
for source in (str(PACKAGE_SOURCE), str(CORE_SOURCE)):
    if source not in sys.path:
        sys.path.insert(0, source)

from llm_api_adapter.errors.llm_api_error import (
    InvalidToolArgumentsError,
    LLMAPIAuthorizationError,
    LLMAPIClientError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
    LLMAPITokenLimitError,
    LLMAPIUsageLimitError,
    ToolChoiceError,
)
from llm_api_adapter.errors.config_errors import LLMConfigError, LLMReasoningLevelError
from llm_api_adapter.adapters.base_adapter import LLMAdapterBase
from llm_api_adapter.adapters.openai_compatible.adapter import OpenAICompatibleAdapter
from llm_api_adapter.llm_registry.llm_registry import Pricing, PricingTier, RegistrySpec
from llm_api_adapter.llm_registry.reasoning import (
    CANONICAL_REASONING_LEVELS,
    resolve_reasoning_level,
)
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent, SyncTransport, TransportRequest
from llm_api_adapter.llms.requests_transport import RequestsSyncTransport
from llm_api_adapter.llms.httpx_transport import HttpxSyncTransport
from llm_api_adapter.llms.openai_compatible.sync_client import OpenAICompatibleSyncClient
from llm_api_adapter.llms.openai_compatible.async_client import (
    OpenAICompatibleAsyncClient,
    _AsyncExecutionMixin,
)
from llm_api_adapter_zai.clients import (
    ZAI_BASE_URL,
    ZAI_CHAT_COMPLETIONS_URL,
    ZaiAsyncClient,
    ZaiSyncClient,
)
from llm_api_adapter_zai import clients as clients_package
from llm_api_adapter_zai.clients import async_client as async_client_module
from llm_api_adapter_zai.clients import sync_client as sync_client_module
from llm_api_adapter_zai.adapter import ZaiAdapter
from llm_api_adapter_zai.adapter import ZaiUsage
from llm_api_adapter_zai.plugin import PLUGIN
import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter.models.messages.chat_message import (
    AIMessage,
    ToolMessage,
    UserMessage,
)
from llm_api_adapter.models.messages.file_parts import DocumentPart, FilePart, ImagePart
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools.tool_call import ToolCall
from llm_api_adapter.models.tools.tool_spec import ToolSpec
from pydantic import BaseModel
from llm_api_adapter_zai.adapter import _ZaiStreamState
from llm_api_adapter_zai.streaming import ZaiStreamAssembler


ZAI_MODELS = tuple(PLUGIN.model_metadata.organization_data["models"])


def _is_loopback(host: Any) -> bool:
    if host is None or host == "":
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="ignore")
    if isinstance(host, str) and host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def deny_external_network(monkeypatch):
    """Keep accidental real transport use local while allowing loopback control sockets."""
    original_getaddrinfo = socket.getaddrinfo
    original_create_connection = socket.create_connection
    original_loop_create_connection = asyncio.BaseEventLoop.create_connection
    import requests

    def guarded_getaddrinfo(host, *args, **kwargs):
        if not _is_loopback(host):
            raise AssertionError("external DNS resolution attempted by a test")
        return original_getaddrinfo(host, *args, **kwargs)

    def guarded_create_connection(address, *args, **kwargs):
        if not _is_loopback(address[0]):
            raise AssertionError("external socket connection attempted by a test")
        return original_create_connection(address, *args, **kwargs)

    async def guarded_loop_create_connection(self, protocol_factory, host=None, port=None, *args, **kwargs):
        if not _is_loopback(host):
            raise AssertionError("external asyncio connection attempted by a test")
        return await original_loop_create_connection(
            self, protocol_factory, host=host, port=port, *args, **kwargs
        )

    def forbidden_requests(*args, **kwargs):
        raise AssertionError("Requests Session transport bypassed its test stub")

    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    monkeypatch.setattr(asyncio.BaseEventLoop, "create_connection", guarded_loop_create_connection)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden_requests)
    monkeypatch.setattr(requests.sessions.Session, "send", forbidden_requests)


@dataclass
class RecordingTransport(SyncTransport):
    body: Any = field(default_factory=lambda: {"ok": True})
    events: list[SSEEvent] = field(default_factory=lambda: [SSEEvent(None, data={"ok": True})])
    requests: list[TransportRequest] = field(default_factory=list)
    http_error: Exception | None = None

    def post_json(self, request, *, http_error_handler=None):
        self.requests.append(request)
        if self.http_error:
            http_error_handler(self.http_error)
        return JSONResponse(self.body)

    def post_multipart(self, request, form, *, http_error_handler=None):
        raise AssertionError("Z.ai must not use multipart")

    def post_sse(self, request, *, http_error_handler=None, stream_error_handler=None):
        self.requests.append(request)
        def events():
            for event in self.events:
                if event.event == "error" or (
                    isinstance(event.data, dict) and event.data.get("type") == "error"
                ):
                    stream_error_handler(event)
                yield event
        return events()


@pytest.mark.unit
def test_zai_clients_are_concrete_compatible_subclasses_with_native_payloads():
    assert clients_package.__all__ == [
        "ZAI_BASE_URL", "ZAI_CHAT_COMPLETIONS_URL", "ZaiAsyncClient",
        "ZaiSyncClient",
    ]
    assert clients_package.ZAI_BASE_URL == ZAI_BASE_URL
    assert clients_package.ZAI_CHAT_COMPLETIONS_URL == ZAI_CHAT_COMPLETIONS_URL
    assert issubclass(ZaiSyncClient, OpenAICompatibleSyncClient)
    assert ZaiSyncClient.__mro__[1] is OpenAICompatibleSyncClient
    assert issubclass(ZaiAsyncClient, OpenAICompatibleAsyncClient)
    assert ZaiAsyncClient.__bases__ == (OpenAICompatibleAsyncClient, ZaiSyncClient)

    sync_signature = inspect.signature(ZaiSyncClient)
    assert list(sync_signature.parameters) == ["api_key", "transport", "sync_transport"]
    assert sync_signature.parameters["api_key"].default is inspect.Parameter.empty
    assert sync_signature.parameters["sync_transport"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "endpoint" not in sync_signature.parameters
    assert list(inspect.signature(ZaiSyncClient.complete).parameters)[1:] == ["model", "timeout", "kwargs"]
    assert list(inspect.signature(ZaiAsyncClient).parameters) == ["api_key", "transport"]
    assert list(inspect.signature(ZaiAsyncClient.complete).parameters)[1:] == ["model", "timeout", "kwargs"]

    sync_parent = OpenAICompatibleSyncClient
    for method in ("complete", "stream", "stream_chat_completion", "_send_request", "_stream_request", "_headers"):
        assert inspect.getattr_static(ZaiSyncClient, method) is inspect.getattr_static(sync_parent, method)
    async_engine = _AsyncExecutionMixin
    for method in ("complete", "stream", "stream_chat_completion", "_send_request", "_stream_request"):
        assert inspect.getattr_static(ZaiAsyncClient, method) is inspect.getattr_static(async_engine, method)
    assert inspect.getattr_static(ZaiAsyncClient, "_headers") is inspect.getattr_static(sync_parent, "_headers")
    for method in ("_prepare_chat_payload_for_model", "_handle_http_error", "_handle_stream_error"):
        assert inspect.getattr_static(ZaiAsyncClient, method) is inspect.getattr_static(ZaiSyncClient, method)

    for key in ("short-key", "long synthetic key with additional safe characters"):
        sync_repr = repr(ZaiSyncClient(key, sync_transport=RecordingTransport()))
        async_repr = repr(ZaiAsyncClient(key))
        assert key not in sync_repr and key not in async_repr

    transports = [RecordingTransport(), RecordingTransport()]
    first = ZaiSyncClient("first sync synthetic key", sync_transport=transports[0])
    second = ZaiSyncClient("second sync synthetic key", sync_transport=transports[1])
    send_clients, stream_clients = [], []
    parent_send = OpenAICompatibleSyncClient._send_request
    parent_stream = OpenAICompatibleSyncClient._stream_request
    def send_seen(self, *args, **kwargs):
        send_clients.append(self)
        return parent_send(self, *args, **kwargs)
    def stream_seen(self, *args, **kwargs):
        stream_clients.append(self)
        return parent_stream(self, *args, **kwargs)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(OpenAICompatibleSyncClient, "_send_request", send_seen)
    monkeypatch.setattr(OpenAICompatibleSyncClient, "_stream_request", stream_seen)
    try:
        first.complete("glm-5.3-flash", timeout=3, max_tokens=17)
        second.chat("glm-5.3-flashx", timeout=4, max_tokens=18)
        first.complete("glm-5.3-flash", timeout=5, max_tokens=19)
        stream_options = [
            {},
            {"stream_options": {"include_usage": False}},
            {"stream_options": {}},
            {"stream_options": None},
        ]
        snapshots = copy.deepcopy(stream_options)
        for index, options in enumerate(stream_options):
            list(
                (first if index % 2 == 0 else second).stream(
                    "glm-5.3-flash",
                    timeout=6 + index,
                    max_tokens=20 + index,
                    **options,
                )
            )
    finally:
        monkeypatch.undo()

    assert send_clients == [first, second, first]
    assert stream_clients == [first, second, first, second]
    assert stream_options == snapshots
    all_requests = transports[0].requests + transports[1].requests
    assert all(request.url == ZAI_CHAT_COMPLETIONS_URL for request in all_requests)
    assert [request.headers_dict()["Authorization"] for request in transports[0].requests] == [
        "Bearer first sync synthetic key"
    ] * 4
    assert [request.headers_dict()["Authorization"] for request in transports[1].requests] == [
        "Bearer second sync synthetic key"
    ] * 3
    assert [request.timeout for request in transports[0].requests] == [3, 5, 6, 8]
    assert [request.timeout for request in transports[1].requests] == [4, 7, 9]
    assert [request.payload["max_tokens"] for request in transports[0].requests] == [17, 19, 20, 22]
    assert [request.payload["max_tokens"] for request in transports[1].requests] == [18, 21, 23]
    assert [request.payload.get("stream_options") for request in transports[0].requests[2:]] == [
        {"include_usage": True}, {},
    ]
    assert [request.payload.get("stream_options") for request in transports[1].requests[1:]] == [
        {"include_usage": False}, None,
    ]


@pytest.mark.unit
def test_zai_async_client_inherits_compatible_engine_and_live_patch_points(monkeypatch):
    seen = []
    async def fake_request(url, **kwargs):
        seen.append(("json", url, kwargs))
        return {"ok": True}
    async def fake_stream(url, **kwargs):
        seen.append(("sse", url, kwargs))
        yield SSEEvent(None, data={"ok": True})

    class GuardAsyncClient:
        async def post(self, *args, **kwargs):
            raise AssertionError("async package seam was bypassed")
        def build_request(self, *args, **kwargs):
            raise AssertionError("async package seam was bypassed")
        async def send(self, *args, **kwargs):
            raise AssertionError("async package seam was bypassed")
        async def aclose(self):
            return None
    class GuardHTTPX:
        AsyncClient = GuardAsyncClient
        TimeoutException = type("TimeoutException", (Exception,), {})
        HTTPStatusError = type("HTTPStatusError", (Exception,), {})
        RequestError = type("RequestError", (Exception,), {})

    first = ZaiAsyncClient("first async synthetic key")
    second = ZaiAsyncClient("second async synthetic key")
    send_clients, stream_clients = [], []
    parent_send = _AsyncExecutionMixin._send_request
    parent_stream = _AsyncExecutionMixin._stream_request
    async def send_seen(self, *args, **kwargs):
        send_clients.append(self)
        return await parent_send(self, *args, **kwargs)
    async def stream_seen(self, *args, **kwargs):
        stream_clients.append(self)
        async for event in parent_stream(self, *args, **kwargs):
            yield event
    monkeypatch.setattr(_AsyncExecutionMixin, "_send_request", send_seen)
    monkeypatch.setattr(_AsyncExecutionMixin, "_stream_request", stream_seen)
    monkeypatch.setattr(async_client_module, "async_request", fake_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", fake_stream)
    monkeypatch.setitem(sys.modules, "httpx", GuardHTTPX)

    stream_options = [
        {},
        {"stream_options": {"include_usage": False}},
        {"stream_options": {}},
        {"stream_options": None},
    ]
    snapshots = copy.deepcopy(stream_options)
    async def exercise():
        assert await first.complete("glm-5.3-flash", timeout=3, max_tokens=17) == {"ok": True}
        assert await second.chat("glm-5.3-flashx", timeout=4, max_tokens=18) == {"ok": True}
        assert await first.complete("glm-5.3-flash", timeout=5, max_tokens=19) == {"ok": True}
        for index, options in enumerate(stream_options):
            client = first if index % 2 == 0 else second
            assert [event async for event in client.stream(
                "glm-5.3-flash", timeout=6 + index, max_tokens=20 + index, **options
            )]
    asyncio.run(exercise())

    assert send_clients == [first, second, first]
    assert stream_clients == [first, second, first, second]
    assert stream_options == snapshots
    assert [item[0] for item in seen] == ["json", "json", "json", "sse", "sse", "sse", "sse"]
    assert all(item[1] == ZAI_CHAT_COMPLETIONS_URL for item in seen)
    assert [item[2]["headers"]["Authorization"] for item in seen] == [
        "Bearer first async synthetic key", "Bearer second async synthetic key",
        "Bearer first async synthetic key", "Bearer first async synthetic key",
        "Bearer second async synthetic key", "Bearer first async synthetic key",
        "Bearer second async synthetic key",
    ]
    assert [item[2]["timeout"] for item in seen] == [3, 4, 5, 6, 7, 8, 9]
    assert [item[2]["payload"]["max_tokens"] for item in seen] == [17, 18, 19, 20, 21, 22, 23]
    assert [item[2]["payload"].get("stream_options") for item in seen[3:]] == [
        {"include_usage": True}, {"include_usage": False}, {}, None,
    ]


@pytest.mark.unit
@pytest.mark.parametrize("body", [None, [], "text", 3])
def test_zai_client_rejects_non_object_json(body):
    with pytest.raises(LLMAPIClientError):
        ZaiSyncClient("synthetic", sync_transport=RecordingTransport(body=body)).complete(model="glm-5.3-flash")


@pytest.mark.unit
@pytest.mark.parametrize("body", [None, [], "text", 3])
def test_zai_async_client_rejects_non_object_json(monkeypatch, body):
    async def fake_request(url, **kwargs):
        return body
    monkeypatch.setattr(async_client_module, "async_request", fake_request)
    with pytest.raises(LLMAPIClientError):
        asyncio.run(
            ZaiAsyncClient("synthetic").complete(model="glm-5.3-flash")
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "payload", "error"),
    [
        (401, {"error": {"message": "denied"}}, LLMAPIAuthorizationError),
        (403, {"error": {"message": "denied"}}, LLMAPIAuthorizationError),
        (429, {"error": {"type": "rate_limit_error"}}, LLMAPIRateLimitError),
        (402, {"error": {"type": "insufficient_balance"}}, LLMAPIUsageLimitError),
        (408, {"error": {}}, LLMAPITimeoutError),
        (504, {"error": {}}, LLMAPITimeoutError),
        (400, {"error": {"type": "context_length_exceeded"}}, LLMAPITokenLimitError),
        (500, {"error": {}}, LLMAPIServerError),
        (422, {"error": {}}, LLMAPIClientError),
    ],
)
def test_zai_http_categories_remain_native(status, payload, error):
    class Response:
        status_code = status
        def json(self): return payload
    class HTTPError(Exception):
        response = Response()
    with pytest.raises(error):
        ZaiSyncClient("synthetic", sync_transport=RecordingTransport(http_error=HTTPError("synthetic"))).complete(model="glm-5.3-flash")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error_type", "error"),
    [
        ("authentication_error", LLMAPIAuthorizationError),
        ("rate_limit_error", LLMAPIRateLimitError),
        ("insufficient_quota", LLMAPIUsageLimitError),
        ("context_length_exceeded", LLMAPITokenLimitError),
        ("timeout_error", LLMAPITimeoutError),
        ("api_error", LLMAPIServerError),
        ("invalid_request", LLMAPIClientError),
    ],
)
def test_zai_sse_categories_remain_native(error_type, error):
    event = SSEEvent("error", {"error": {"type": error_type, "message": "synthetic"}})
    client = ZaiSyncClient("synthetic", sync_transport=RecordingTransport(events=[event]))
    with pytest.raises(error):
        list(client.stream(model="glm-5.3-flash"))


@pytest.mark.unit
def test_zai_direct_client_reuses_parent_transport_engine(monkeypatch):
    transports = [RecordingTransport(), RecordingTransport()]
    first = ZaiSyncClient("first direct synthetic key", sync_transport=transports[0])
    second = ZaiSyncClient("second direct synthetic key", sync_transport=transports[1])
    send_clients, stream_clients = [], []
    parent_send = OpenAICompatibleSyncClient._send_request
    parent_stream = OpenAICompatibleSyncClient._stream_request
    def send_seen(self, *args, **kwargs):
        send_clients.append(self)
        return parent_send(self, *args, **kwargs)
    def stream_seen(self, *args, **kwargs):
        stream_clients.append(self)
        return parent_stream(self, *args, **kwargs)
    monkeypatch.setattr(OpenAICompatibleSyncClient, "_send_request", send_seen)
    monkeypatch.setattr(OpenAICompatibleSyncClient, "_stream_request", stream_seen)
    stream_options = [{}, {"stream_options": {"include_usage": False}}]
    snapshots = copy.deepcopy(stream_options)
    first.complete("glm-5.3-flash", timeout=2, max_tokens=11)
    second.chat("glm-5.3-flashx", timeout=3, max_tokens=12)
    first.complete("glm-5.3-flash", timeout=4, max_tokens=13)
    list(first.stream("glm-5.3-flash", timeout=5, max_tokens=14, **stream_options[0]))
    list(second.stream("glm-5.3-flashx", timeout=6, max_tokens=15, **stream_options[1]))
    assert send_clients == [first, second, first]
    assert stream_clients == [first, second]
    assert stream_options == snapshots
    assert [request.url for transport in transports for request in transport.requests] == [
        ZAI_CHAT_COMPLETIONS_URL,
    ] * 5
    assert [request.headers_dict()["Authorization"] for request in transports[0].requests] == [
        "Bearer first direct synthetic key",
    ] * 3
    assert [request.headers_dict()["Authorization"] for request in transports[1].requests] == [
        "Bearer second direct synthetic key",
    ] * 2
    assert [request.timeout for request in transports[0].requests] == [2, 4, 5,]
    assert [request.timeout for request in transports[1].requests] == [3, 6]
    assert [request.payload["max_tokens"] for transport in transports for request in transport.requests] == [11, 13, 14, 12, 15]
    assert transports[0].requests[2].payload["stream_options"] == {"include_usage": True}
    assert transports[1].requests[1].payload["stream_options"] == {"include_usage": False}
    invalid = ZaiSyncClient("invalid synthetic", sync_transport=RecordingTransport(body=[]))
    with pytest.raises(LLMAPIClientError):
        invalid.complete(model="glm-5.3-flash", max_tokens=1)


@pytest.mark.unit
def test_zai_adapter_factories_reuse_exact_injected_transport(monkeypatch, zai_adapter_runtime):
    from llm_api_adapter_zai import adapter as adapter_module
    created = []
    adapter_factory = adapter_module.create_sync_transport
    def counted_factory(name):
        transport = adapter_factory(name)
        created.append(transport)
        return transport
    monkeypatch.setattr(adapter_module, "create_sync_transport", counted_factory)
    adapter = ZaiAdapter("first adapter synthetic key", ZAI_MODELS[0])
    initial_transport = adapter._sync_transport
    assert created == [initial_transport]

    def forbidden_factory(self, transport):
        raise AssertionError("a client factory allocated a duplicate sync transport")
    monkeypatch.setattr(OpenAICompatibleSyncClient, "_create_sync_transport", forbidden_factory)
    first_sync = adapter._create_sync_client()
    first_async = adapter._create_async_client()
    assert first_sync is not adapter._create_sync_client()
    assert first_async is not adapter._create_async_client()
    assert first_sync._sync_transport is initial_transport
    assert first_async._sync_transport is None
    assert first_sync.api_key == first_async.api_key == "first adapter synthetic key"

    second_transport = RecordingTransport()
    adapter.api_key = "second adapter synthetic key"
    adapter._sync_transport = second_transport
    second_sync = adapter._create_sync_client()
    second_async = adapter._create_async_client()
    assert second_sync.api_key == second_async.api_key == "second adapter synthetic key"
    assert second_sync._sync_transport is second_transport
    assert first_sync.api_key == first_async.api_key == "first adapter synthetic key"
    assert first_sync._sync_transport is initial_transport
    assert len(created) == 1
    assert not hasattr(adapter, "_client") and not hasattr(adapter, "_async_client")


@pytest.mark.unit
def test_zai_async_factory_uses_shared_async_execution_and_live_patch_points(monkeypatch):
    assert ZaiAsyncClient._prepare_chat_payload_for_model is ZaiSyncClient._prepare_chat_payload_for_model
    assert inspect.getattr_static(ZaiAsyncClient, "_handle_http_error") is inspect.getattr_static(ZaiSyncClient, "_handle_http_error")
    assert inspect.getattr_static(ZaiAsyncClient, "_handle_stream_error") is inspect.getattr_static(ZaiSyncClient, "_handle_stream_error")
    client = ZaiAsyncClient("async factory synthetic key")
    assert client._sync_transport is None
    assert inspect.getattr_static(ZaiAsyncClient, "_send_request") is inspect.getattr_static(_AsyncExecutionMixin, "_send_request")
    assert inspect.getattr_static(ZaiAsyncClient, "_stream_request") is inspect.getattr_static(_AsyncExecutionMixin, "_stream_request")
    assert inspect.getattr_static(ZaiAsyncClient, "_headers") is inspect.getattr_static(OpenAICompatibleSyncClient, "_headers")


@pytest.mark.unit
@pytest.mark.parametrize("case", ["default_sync", "async", "httpx"])
def test_zai_without_httpx_and_anyio_subprocess(case):
    repo = ROOT
    child = r'''import importlib.abc, sys
from pathlib import Path
core, package = map(Path, sys.argv[1:3])
sys.path[:0] = [str(package), str(core)]
blocked = ("httpx", "anyio")
assert not any(name == root or name.startswith(root + ".") for name in sys.modules for root in blocked)
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == root or fullname.startswith(root + ".") for root in blocked):
            raise ModuleNotFoundError("optional runtime unavailable: " + fullname)
sys.meta_path.insert(0, Guard())
import requests
from llm_api_adapter_zai.plugin import PLUGIN
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
model = next(iter(PLUGIN.model_metadata.organization_data["models"]))
mr = RegistrySpec(); mr.register_organization_metadata(PLUGIN.model_metadata)
sr = ServiceProviderRegistry(); PLUGIN.register(sr)
import llm_api_adapter.universal_adapter as um
import llm_api_adapter.adapters.base_adapter as bm
um.LLM_REGISTRY = mr; bm.LLM_REGISTRY = mr; um.SERVICE_PROVIDER_REGISTRY = sr
calls = []
class Response:
    status_code = 200
    headers = {"Content-Type": "application/json"}
    def __init__(self, stream=False): self.stream = stream
    def raise_for_status(self): pass
    def json(self): return {"id":"x","model":model,"choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}
    def iter_lines(self, **kwargs): return iter([b'data: {"id":"s","model":"' + model.encode() + b'","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}', b"", b"data: [DONE]", b""])
    def close(self): pass
def post(url, **kwargs): calls.append((url, kwargs)); return Response(kwargs.get("stream", False))
def forbid_network(*args, **kwargs): raise AssertionError("unexpected requests transport path")
requests.sessions.Session.request = forbid_network
requests.sessions.Session.send = forbid_network
requests.post = post
mode = sys.argv[3]
if mode == "default_sync":
    a = UniversalLLMAPIAdapter(organization="zai", model=model, api_key="synthetic")
    assert a.chat([{"role":"user","content":"hello"}]).content == "ok"
    assert list(a.stream_chat([{"role":"user","content":"hello"}])) == ["ok"]
    assert len(calls) == 2 and all(x[0].startswith("https://api.z.ai/") for x in calls)
elif mode == "async":
    a = UniversalLLMAPIAdapter(organization="zai", model=model, api_key="synthetic")
    async def check():
        try: await a.achat([{"role":"user","content":"hello"}])
        except ImportError as e: assert "llm-api-adapter[async]" in str(e)
        else: raise AssertionError("async dependency error missing")
        try: await anext(a.astream_chat([{"role":"user","content":"hello"}]))
        except ImportError as e: assert "llm-api-adapter[async]" in str(e)
        else: raise AssertionError("async stream dependency error missing")
    import asyncio; asyncio.run(check()); assert calls == []
else:
    try: UniversalLLMAPIAdapter(organization="zai", model=model, api_key="synthetic", transport="httpx")
    except ImportError as e: assert "llm-api-adapter[httpx]" in str(e)
    else: raise AssertionError("HTTPX dependency error missing")
    assert calls == []
assert not any(name == root or name.startswith(root + ".") for name in sys.modules for root in blocked)
'''
    provider_secrets = {
        "ZAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
        "GEMINI_API_KEY", "MISTRAL_API_KEY", "XAI_API_KEY", "QWEN_API_KEY",
        "KIMI_API_KEY", "DEEPSEEK_API_KEY", "QWEN_WORKSPACE_ID",
    }
    env = {
        key: value for key, value in os.environ.items()
        if key not in provider_secrets
        and not any(marker in key.upper() for marker in ("ZAI", "API_KEY", "OPENAI", "ANTHROPIC", "GOOGLE"))
    }
    env["PYTHON_DOTENV_DISABLED"] = "1"
    result = subprocess.run(
        [sys.executable, "-I", "-c", child, str(repo / "src"), str(repo / "packages/organizations/zai/src"), case],
        cwd=repo, env=env, text=True, capture_output=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def zai_adapter_runtime(monkeypatch):
    """Install the real package registry into the adapter and facade lookup sites."""
    model_registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert model_registry.register_organization_metadata(PLUGIN.model_metadata)
    provider_registry = ServiceProviderRegistry()
    PLUGIN.register(provider_registry)
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", model_registry)
    monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", provider_registry)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", model_registry)
    return model_registry


def _native_adapter(mode, model, transport="requests"):
    if mode == "facade":
        facade = UniversalLLMAPIAdapter(
            organization="zai", model=model, api_key="synthetic-zai-key", transport=transport
        )
        return facade, facade.adapter
    adapter = ZaiAdapter(
        api_key="synthetic-zai-key", model=model, transport=transport
    )
    return adapter, adapter


def _response_payload(model):
    return {
        "id": "synthetic-response-id",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "synthetic answer"},
            "finish_reason": "stop",
        }],
    }


def _stream_payload(model):
    return {
        "id": "synthetic-stream-id",
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"content": "synthetic answer"},
            "finish_reason": "stop",
        }],
    }


class _ControlledHTTPResponse:
    """Minimal response surface consumed by the real sync transport implementations."""

    status_code = 200

    def __init__(self, payload, *, streaming=False):
        self.payload = payload
        self.streaming = streaming
        self.closed = False

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload

    def iter_lines(self, **kwargs):
        del kwargs
        data = _stream_payload(self.payload["model"])
        import json
        yield ("data: " + json.dumps(data)).encode()
        yield b""
        yield b"data: [DONE]"
        yield b""

    def close(self):
        self.closed = True


class _ControlledHTTPXClient:
    """HTTPX client double that records calls while preserving actual transport routing."""

    def __init__(self, observed):
        self.observed = observed
        self.closed = False

    def post(self, url, *, headers, json, timeout):
        self.observed.append(("json", url, headers, json, timeout))
        return _ControlledHTTPResponse(_response_payload(json["model"]))

    def build_request(self, method, url, *, headers, json, timeout):
        self.observed.append(("build", method, url, headers, json, timeout))
        return (method, url, headers, json, timeout)

    def send(self, request, *, stream):
        method, url, headers, payload, timeout = request
        self.observed.append(("send", method, url, headers, payload, timeout, stream))
        return _ControlledHTTPResponse(_response_payload(payload["model"]), streaming=True)

    def close(self):
        self.closed = True


def _install_sync_http_mock(monkeypatch, transport_name):
    """Patch only the backend I/O methods used by the selected real sync transport."""
    observed = []
    if transport_name == "requests":
        from llm_api_adapter.llms import requests_transport

        def post(url, *, headers, json, timeout, stream=False):
            observed.append(("stream" if stream else "json", url, headers, json, timeout, stream))
            return _ControlledHTTPResponse(_response_payload(json["model"]), streaming=stream)

        monkeypatch.setattr(requests_transport.requests, "post", post)
    else:
        import httpx

        monkeypatch.setattr(
            httpx, "Client", lambda: _ControlledHTTPXClient(observed)
        )
    return observed


def _assert_native_request(observed, transport_name, method_kind, model, *, timeout):
    assert observed
    expected_io_kind = "json" if method_kind == "chat" else "stream"
    if transport_name == "requests":
        assert len(observed) == 1
        call = observed[-1]
        assert call[0] == expected_io_kind
        _, url, headers, payload, actual_timeout, *stream_args = call
    elif method_kind == "chat":
        assert len(observed) == 1
        _, url, headers, payload, actual_timeout = observed[-1]
        assert observed[-1][0] == "json"
    else:
        assert len(observed) == 2
        built, sent = observed[-2:]
        assert built[0] == "build" and sent[0] == "send"
        _, _, url, headers, payload, actual_timeout = built
        assert sent[-1] is True
    assert url == ZAI_CHAT_COMPLETIONS_URL
    assert headers == {
        "Authorization": "Bearer synthetic-zai-key",
        "Content-Type": "application/json",
    }
    assert actual_timeout == timeout
    expected_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "preserve this request"}],
        "max_tokens": 17,
        "temperature": 0.4,
        "top_p": 0.7,
    }
    if method_kind == "stream":
        expected_payload.update({
            "stream": True,
            "stream_options": {"include_usage": True},
        })
    assert payload == expected_payload


@pytest.mark.unit
def test_zai_native_constructor_shape_equality_and_supported_imports(zai_adapter_runtime):
    """Keep the native dataclass's caller-visible positional and keyword contract."""
    from llm_api_adapter_zai import adapter as public_adapter_module

    signature = inspect.signature(ZaiAdapter)
    expected = [
        ("api_key", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("model", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("company", inspect.Parameter.POSITIONAL_OR_KEYWORD, "zai"),
        ("transport", inspect.Parameter.POSITIONAL_OR_KEYWORD, "requests"),
        ("pricing", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
        ("is_reasoning", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
        ("is_adaptive_thinking", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
        ("service_provider", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
    ]
    assert [
        (name, item.kind, item.default)
        for name, item in signature.parameters.items()
    ] == expected
    field_names = {item.name for item in dataclass_fields(ZaiAdapter)}
    assert "base_url" not in field_names and "api_variant" not in field_names
    assert public_adapter_module.ZaiAdapter is ZaiAdapter
    assert PLUGIN.register is not None
    first = ZaiAdapter("synthetic-zai-key", ZAI_MODELS[0])
    second = ZaiAdapter(api_key="synthetic-zai-key", model=ZAI_MODELS[0])
    assert first == second
    second._sync_transport = RecordingTransport()
    assert first == second
    resolved_spec = first.model_spec
    first.model_spec = None
    assert first != second
    first.model_spec = resolved_spec
    assert first == second
    assert first.service_provider == first.company == "zai"
    explicit = ZaiAdapter(
        api_key="synthetic-zai-key", model=ZAI_MODELS[0], service_provider="zai"
    )
    assert explicit == first
    assert ZaiAdapter(
        "synthetic-zai-key", ZAI_MODELS[0], "zai", "requests", None, False, False, None
    ) == first
    assert inspect.signature(ZaiAdapter).parameters["model"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    with pytest.raises(TypeError):
        ZaiAdapter(api_key="synthetic-zai-key", model=ZAI_MODELS[0], base_url="https://x.invalid")
    with pytest.raises(TypeError):
        ZaiAdapter(api_key="synthetic-zai-key", model=ZAI_MODELS[0], api_variant="chat_completions")
    assert "base_url" not in signature.parameters and "api_variant" not in signature.parameters


@pytest.mark.unit
@pytest.mark.parametrize(
    "constructor_kwargs",
    [
        {"api_key": "", "model": ZAI_MODELS[0]},
        {"api_key": "synthetic-zai-key", "model": ZAI_MODELS[0], "transport": "invalid"},
        {"api_key": "synthetic-zai-key", "model": ZAI_MODELS[0], "service_provider": ""},
        {"api_key": "synthetic-zai-key", "model": ZAI_MODELS[0], "service_provider": object()},
    ],
)
def test_zai_native_constructor_validation_stays_at_construction_without_io(
    zai_adapter_runtime, monkeypatch, constructor_kwargs
):
    """Preserve existing native configuration errors before constructing a usable adapter."""
    sync_sends = []
    async_sends = []

    def fake_post(*args, **kwargs):
        sync_sends.append((args, kwargs))
        raise AssertionError("invalid construction must not send")

    async def fake_async_request(url, **kwargs):
        async_sends.append((url, kwargs))
        raise AssertionError("invalid construction must not send")

    from llm_api_adapter.llms import requests_transport
    monkeypatch.setattr(requests_transport.requests, "post", fake_post)
    monkeypatch.setattr(async_client_module, "async_request", fake_async_request)
    with pytest.raises(ValueError):
        ZaiAdapter(**constructor_kwargs)
    assert sync_sends == [] and async_sends == []


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("explicit_provider", [False, True])
def test_zai_facade_resolves_implicit_and_explicit_native_service_provider(
    zai_adapter_runtime, model, explicit_provider
):
    """Resolve every registered Z.ai profile under implicit and explicit provider selection."""
    options = {"service_provider": "zai"} if explicit_provider else {}
    facade = UniversalLLMAPIAdapter(
        organization="zai",
        model=model,
        api_key="synthetic-zai-key",
        **options,
    )
    assert isinstance(facade.adapter, ZaiAdapter)
    assert facade.adapter.service_provider == "zai"
    assert facade.adapter.model_spec is zai_adapter_runtime.organizations["zai"].models[model]


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("mode", ["direct", "facade"])
@pytest.mark.parametrize("transport_name", ["requests", "httpx"])
@pytest.mark.parametrize("method_kind", ["chat", "stream"])
def test_zai_native_sync_methods_preserve_real_transport_wire_contract(
    zai_adapter_runtime, monkeypatch, model, mode, transport_name, method_kind
):
    observed = _install_sync_http_mock(monkeypatch, transport_name)
    caller, adapter = _native_adapter(mode, model, transport_name)
    transport_type = RequestsSyncTransport if transport_name == "requests" else HttpxSyncTransport
    assert isinstance(adapter._sync_transport, transport_type)
    method = getattr(caller, "stream_chat" if method_kind == "stream" else "chat")
    kwargs = dict(
        max_tokens=17, temperature=0.4, top_p=0.7, timeout_s=8.25
    )
    completed = []
    if method_kind == "chat":
        response = method([UserMessage("preserve this request")], **kwargs)
        assert isinstance(response, ChatResponse)
    else:
        assert list(method([UserMessage("preserve this request")], on_done=completed.append, **kwargs)) == ["synthetic answer"]
        assert len(completed) == 1
        assert completed[0].response_id == "synthetic-stream-id"
        assert completed[0].model == model and completed[0].content == "synthetic answer"
        kwargs["max_tokens"] = 17
    if method_kind == "chat":
        assert response.content == "synthetic answer"
        assert response.response_id == "synthetic-response-id"
        assert response.model == model
    _assert_native_request(observed, transport_name, method_kind, model, timeout=8.25)


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("mode", ["direct", "facade"])
@pytest.mark.parametrize("method_kind", ["achat", "astream_chat"])
def test_zai_native_async_methods_preserve_nondefault_wire_contract(
    zai_adapter_runtime, monkeypatch, model, mode, method_kind
):
    observed = []

    async def async_request(url, **kwargs):
        observed.append(("json", url, kwargs))
        return _response_payload(model)

    async def async_stream(url, **kwargs):
        observed.append(("stream", url, kwargs))
        yield SSEEvent(None, data=_stream_payload(model))

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    caller, _ = _native_adapter(mode, model)
    request_kwargs = {
        "max_tokens": 23, "temperature": 0.6, "top_p": 0.8, "timeout_s": 6.75
    }
    if method_kind == "achat":
        response = asyncio.run(
            caller.achat([UserMessage("preserve this request")], **request_kwargs)
        )
        assert isinstance(response, ChatResponse)
        assert response.content == "synthetic answer"
        kind, url, kwargs = observed[-1]
        assert kind == "json"
        assert response.response_id == "synthetic-response-id" and response.model == model
    else:
        completed = []
        async def consume():
            return [
                text async for text in caller.astream_chat(
                    [UserMessage("preserve this request")],
                    on_done=completed.append,
                    **request_kwargs,
                )
            ]

        assert asyncio.run(consume()) == ["synthetic answer"]
        assert len(completed) == 1
        assert completed[0].response_id == "synthetic-stream-id"
        assert completed[0].model == model and completed[0].content == "synthetic answer"
        kind, url, kwargs = observed[-1]
        assert kind == "stream"
    assert len(observed) == 1
    assert url == ZAI_CHAT_COMPLETIONS_URL
    assert kwargs["headers"] == {
        "Authorization": "Bearer synthetic-zai-key",
        "Content-Type": "application/json",
    }
    assert kwargs["timeout"] == 6.75
    expected_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "preserve this request"}],
        "max_tokens": 23,
        "temperature": 0.6,
        "top_p": 0.8,
    }
    if method_kind == "astream_chat":
        expected_payload.update({
            "stream": True,
            "stream_options": {"include_usage": True},
        })
    assert kwargs["payload"] == expected_payload


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("mode", ["direct", "facade"])
@pytest.mark.parametrize("profile", ["omitted", "explicit_defaults"])
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
def test_zai_native_methods_omit_native_default_request_fields(
    zai_adapter_runtime, monkeypatch, model, mode, profile, method_kind
):
    observed = []
    caller, adapter = _native_adapter(mode, model)
    adapter._sync_transport = RecordingTransport(body=_response_payload(model))

    async def async_request(url, **kwargs):
        observed.append((url, kwargs))
        return _response_payload(model)

    async def async_stream(url, **kwargs):
        observed.append((url, kwargs))
        yield SSEEvent(None, data=_stream_payload(model))

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    method = getattr(caller, method_kind)
    request_kwargs = {} if profile == "omitted" else {
        "max_tokens": None, "temperature": 1.0, "top_p": 1.0, "timeout_s": None
    }
    if method_kind == "chat":
        response = method([UserMessage("defaults")], **request_kwargs)
        assert response.content == "synthetic answer"
        payload = adapter._sync_transport.requests[-1].payload
    elif method_kind == "stream_chat":
        adapter._sync_transport.events = [SSEEvent(None, data=_stream_payload(model))]
        assert list(method([UserMessage("defaults")], **request_kwargs)) == ["synthetic answer"]
        payload = adapter._sync_transport.requests[-1].payload
    elif method_kind == "achat":
        response = asyncio.run(method([UserMessage("defaults")], **request_kwargs))
        assert response.content == "synthetic answer"
        payload = observed[-1][1]["payload"]
    else:
        async def consume():
            return [text async for text in method([UserMessage("defaults")], **request_kwargs)]
        assert asyncio.run(consume()) == ["synthetic answer"]
        payload = observed[-1][1]["payload"]
    if method_kind in {"chat", "stream_chat"}:
        assert len(adapter._sync_transport.requests) == 1
        assert adapter._sync_transport.requests[-1].timeout is None
    else:
        assert len(observed) == 1
        assert observed[-1][1]["timeout"] is None
    expected_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "defaults"}],
    }
    if method_kind in {"stream_chat", "astream_chat"}:
        expected_payload.update({
            "stream": True,
            "stream_options": {"include_usage": True},
        })
    assert payload == expected_payload


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
def test_zai_public_methods_reject_invalid_requests_before_any_transport(
    zai_adapter_runtime, monkeypatch, model, method_kind
):
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    transport = RecordingTransport(body=_response_payload(model))
    adapter._sync_transport = transport
    async_calls = []

    async def async_request(url, **kwargs):
        async_calls.append((url, kwargs))
        return _response_payload(model)

    async def async_stream(url, **kwargs):
        async_calls.append((url, kwargs))
        yield SSEEvent(None, data=_stream_payload(model))

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    model_spec = zai_adapter_runtime.organizations["zai"].models[model]
    tool_choice_rule = next(
        rule for rule in model_spec.request_rules.rules
        if rule.handler == "restrict_tool_choice"
    )
    allowed_tool_choices = set(tool_choice_rule.arguments["allowed_values"])
    assert "none" not in allowed_tool_choices
    valid_tool = ToolSpec(
        name="lookup",
        description="Look up a value.",
        json_schema={"type": "object", "properties": {}},
    )
    invalid_cases = [
        ({"max_tokens": True}, ValueError),
        ({"max_tokens": 0}, ValueError),
        ({"max_tokens": -1}, ValueError),
        ({"max_tokens": 1.5}, ValueError),
        ({"max_tokens": model_spec.limits.max_output_tokens + 1}, ValueError),
        ({"temperature": 2.1}, ValueError),
        ({"temperature": -0.1}, ValueError),
        ({"top_p": 1.1}, ValueError),
        ({"top_p": -0.1}, ValueError),
        ({"tool_choice": "none", "tools": [valid_tool]}, ToolChoiceError),
        ({"parallel_tool_calls": True}, NotImplementedError),
        ({"parallel_tool_calls": False}, NotImplementedError),
        ({"messages": [UserMessage("doc", files=[DocumentPart(url="https://example.test/file.pdf")])]}, ValueError),
        ({"messages": [UserMessage("doc", files=[DocumentPart(data=b"%PDF", media_type="application/pdf")])]}, ValueError),
        ({"json_schema": {"type": "object"}}, NotImplementedError),
        ({"response_model": dict}, NotImplementedError),
    ]
    if (
        model_spec.reasoning_capability is not None
        and getattr(model_spec.reasoning_capability, "allowed_values", ())
    ):
        invalid_cases.append(({"reasoning_level": "not-a-registered-level"}, LLMReasoningLevelError))
    for invalid_kwargs, expected_error in invalid_cases:
        messages = invalid_kwargs.pop("messages", [UserMessage("invalid")])
        if method_kind == "chat":
            invocation = lambda: adapter.chat(messages, **invalid_kwargs)
        elif method_kind == "achat":
            invocation = lambda: asyncio.run(adapter.achat(messages, **invalid_kwargs))
        elif method_kind == "stream_chat":
            invocation = lambda: list(adapter.stream_chat(messages, **invalid_kwargs))
        else:
            async def consume():
                async for _ in adapter.astream_chat(messages, **invalid_kwargs):
                    pass
            invocation = lambda: asyncio.run(consume())
        with pytest.raises(expected_error) as caught:
            invocation()
        assert type(caught.value) is expected_error
    assert transport.requests == []
    assert async_calls == []


@pytest.mark.unit
def test_zai_sync_transport_replacement_drives_json_and_sse_calls(zai_adapter_runtime):
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=ZAI_MODELS[0])
    replacement = RecordingTransport(
        body=_response_payload(ZAI_MODELS[0]),
        events=[SSEEvent(None, data=_stream_payload(ZAI_MODELS[0]))],
    )
    adapter._sync_transport = replacement
    assert adapter.chat([UserMessage("replacement")]).content == "synthetic answer"
    assert list(adapter.stream_chat([UserMessage("replacement")])) == ["synthetic answer"]
    assert len(replacement.requests) == 2
    assert "stream" not in replacement.requests[0].payload
    assert replacement.requests[1].payload["stream"] is True


@pytest.mark.unit
def test_zai_sync_stream_iterator_defers_valid_preparation_until_consumption(
    zai_adapter_runtime, monkeypatch
):
    """A valid stream iterator does not prepare or send until iteration begins."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=ZAI_MODELS[0])
    transport = RecordingTransport(events=[SSEEvent(None, data=_stream_payload(ZAI_MODELS[0]))])
    adapter._sync_transport = transport
    preparations = []
    original_prepare = LLMAdapterBase._prepare_chat_request

    def observed_prepare(self, *args, **kwargs):
        preparations.append((args, kwargs))
        return original_prepare(self, *args, **kwargs)

    monkeypatch.setattr(LLMAdapterBase, "_prepare_chat_request", observed_prepare)
    iterator = adapter.stream_chat([UserMessage("lazy")])
    assert preparations == [] and transport.requests == []
    assert list(iterator) == ["synthetic answer"]
    assert preparations and len(transport.requests) == 1


@pytest.mark.unit
def test_zai_sync_stream_iterator_defers_invalid_max_tokens_until_consumption(
    zai_adapter_runtime, monkeypatch
):
    """Invalid output limits fail at iteration, before any transport send."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=ZAI_MODELS[0])
    transport = RecordingTransport(events=[SSEEvent(None, data=_stream_payload(ZAI_MODELS[0]))])
    adapter._sync_transport = transport
    preparations = []
    original_prepare = LLMAdapterBase._prepare_chat_request

    def observed_prepare(self, *args, **kwargs):
        preparations.append((args, kwargs))
        return original_prepare(self, *args, **kwargs)

    monkeypatch.setattr(LLMAdapterBase, "_prepare_chat_request", observed_prepare)
    invalid_iterator = adapter.stream_chat([UserMessage("lazy invalid")], max_tokens=0)
    assert preparations == [] and transport.requests == []
    with pytest.raises(ValueError) as caught:
        list(invalid_iterator)
    assert type(caught.value) is ValueError
    assert preparations and transport.requests == []


@pytest.mark.unit
def test_zai_native_compatible_inheritance_and_method_ownership():
    """Keep compatible execution inherited and the two native stream adapters package-owned."""
    assert ZaiAdapter.__bases__ == (OpenAICompatibleAdapter,)


@pytest.mark.unit
def test_zai_native_stream_entry_points_remain_package_owned():
    """The native sync and internal async generators adapt Core lifecycle services."""
    assert "stream_chat" in ZaiAdapter.__dict__
    assert "_astream_chat" in ZaiAdapter.__dict__


@pytest.mark.unit
@pytest.mark.parametrize("method_name", ["chat", "achat", "astream_chat"])
def test_zai_native_methods_are_directly_inherited(method_name):
    """Prove each common method has no Z.ai wrapper, alias, or copied definition."""
    assert method_name not in ZaiAdapter.__dict__
    assert getattr(ZaiAdapter, method_name) is getattr(OpenAICompatibleAdapter, method_name)


@pytest.mark.unit
def test_zai_package_owned_stream_methods_delegate_to_shared_lifecycle(zai_adapter_runtime, monkeypatch):
    """Observe existing lifecycle helpers while retaining the real native stream state."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=ZAI_MODELS[0])
    adapter._sync_transport = RecordingTransport(events=[SSEEvent(None, data=_stream_payload(ZAI_MODELS[0]))])
    sync_calls = []
    async_calls = []
    original_sync = LLMAdapterBase._run_sync_stream
    original_async = LLMAdapterBase._run_async_stream

    def sync_spy(self, *args, **kwargs):
        sync_calls.append((self, args, kwargs))
        return original_sync(self, *args, **kwargs)

    async def async_spy(self, *args, **kwargs):
        async_calls.append((self, args, kwargs))
        async for item in original_async(self, *args, **kwargs):
            yield item

    async def async_stream(url, **kwargs):
        yield SSEEvent(None, data=_stream_payload(ZAI_MODELS[0]))

    monkeypatch.setattr(LLMAdapterBase, "_run_sync_stream", sync_spy)
    monkeypatch.setattr(LLMAdapterBase, "_run_async_stream", async_spy)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    assert list(adapter.stream_chat([UserMessage("sync lifecycle")])) == ["synthetic answer"]

    async def consume_async():
        return [text async for text in adapter.astream_chat([UserMessage("async lifecycle")])]

    assert asyncio.run(consume_async()) == ["synthetic answer"]
    assert len(sync_calls) == 1 and sync_calls[0][0] is adapter
    assert len(async_calls) == 1 and async_calls[0][0] is adapter
    sync_state = sync_calls[0][1][1]
    async_state = async_calls[0][1][1]
    assert type(sync_state) is _ZaiStreamState
    assert type(async_state) is _ZaiStreamState
    assert isinstance(sync_state.assembler, ZaiStreamAssembler)
    assert isinstance(async_state.assembler, ZaiStreamAssembler)


@pytest.mark.unit
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
def test_zai_unknown_native_model_fails_capability_preflight_before_io(
    zai_adapter_runtime, monkeypatch, method_kind
):
    """An unknown native identifier receives no inferred model capabilities."""
    with pytest.warns(UserWarning, match="not verified"):
        adapter = ZaiAdapter(api_key="synthetic-zai-key", model="unregistered-zai-model")
    assert adapter.model_spec is None
    transport = RecordingTransport()
    adapter._sync_transport = transport
    async_calls = []

    async def async_request(url, **kwargs):
        async_calls.append((url, kwargs))
        return _response_payload("unregistered-zai-model")

    async def async_stream(url, **kwargs):
        async_calls.append((url, kwargs))
        yield SSEEvent(None, data=_stream_payload("unregistered-zai-model"))

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    if method_kind == "chat":
        with pytest.raises(NotImplementedError):
            adapter.chat([UserMessage("unknown model")])
    elif method_kind == "achat":
        with pytest.raises(NotImplementedError):
            asyncio.run(adapter.achat([UserMessage("unknown model")]))
    elif method_kind == "stream_chat":
        with pytest.raises(NotImplementedError):
            list(adapter.stream_chat([UserMessage("unknown model")]))
    else:
        async def consume():
            async for _ in adapter.astream_chat([UserMessage("unknown model")]):
                pass
        with pytest.raises(NotImplementedError):
            asyncio.run(consume())
    assert transport.requests == [] and async_calls == []


@pytest.mark.unit
@pytest.mark.parametrize("config", [{"base_url": "https://example.test"}, {"api_variant": "chat_completions"}])
def test_zai_facade_rejects_native_connection_overrides_before_io(
    zai_adapter_runtime, config
):
    """Compatible-only connection settings fail at native facade configuration."""
    with pytest.raises(LLMConfigError):
        UniversalLLMAPIAdapter(
            organization="zai", model=ZAI_MODELS[0], api_key="synthetic-zai-key", **config
        )


_NO_REASONING_ARGUMENT = object()


def _reasoning_response_payload(model, reasoning_content="synthetic private thought"):
    message = {"role": "assistant", "content": "synthetic visible answer"}
    if reasoning_content is not _NO_REASONING_ARGUMENT:
        message["reasoning_content"] = reasoning_content
    return {
        "id": "synthetic-reasoning-response",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


def _reasoning_stream_events(model, reasoning_content="synthetic private thought"):
    delta = {"content": "synthetic visible answer"}
    if reasoning_content is not _NO_REASONING_ARGUMENT:
        delta["reasoning_content"] = reasoning_content
    return [
        SSEEvent(None, data={
            "id": "synthetic-reasoning-stream",
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}],
        }),
    ]


def _run_native_reasoning_request(
    monkeypatch,
    model,
    method_kind,
    *,
    reasoning_level=_NO_REASONING_ARGUMENT,
    capture_reasoning=False,
    reasoning_content="synthetic private thought",
):
    """Run a real native method with controlled transport I/O and return evidence."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    transport = RecordingTransport(
        body=_reasoning_response_payload(model, reasoning_content),
        events=_reasoning_stream_events(model, reasoning_content),
    )
    adapter._sync_transport = transport
    async_sends = []

    async def async_request(url, **kwargs):
        async_sends.append(("chat", url, kwargs))
        return _reasoning_response_payload(model, reasoning_content)

    async def async_stream_request(url, **kwargs):
        async_sends.append(("stream", url, kwargs))
        for event in _reasoning_stream_events(model, reasoning_content):
            yield event

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream_request)
    kwargs = {"capture_reasoning": capture_reasoning}
    if reasoning_level is not _NO_REASONING_ARGUMENT:
        kwargs["reasoning_level"] = reasoning_level
    completed = []
    reasoning_callbacks = []
    if method_kind in {"stream_chat", "astream_chat"}:
        kwargs["on_done"] = completed.append
        kwargs["on_reasoning"] = lambda event: reasoning_callbacks.append(event)

    if method_kind == "chat":
        response = adapter.chat([UserMessage("reasoning contract")], **kwargs)
    elif method_kind == "achat":
        response = asyncio.run(adapter.achat([UserMessage("reasoning contract")], **kwargs))
    elif method_kind == "stream_chat":
        visible = list(adapter.stream_chat([UserMessage("reasoning contract")], **kwargs))
        assert len(completed) == 1
        response = completed[0]
    else:
        async def consume():
            return [text async for text in adapter.astream_chat(
                [UserMessage("reasoning contract")], **kwargs
            )]
        visible = asyncio.run(consume())
        assert len(completed) == 1
        response = completed[0]

    if method_kind in {"chat", "stream_chat"}:
        assert len(transport.requests) == 1
        payload = transport.requests[0].payload
        sends = len(transport.requests)
    else:
        assert len(async_sends) == 1
        payload = async_sends[0][2]["payload"]
        sends = len(async_sends)
    return response, payload, sends, visible if method_kind in {
        "stream_chat", "astream_chat"
    } else None, reasoning_callbacks


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
def test_zai_reasoning_controls_follow_registered_values_and_resolver(
    zai_adapter_runtime, monkeypatch, model, method_kind
):
    """Cover omitted, native, canonical, and context-scaled levels on each entry point."""
    model_spec = zai_adapter_runtime.organizations["zai"].models[model]
    capability = model_spec.reasoning_capability
    registered_values = capability.allowed_values
    context = model_spec.limits.context_window_tokens
    working_values = tuple(value for value in registered_values if value != "none")
    minimum = working_values[0]
    maximum = working_values[-1]

    for reasoning_level in (_NO_REASONING_ARGUMENT, None):
        payloads = {}
        for capture in (False, True):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                response, payload, sends, visible, callbacks = _run_native_reasoning_request(
                    monkeypatch,
                    model,
                    method_kind,
                    reasoning_level=reasoning_level,
                    capture_reasoning=capture,
                    reasoning_content=_NO_REASONING_ARGUMENT,
                )
            assert sends == 1 and caught == []
            assert "thinking" not in payload and "reasoning_effort" not in payload
            assert "capture_reasoning" not in payload
            assert response.reasoning_events == []
            assert response.content == "synthetic visible answer"
            if method_kind in {"stream_chat", "astream_chat"}:
                assert visible == ["synthetic visible answer"] and callbacks == []
            payloads[capture] = copy.deepcopy(payload)
        assert payloads[False] == payloads[True]

    category_cutpoints = tuple(
        (
            (index * context) // len(working_values),
            working_values[index - 1],
            working_values[index],
        )
        for index in range(1, len(working_values))
    )
    numeric_inputs = tuple(dict.fromkeys((
        0,
        1,
        context - 1,
        context,
        context + 1,
        *(point for cutpoint, _, _ in category_cutpoints for point in (cutpoint, cutpoint + 1)),
    )))
    inputs = (*registered_values, *CANONICAL_REASONING_LEVELS, *numeric_inputs)
    resolved_payloads = {}
    for value in inputs:
        resolution = resolve_reasoning_level(model_spec, value)
        payloads = {}
        for capture in (False, True):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                response, payload, sends, visible, callbacks = _run_native_reasoning_request(
                    monkeypatch,
                    model,
                    method_kind,
                    reasoning_level=value,
                    capture_reasoning=capture,
                )
            assert sends == 1
            assert [str(item.message) for item in caught] == (
                [resolution.warning] if resolution.warning else []
            )
            assert all(issubclass(item.category, UserWarning) for item in caught)
            assert payload["thinking"] == {"type": "enabled"}
            assert payload["reasoning_effort"] == resolution.provider_value
            assert "capture_reasoning" not in payload
            assert response.content == "synthetic visible answer"
            assert "synthetic private thought" not in response.content
            expected_reasoning = ["synthetic private thought"] if capture else []
            assert [event.text for event in response.reasoning_events] == expected_reasoning
            if method_kind in {"stream_chat", "astream_chat"}:
                assert visible == ["synthetic visible answer"]
                assert [event.text for event in callbacks] == expected_reasoning
            if value in registered_values:
                assert resolution.reason == "categorical_direct_match"
                assert payload["reasoning_effort"] == value
            payloads[capture] = copy.deepcopy(payload)
        assert payloads[False] == payloads[True]
        resolved_payloads[value] = payloads[False]

    assert resolved_payloads["none"]["reasoning_effort"] == minimum
    assert resolved_payloads["minimal"]["reasoning_effort"] == minimum
    assert resolved_payloads[0]["reasoning_effort"] == minimum
    assert resolved_payloads[1]["reasoning_effort"] == minimum
    assert resolved_payloads[CANONICAL_REASONING_LEVELS[-1]]["reasoning_effort"] == maximum
    assert resolved_payloads[context]["reasoning_effort"] == maximum
    assert resolved_payloads[context + 1]["reasoning_effort"] == maximum
    for cutpoint, lower_value, upper_value in category_cutpoints:
        assert resolved_payloads[cutpoint]["reasoning_effort"] == lower_value
        assert resolved_payloads[cutpoint + 1]["reasoning_effort"] == upper_value


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
def test_zai_reasoning_none_warns_and_uses_registered_minimum(
    zai_adapter_runtime, monkeypatch, model, method_kind
):
    """The explicit string none preserves native always-on minimum-effort behavior."""
    model_spec = zai_adapter_runtime.organizations["zai"].models[model]
    minimum = next(
        item for item in model_spec.reasoning_capability.allowed_values if item != "none"
    )
    with pytest.warns(UserWarning, match=f"{model}.*minimum supported level.*{minimum}") as caught:
        response, payload, sends, _, _ = _run_native_reasoning_request(
            monkeypatch, model, method_kind, reasoning_level="none"
        )
    assert len(caught) == 1 and sends == 1
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == minimum
    assert "synthetic private thought" not in response.content


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
def test_zai_invalid_reasoning_fails_before_sync_or_async_send(
    zai_adapter_runtime, monkeypatch, model, method_kind
):
    """Bad reasoning controls retain the inherited public error and reject locally."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    transport = RecordingTransport()
    adapter._sync_transport = transport
    async_sends = []

    async def async_request(url, **kwargs):
        async_sends.append((url, kwargs))
        return _response_payload(model)

    async def async_stream_request(url, **kwargs):
        async_sends.append((url, kwargs))
        yield SSEEvent(None, data=_stream_payload(model))

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream_request)
    invalid_values = (True, False, -1, 0.5, "", "unregistered reasoning")
    for value in invalid_values:
        if method_kind == "chat":
            invoke = lambda: adapter.chat([UserMessage("invalid")], reasoning_level=value)
        elif method_kind == "achat":
            invoke = lambda: asyncio.run(
                adapter.achat([UserMessage("invalid")], reasoning_level=value)
            )
        elif method_kind == "stream_chat":
            invoke = lambda: list(
                adapter.stream_chat([UserMessage("invalid")], reasoning_level=value)
            )
        else:
            async def consume(value=value):
                async for _ in adapter.astream_chat(
                    [UserMessage("invalid")], reasoning_level=value
                ):
                    pass
            invoke = lambda: asyncio.run(consume())
        with pytest.raises(LLMReasoningLevelError) as caught:
            invoke()
        assert type(caught.value) is LLMReasoningLevelError
        assert transport.requests == [] and async_sends == []


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
def test_zai_reasoning_capture_is_opt_in_separate_and_wire_neutral(
    zai_adapter_runtime, monkeypatch, model, method_kind
):
    """Capture only exposes provider text; it never changes visible text or request controls."""
    model_spec = zai_adapter_runtime.organizations["zai"].models[model]
    provider_effort = model_spec.reasoning_capability.allowed_values[0]
    results = {}
    for capture in (False, True):
        response, payload, sends, visible, callbacks = _run_native_reasoning_request(
            monkeypatch,
            model,
            method_kind,
            reasoning_level=provider_effort,
            capture_reasoning=capture,
        )
        results[capture] = payload
        assert sends == 1
        assert response.content == "synthetic visible answer"
        assert "synthetic private thought" not in response.content
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == provider_effort
        if method_kind in {"stream_chat", "astream_chat"}:
            assert visible == ["synthetic visible answer"]
            assert [event.text for event in callbacks] == (
                ["synthetic private thought"] if capture else []
            )
        assert [event.text for event in response.reasoning_events] == (
            ["synthetic private thought"] if capture else []
        )
    assert results[False] == results[True]
    assert "capture_reasoning" not in results[True]


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
@pytest.mark.parametrize("reported_reasoning", [_NO_REASONING_ARGUMENT, None, ""])
def test_zai_capture_does_not_synthesize_missing_reasoning(
    zai_adapter_runtime, monkeypatch, model, method_kind, reported_reasoning
):
    """Opt-in capture leaves absent, null, and empty provider reasoning unavailable."""
    model_spec = zai_adapter_runtime.organizations["zai"].models[model]
    provider_effort = model_spec.reasoning_capability.allowed_values[0]
    response, payload, sends, visible, callbacks = _run_native_reasoning_request(
        monkeypatch,
        model,
        method_kind,
        reasoning_level=provider_effort,
        capture_reasoning=True,
        reasoning_content=reported_reasoning,
    )
    assert sends == 1
    assert response.reasoning_events == []
    assert "synthetic private thought" not in response.content
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == provider_effort
    if method_kind in {"stream_chat", "astream_chat"}:
        assert visible == ["synthetic visible answer"]
        assert callbacks == []


def _overlap_stream_events(model, label):
    """Build unique tool, reasoning-only, and mixed events for one request."""
    first_delta = {
        "tool_calls": [{
            "index": 0,
            "id": f"call-{label}",
            "type": "function",
            "function": {"name": f"lookup_{label}", "arguments": '{"key":"' + label + '"}'},
        }],
    }
    initial_visible = {"A": "A1", "B": "Bee", "C": "none"}[label]
    resumed_partial_visible = {"A": "a", "B": "x", "C": "!"}[label]
    resumed_final_visible = {"A": "2", "B": "Y", "C": "?"}[label]
    return [
        SSEEvent(None, data={"id": f"response-{label}", "model": model,
            "choices": [{"index": 0, "delta": first_delta, "finish_reason": None}]}),
        SSEEvent(None, data={"id": f"response-{label}", "model": model,
            "choices": [{"index": 0, "delta": {"reasoning_content": f"{label}-thought-1"}, "finish_reason": None}]}),
        SSEEvent(None, data={"id": f"response-{label}", "model": model,
            "choices": [{"index": 0, "delta": {"content": initial_visible}, "finish_reason": None}]}),
        SSEEvent(None, data={"id": f"response-{label}", "model": model,
            "choices": [{"index": 0, "delta": {
                "reasoning_content": f"{label}-thought-2",
                "content": resumed_partial_visible,
            }, "finish_reason": None}]}),
        SSEEvent(None, data={"id": f"response-{label}", "model": model,
            "choices": [{"index": 0, "delta": {"content": resumed_final_visible}, "finish_reason": "tool_calls"}]}),
    ]


def _overlap_tool(label):
    return ToolSpec(
        name=f"lookup_{label}",
        description="synthetic overlap tool",
        json_schema={"type": "object", "properties": {"key": {"type": "string"}}},
    )


def _assert_adapter_attributes_unchanged(adapter, original_items):
    """Check every adapter attribute and retained value identity, without name guesses."""
    current = vars(adapter)
    assert current.keys() == original_items.keys()
    assert all(current[name] is value for name, value in original_items.items())


def _overlap_callbacks(callbacks):
    """Record shared callback order while retaining each callback's own capture list."""
    def on_chunk(chunk):
        callbacks["chunks"].append(chunk)
        callbacks["order"].append(("chunk", chunk.text))

    def on_delta(text):
        callbacks["deltas"].append(text)
        callbacks["order"].append(("delta", text))

    def on_reasoning(event):
        callbacks["reasoning"].append(event)
        callbacks["order"].append(("reasoning", event.text))

    def on_tool_call(call):
        callbacks["tools"].append(call)
        callbacks["order"].append(("tool", call.call_id))

    def on_done(response):
        callbacks["done"].append(response)
        callbacks["order"].append(("done", response.response_id))

    return on_chunk, on_delta, on_reasoning, on_tool_call, on_done


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
def test_zai_sync_streams_keep_reasoning_tools_and_callbacks_request_local(
    zai_adapter_runtime, monkeypatch, model
):
    """Interleave native streams while preserving all adapter attribute identities."""
    effort = zai_adapter_runtime.organizations["zai"].models[model].reasoning_capability.allowed_values[0]
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)

    class InterleavedTransport(RecordingTransport):
        def post_sse(self, request, *, http_error_handler=None, stream_error_handler=None):
            self.requests.append(request)
            label = request.payload["messages"][0]["content"]
            def events():
                yield from _overlap_stream_events(model, label)
            return events()

    adapter._sync_transport = InterleavedTransport()
    original_state_factory = ZaiAdapter._new_stream_state
    states = []

    def retain_state(*, buffer_chars, capture_reasoning):
        state = original_state_factory(
            buffer_chars=buffer_chars, capture_reasoning=capture_reasoning
        )
        states.append(state)
        return state

    monkeypatch.setattr(ZaiAdapter, "_new_stream_state", staticmethod(retain_state))
    original_items = dict(vars(adapter))
    observed = {label: {"chunks": [], "deltas": [], "reasoning": [], "tools": [], "done": [], "order": []}
                for label in "ABC"}
    iterators = {}
    for label, capture, limit in (("A", True, 2), ("B", True, 3), ("C", False, 4)):
        callbacks = observed[label]
        on_chunk, on_delta, on_reasoning, on_tool_call, on_done = _overlap_callbacks(callbacks)
        iterators[label] = adapter.stream_chat(
            [UserMessage(label)],
            reasoning_level=effort,
            tools=[_overlap_tool(label)],
            capture_reasoning=capture,
            buffer_chars=limit,
            on_chunk=on_chunk,
            on_delta=on_delta,
            on_reasoning=on_reasoning,
            on_tool_call=on_tool_call,
            on_done=on_done,
        )

    # First yields happen before thought-2 is sent, leaving the other streams suspended.
    assert next(iterators["A"]) == "A1"
    _assert_adapter_attributes_unchanged(adapter, original_items)
    assert next(iterators["B"]) == "Bee"
    _assert_adapter_attributes_unchanged(adapter, original_items)
    assert next(iterators["C"]) == "none"
    _assert_adapter_attributes_unchanged(adapter, original_items)
    assert len(states) == 3
    assert all(type(state) is _ZaiStreamState for state in states)
    assert len({id(state) for state in states}) == 3
    assert len({id(state.chunk_buffer) for state in states}) == 3
    assert len({id(state.usage_tracker) for state in states}) == 3
    assert len({id(state.assembler) for state in states}) == 3
    assert states[0].reasoning_collector is not states[1].reasoning_collector
    assert states[2].reasoning_collector is None

    assert [event.text for event in observed["A"]["reasoning"]] == ["A-thought-1"]
    assert [event.text for event in observed["B"]["reasoning"]] == ["B-thought-1"]
    assert observed["C"]["reasoning"] == []

    assert next(iterators["A"]) == "a2"
    _assert_adapter_attributes_unchanged(adapter, original_items)
    assert [event.text for event in observed["A"]["reasoning"]] == ["A-thought-1", "A-thought-2"]
    assert [event.text for event in observed["B"]["reasoning"]] == ["B-thought-1"]
    assert observed["C"]["reasoning"] == []
    assert next(iterators["B"]) == "xY"
    _assert_adapter_attributes_unchanged(adapter, original_items)
    assert [event.text for event in observed["B"]["reasoning"]] == ["B-thought-1", "B-thought-2"]
    assert observed["C"]["reasoning"] == []
    assert next(iterators["C"]) == "!?"
    _assert_adapter_attributes_unchanged(adapter, original_items)

    for label in "ABC":
        with pytest.raises(StopIteration):
            next(iterators[label])
        _assert_adapter_attributes_unchanged(adapter, original_items)

    for label, visible in (("A", "A1a2"), ("B", "BeexY"), ("C", "none!?")):
        callbacks = observed[label]
        expected_chunks = {"A": ["A1", "a2"], "B": ["Bee", "xY"], "C": ["none", "!?"]}[label]
        assert [chunk.text for chunk in callbacks["chunks"]] == expected_chunks
        assert callbacks["deltas"] == expected_chunks
        assert [call.call_id for call in callbacks["tools"]] == [f"call-{label}"]
        assert [(call.name, call.arguments) for call in callbacks["tools"]] == [
            (f"lookup_{label}", {"key": label})
        ]
        assert len(callbacks["done"]) == 1
        response = callbacks["done"][0]
        assert response.content == visible
        assert [call.call_id for call in response.tool_calls] == [f"call-{label}"]
        assert [(call.name, call.arguments) for call in response.tool_calls] == [
            (f"lookup_{label}", {"key": label})
        ]
        assert [event.text for event in callbacks["reasoning"]] == (
            [f"{label}-thought-1", f"{label}-thought-2"] if label != "C" else []
        )
        assert [event.text for event in response.reasoning_events] == (
            [f"{label}-thought-1", f"{label}-thought-2"] if label != "C" else []
        )
        assert callbacks["order"] == (
            [
                ("reasoning", f"{label}-thought-1"),
                ("chunk", expected_chunks[0]),
                ("delta", expected_chunks[0]),
                ("reasoning", f"{label}-thought-2"),
                ("chunk", expected_chunks[1]),
                ("delta", expected_chunks[1]),
                ("tool", f"call-{label}"),
                ("done", f"response-{label}"),
            ] if label != "C" else [
                ("chunk", expected_chunks[0]),
                ("delta", expected_chunks[0]),
                ("chunk", expected_chunks[1]),
                ("delta", expected_chunks[1]),
                ("tool", f"call-{label}"),
                ("done", f"response-{label}"),
            ]
        )
    _assert_adapter_attributes_unchanged(adapter, original_items)
    assert all(
        request.payload["reasoning_effort"] == effort
        and request.payload["thinking"] == {"type": "enabled"}
        and "capture_reasoning" not in request.payload
        for request in adapter._sync_transport.requests
    )


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
def test_zai_async_streams_keep_reasoning_tools_and_callbacks_request_local(
    zai_adapter_runtime, monkeypatch, model
):
    """Overlap native async streams behind a deterministic transport start barrier."""
    effort = zai_adapter_runtime.organizations["zai"].models[model].reasoning_capability.allowed_values[0]
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    original_state_factory = ZaiAdapter._new_stream_state
    states = []

    def retain_state(*, buffer_chars, capture_reasoning):
        state = original_state_factory(
            buffer_chars=buffer_chars, capture_reasoning=capture_reasoning
        )
        states.append(state)
        return state

    monkeypatch.setattr(ZaiAdapter, "_new_stream_state", staticmethod(retain_state))
    original_items = dict(vars(adapter))
    start_count = 0
    all_started = None
    sent_payloads = []

    async def async_stream(url, **kwargs):
        nonlocal start_count
        assert url == ZAI_CHAT_COMPLETIONS_URL
        label = kwargs["payload"]["messages"][0]["content"]
        sent_payloads.append(kwargs["payload"])
        start_count += 1
        if start_count == 3:
            all_started.set()
        await asyncio.wait_for(all_started.wait(), timeout=5)
        for event in _overlap_stream_events(model, label):
            yield event

    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    observed = {label: {"chunks": [], "deltas": [], "reasoning": [], "tools": [], "done": [], "order": []}
                for label in "ABC"}

    async def run_overlap():
        nonlocal all_started
        all_started = asyncio.Event()
        streams = {}
        for label, capture, limit in (("A", True, 2), ("B", True, 3), ("C", False, 4)):
            callbacks = observed[label]
            on_chunk, on_delta, on_reasoning, on_tool_call, on_done = _overlap_callbacks(callbacks)
            streams[label] = adapter.astream_chat(
                [UserMessage(label)],
                reasoning_level=effort,
                tools=[_overlap_tool(label)],
                capture_reasoning=capture,
                buffer_chars=limit,
                on_chunk=on_chunk,
                on_delta=on_delta,
                on_reasoning=on_reasoning,
                on_tool_call=on_tool_call,
                on_done=on_done,
            )
        first_chunks = await asyncio.gather(*(anext(streams[label]) for label in "ABC"))
        assert first_chunks == ["A1", "Bee", "none"]
        _assert_adapter_attributes_unchanged(adapter, original_items)
        assert start_count == 3 and len(states) == 3
        assert all(type(state) is _ZaiStreamState for state in states)
        assert len({id(state) for state in states}) == 3
        assert len({id(state.chunk_buffer) for state in states}) == 3
        assert len({id(state.usage_tracker) for state in states}) == 3
        assert len({id(state.assembler) for state in states}) == 3
        assert states[0].reasoning_collector is not states[1].reasoning_collector
        assert states[2].reasoning_collector is None
        assert [event.text for event in observed["A"]["reasoning"]] == ["A-thought-1"]
        assert [event.text for event in observed["B"]["reasoning"]] == ["B-thought-1"]
        assert observed["C"]["reasoning"] == []
        assert await anext(streams["A"]) == "a2"
        _assert_adapter_attributes_unchanged(adapter, original_items)
        assert [event.text for event in observed["A"]["reasoning"]] == ["A-thought-1", "A-thought-2"]
        assert [event.text for event in observed["B"]["reasoning"]] == ["B-thought-1"]
        assert observed["C"]["reasoning"] == []
        assert await anext(streams["B"]) == "xY"
        _assert_adapter_attributes_unchanged(adapter, original_items)
        assert [event.text for event in observed["B"]["reasoning"]] == ["B-thought-1", "B-thought-2"]
        assert observed["C"]["reasoning"] == []
        assert await anext(streams["C"]) == "!?"
        _assert_adapter_attributes_unchanged(adapter, original_items)
        for label in "ABC":
            with pytest.raises(StopAsyncIteration):
                await anext(streams[label])
            _assert_adapter_attributes_unchanged(adapter, original_items)

    asyncio.run(run_overlap())
    for label, visible in (("A", "A1a2"), ("B", "BeexY"), ("C", "none!?")):
        callbacks = observed[label]
        expected_chunks = {"A": ["A1", "a2"], "B": ["Bee", "xY"], "C": ["none", "!?"]}[label]
        assert [chunk.text for chunk in callbacks["chunks"]] == expected_chunks
        assert callbacks["deltas"] == expected_chunks
        assert [call.call_id for call in callbacks["tools"]] == [f"call-{label}"]
        assert [(call.name, call.arguments) for call in callbacks["tools"]] == [
            (f"lookup_{label}", {"key": label})
        ]
        assert len(callbacks["done"]) == 1
        response = callbacks["done"][0]
        assert response.content == visible
        assert [call.call_id for call in response.tool_calls] == [f"call-{label}"]
        assert [(call.name, call.arguments) for call in response.tool_calls] == [
            (f"lookup_{label}", {"key": label})
        ]
        assert [event.text for event in callbacks["reasoning"]] == (
            [f"{label}-thought-1", f"{label}-thought-2"] if label != "C" else []
        )
        assert [event.text for event in response.reasoning_events] == (
            [f"{label}-thought-1", f"{label}-thought-2"] if label != "C" else []
        )
        assert callbacks["order"] == (
            [
                ("reasoning", f"{label}-thought-1"),
                ("chunk", expected_chunks[0]),
                ("delta", expected_chunks[0]),
                ("reasoning", f"{label}-thought-2"),
                ("chunk", expected_chunks[1]),
                ("delta", expected_chunks[1]),
                ("tool", f"call-{label}"),
                ("done", f"response-{label}"),
            ] if label != "C" else [
                ("chunk", expected_chunks[0]),
                ("delta", expected_chunks[0]),
                ("chunk", expected_chunks[1]),
                ("delta", expected_chunks[1]),
                ("tool", f"call-{label}"),
                ("done", f"response-{label}"),
            ]
        )
    _assert_adapter_attributes_unchanged(adapter, original_items)
    assert len(sent_payloads) == 3
    assert all(
        payload["reasoning_effort"] == effort
        and payload["thinking"] == {"type": "enabled"}
        and "capture_reasoning" not in payload
        for payload in sent_payloads
    )


def _invoke_tool_method(adapter, method_kind, messages, kwargs):
    """Run one public method against the configured controlled transport."""
    method = getattr(adapter, method_kind)
    if method_kind == "chat":
        return method(messages, **kwargs)
    if method_kind == "stream_chat":
        return list(method(messages, **kwargs))
    if method_kind == "achat":
        return asyncio.run(method(messages, **kwargs))

    async def consume():
        return [item async for item in method(messages, **kwargs)]

    return asyncio.run(consume())


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
@pytest.mark.parametrize("description", [None, "Find a value."])
@pytest.mark.parametrize("selection", ["omitted", "auto"])
def test_zai_tools_are_declared_with_optional_description_and_auto_selection(
    zai_adapter_runtime, monkeypatch, model, method_kind, description, selection
):
    """All native methods preserve declarations and automatic-only selection."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    adapter._sync_transport = RecordingTransport(
        body=_response_payload(model),
        events=[SSEEvent(None, data=_stream_payload(model))],
    )
    async_calls = []

    async def async_request(url, **kwargs):
        async_calls.append((url, kwargs))
        return _response_payload(model)

    async def async_stream(url, **kwargs):
        async_calls.append((url, kwargs))
        yield SSEEvent(None, data=_stream_payload(model))

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    tool = ToolSpec(
        name="lookup_value",
        description=description,
        json_schema={"type": "object", "properties": {"key": {"type": "string"}}},
    )
    kwargs = {"tools": [tool]}
    if selection == "auto":
        kwargs["tool_choice"] = "auto"

    _invoke_tool_method(adapter, method_kind, [UserMessage("look it up")], kwargs)
    if method_kind in {"chat", "stream_chat"}:
        assert len(adapter._sync_transport.requests) == 1
        payload = adapter._sync_transport.requests[0].payload
    else:
        assert len(async_calls) == 1
        payload = async_calls[0][1]["payload"]
    expected_function = {
        "name": "lookup_value",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
    }
    if description is not None:
        expected_function["description"] = description
    expected = {
        "model": model,
        "messages": [{"role": "user", "content": "look it up"}],
        "tools": [{"type": "function", "function": expected_function}],
    }
    if selection == "auto":
        expected["tool_choice"] = "auto"
    if method_kind in {"stream_chat", "astream_chat"}:
        expected.update({"stream": True, "stream_options": {"include_usage": True}})
    assert payload == expected


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
@pytest.mark.parametrize(("count", "accepted"), [(128, True), (129, False)])
def test_zai_tool_count_boundary_applies_to_every_native_method(
    zai_adapter_runtime, monkeypatch, model, method_kind, count, accepted
):
    """The existing 128 declaration limit is shared across all native entry points."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    adapter._sync_transport = RecordingTransport(
        body=_response_payload(model),
        events=[SSEEvent(None, data=_stream_payload(model))],
    )
    async_calls = []

    async def async_request(url, **kwargs):
        async_calls.append((url, kwargs))
        return _response_payload(model)

    async def async_stream(url, **kwargs):
        async_calls.append((url, kwargs))
        yield SSEEvent(None, data=_stream_payload(model))

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    tools = [ToolSpec(name=f"tool_{index}", json_schema={"type": "object"}) for index in range(count)]
    kwargs = {"tools": tools}
    if accepted:
        _invoke_tool_method(adapter, method_kind, [UserMessage("choose")], kwargs)
        if method_kind in {"chat", "stream_chat"}:
            assert len(adapter._sync_transport.requests) == 1
            payload = adapter._sync_transport.requests[0].payload
        else:
            assert len(async_calls) == 1
            payload = async_calls[0][1]["payload"]
        assert payload["tools"] == [
            {"type": "function", "function": {
                "name": f"tool_{index}", "parameters": {"type": "object"}
            }}
            for index in range(128)
        ]
    else:
        with pytest.raises(ValueError, match="at most 128"):
            _invoke_tool_method(adapter, method_kind, [UserMessage("choose")], kwargs)
        assert adapter._sync_transport.requests == []
        assert async_calls == []


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
def test_zai_completed_tool_calls_and_full_caller_history_are_preserved(
    zai_adapter_runtime, monkeypatch, model, method_kind
):
    """A caller builds the next turn from real tool results and optional response metadata."""
    response_payload = {
        "id": "tool-response-id", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
            "tool_calls": [
                {"id": "call-weather-7", "type": "function", "function": {
                    "name": "get_weather", "arguments": '{"city":"Paris"}'
                }},
                {"id": "call-add-9", "type": "function", "function": {
                    "name": "add", "arguments": '{"left":2,"right":3}'
                }},
            ]}, "finish_reason": "tool_calls"}],
    }
    second_response_payload = _response_payload(model)
    stream_tool_events = [
        SSEEvent(None, data={"id": "stream-tool-response", "model": model, "choices": [{
            "index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call-weather-7", "type": "function", "function": {
                    "name": "get_weather", "arguments": '{"city":'
                }},
                {"index": 1, "id": "call-add-9", "type": "function", "function": {
                    "name": "add", "arguments": '{"left":'
                }},
            ]}, "finish_reason": None,
        }]}),
        SSEEvent(None, data={"id": "stream-tool-response", "model": model, "choices": [{
            "index": 0, "delta": {"tool_calls": [
                {"index": 1, "function": {"arguments": '2,"right":3}'}},
                {"index": 0, "function": {"arguments": '"Paris"}'}},
            ]}, "finish_reason": "tool_calls",
        }]})
    ]
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    adapter._sync_transport = RecordingTransport(
        body=response_payload, events=stream_tool_events
    )
    async_calls = []
    async_stream_count = 0

    async def async_request(url, **kwargs):
        async_calls.append((url, kwargs))
        return response_payload if len(async_calls) == 1 else second_response_payload

    async def async_stream(url, **kwargs):
        nonlocal async_stream_count
        async_calls.append((url, kwargs))
        async_stream_count += 1
        events = stream_tool_events if async_stream_count == 1 else [
            SSEEvent(None, data=_stream_payload(model))
        ]
        for event in events:
            yield event

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    tool_specs = [
        ToolSpec(name="get_weather", description="Read the weather.",
            json_schema={"type": "object", "properties": {"city": {"type": "string"}}}),
        ToolSpec(name="add", json_schema={"type": "object", "properties": {
            "left": {"type": "integer"}, "right": {"type": "integer"}
        }}),
    ]
    declarations = [
        {"type": "function", "function": {
            "name": "get_weather", "description": "Read the weather.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }},
        {"type": "function", "function": {
            "name": "add", "parameters": {"type": "object", "properties": {
                "left": {"type": "integer"}, "right": {"type": "integer"}
            }},
        }},
    ]
    first_messages = [UserMessage("What is the weather in Paris and add 2 + 3?")]
    first_kwargs = {"tools": tool_specs, "tool_choice": "auto"}
    if method_kind in {"chat", "achat"}:
        tool_response = _invoke_tool_method(adapter, method_kind, first_messages, first_kwargs)
    else:
        completed = []
        visible = _invoke_tool_method(
            adapter, method_kind, first_messages, {**first_kwargs, "on_done": completed.append}
        )
        assert visible == [] and len(completed) == 1
        tool_response = completed[0]
    if method_kind in {"chat", "stream_chat"}:
        first_payload = adapter._sync_transport.requests[0].payload
        assert len(adapter._sync_transport.requests) == 1
    else:
        first_payload = async_calls[0][1]["payload"]
        assert len(async_calls) == 1
    assert first_payload["tools"] == declarations and first_payload["tool_choice"] == "auto"
    assert [
        (call.name, call.arguments, call.call_id) for call in tool_response.tool_calls
    ] == [
        ("get_weather", {"city": "Paris"}, "call-weather-7"),
        ("add", {"left": 2, "right": 3}, "call-add-9"),
    ]
    tool_response.provider_data = {"opaque": "provider-replay-sentinel"}
    assert tool_response.response_id == (
        "tool-response-id" if method_kind in {"chat", "achat"} else "stream-tool-response"
    )
    history = [
        *first_messages,
        AIMessage(content="", tool_calls=tool_response.tool_calls),
        ToolMessage(content='{"temperature_c": 18}', tool_call_id="call-weather-7"),
        ToolMessage(content="5", tool_call_id="call-add-9"),
        UserMessage("Use the results and keep this history."),
    ]
    adapter._sync_transport.body = second_response_payload
    adapter._sync_transport.events = [SSEEvent(None, data=_stream_payload(model))]
    _invoke_tool_method(
        adapter, method_kind, history,
        {**first_kwargs, "previous_response": tool_response},
    )
    if method_kind in {"chat", "stream_chat"}:
        assert len(adapter._sync_transport.requests) == 2
        payload_with_previous = copy.deepcopy(adapter._sync_transport.requests[1].payload)
    else:
        assert len(async_calls) == 2
        payload_with_previous = copy.deepcopy(async_calls[1][1]["payload"])
    _invoke_tool_method(adapter, method_kind, history, first_kwargs)
    if method_kind in {"chat", "stream_chat"}:
        assert len(adapter._sync_transport.requests) == 3
        payload_without_previous = adapter._sync_transport.requests[2].payload
    else:
        assert len(async_calls) == 3
        payload_without_previous = async_calls[2][1]["payload"]
    assert payload_with_previous == payload_without_previous
    assert payload_with_previous["messages"] == [
        {"role": "user", "content": "What is the weather in Paris and add 2 + 3?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-weather-7", "type": "function", "function": {
                "name": "get_weather", "arguments": '{"city": "Paris"}'
            }},
            {"id": "call-add-9", "type": "function", "function": {
                "name": "add", "arguments": '{"left": 2, "right": 3}'
            }},
        ]},
        {"role": "tool", "tool_call_id": "call-weather-7", "content": '{"temperature_c": 18}'},
        {"role": "tool", "tool_call_id": "call-add-9", "content": "5"},
        {"role": "user", "content": "Use the results and keep this history."},
    ]
    assert payload_with_previous["tools"] == declarations
    assert payload_with_previous["tool_choice"] == "auto"
    expected_keys = {"model", "messages", "tools", "tool_choice"}
    if method_kind in {"stream_chat", "astream_chat"}:
        expected_keys.update({"stream", "stream_options"})
    assert set(payload_with_previous) == expected_keys


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["stream_chat", "astream_chat"])
def test_zai_fragmented_tool_arguments_keep_call_index_associations(
    zai_adapter_runtime, monkeypatch, model, method_kind
):
    """One consumed stream delivers associated calls before its single final callback."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    sync_events = [
        SSEEvent(None, data={"id": "fragmented-id", "model": model, "choices": [
            {"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call-a", "type": "function", "function": {"name": "weather", "arguments": '{"city":'}},
                {"index": 1, "id": "call-b", "type": "function", "function": {"name": "add", "arguments": '{"left":'}},
            ]}, "finish_reason": None}]}),
        SSEEvent(None, data={"id": "fragmented-id", "model": model, "choices": [
            {"index": 0, "delta": {"tool_calls": [
                {"index": 1, "function": {"arguments": "2,\"right\":3}"}},
                {"index": 0, "function": {"arguments": "\"Paris\"}"}},
            ]}, "finish_reason": "tool_calls"}]})]
    adapter._sync_transport = RecordingTransport(events=sync_events)
    async_calls = []

    async def async_stream(url, **kwargs):
        async_calls.append((url, kwargs))
        for event in sync_events:
            yield event

    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    tools = [
        ToolSpec(name="weather", json_schema={"type": "object", "properties": {
            "city": {"type": "string"}
        }}),
        ToolSpec(name="add", json_schema={"type": "object", "properties": {
            "left": {"type": "integer"}, "right": {"type": "integer"}
        }}),
    ]
    callbacks = []
    kwargs = {
        "tools": tools,
        "on_tool_call": lambda call: callbacks.append(("tool", call)),
        "on_done": lambda response: callbacks.append(("done", response)),
    }
    if method_kind == "stream_chat":
        visible = list(adapter.stream_chat([UserMessage("run both")], **kwargs))
    else:
        async def consume():
            return [item async for item in adapter.astream_chat([UserMessage("run both")], **kwargs)]
        visible = asyncio.run(consume())
    assert visible == []
    if method_kind == "stream_chat":
        assert len(adapter._sync_transport.requests) == 1
        sent_payload = adapter._sync_transport.requests[0].payload
    else:
        assert len(async_calls) == 1
        sent_payload = async_calls[0][1]["payload"]
    assert sent_payload["tools"] == [
        {"type": "function", "function": {
            "name": "weather", "parameters": {"type": "object", "properties": {
                "city": {"type": "string"}
            }}
        }},
        {"type": "function", "function": {
            "name": "add", "parameters": {"type": "object", "properties": {
                "left": {"type": "integer"}, "right": {"type": "integer"}
            }}
        }},
    ]
    assert [kind for kind, _ in callbacks] == ["tool", "tool", "done"]
    calls = [call for kind, call in callbacks if kind == "tool"]
    completed = [response for kind, response in callbacks if kind == "done"]
    assert len(completed) == 1
    expected_calls = [
        ("weather", {"city": "Paris"}, "call-a"),
        ("add", {"left": 2, "right": 3}, "call-b"),
    ]
    assert [(item.name, item.arguments, item.call_id) for item in calls] == expected_calls
    assert [(item.name, item.arguments, item.call_id) for item in completed[0].tool_calls] == expected_calls
    assert completed[0].response_id == "fragmented-id"


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
@pytest.mark.parametrize(
    ("case_name", "choice", "parallel", "error_type"),
    [
        ("none_string", "none", None, ToolChoiceError),
        ("any_string", "any", None, ToolChoiceError),
        ("required_string", "required", None, ToolChoiceError),
        ("none_dict", {"type": "none"}, None, ToolChoiceError),
        ("any_dict", {"type": "any"}, None, ToolChoiceError),
        ("known_named_string", "known_tool", None, ToolChoiceError),
        ("known_named_dict", {"type": "tool", "name": "known_tool"}, None, ToolChoiceError),
        ("missing_named_string", "missing_tool", None, ToolChoiceError),
        ("missing_named_dict", {"type": "tool", "name": "missing_tool"}, None, ToolChoiceError),
        ("parallel_false", None, False, NotImplementedError),
        ("parallel_true", None, True, NotImplementedError),
    ],
)
def test_zai_unsupported_tool_controls_fail_before_any_send(
    zai_adapter_runtime, monkeypatch, model, method_kind, case_name,
    choice, parallel, error_type
):
    """Rule-rejected choices and either parallel value fail before each transport."""
    model_spec = zai_adapter_runtime.organizations["zai"].models[model]
    tool_choice_rule = next(
        rule for rule in model_spec.request_rules.rules
        if rule.handler == "restrict_tool_choice"
    )
    allowed_choices = set(tool_choice_rule.arguments["allowed_values"])
    assert "auto" in allowed_choices
    if case_name not in {"parallel_false", "parallel_true"}:
        if isinstance(choice, str):
            assert choice not in allowed_choices
        else:
            assert choice["type"] not in allowed_choices
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    adapter._sync_transport = RecordingTransport(
        body=_response_payload(model),
        events=[SSEEvent(None, data=_stream_payload(model))],
    )
    async_calls = []

    async def async_request(url, **kwargs):
        async_calls.append((url, kwargs))
        return _response_payload(model)

    async def async_stream(url, **kwargs):
        async_calls.append((url, kwargs))
        yield SSEEvent(None, data=_stream_payload(model))

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    kwargs = {"tools": [ToolSpec(name="known_tool", json_schema={"type": "object"})]}
    if choice is not None:
        kwargs["tool_choice"] = choice
    if parallel is not None:
        kwargs["parallel_tool_calls"] = parallel
    with pytest.raises(error_type):
        _invoke_tool_method(adapter, method_kind, [UserMessage("invalid control")], kwargs)
    assert len(adapter._sync_transport.requests) == 0
    assert len(async_calls) == 0


class _PortableResponseModel(BaseModel):
    answer: str


_IMAGE_CASES = (
    "typed_url",
    "typed_png_bytes",
    "typed_webp_bytes",
    "typed_data_url",
    "mixed_typed_and_dictionary",
)


def _image_case_messages(case_name):
    """Return an image conversation and independently declared wire messages."""
    png_data = b"png-image-fixture"
    webp_data = b"webp-image-fixture"
    if case_name == "typed_url":
        return [
            UserMessage("Describe this chart.", files=[
                ImagePart(url="https://images.example/chart.png")
            ])
        ], [{"role": "user", "content": [
            {"type": "text", "text": "Describe this chart."},
            {"type": "image_url", "image_url": {
                "url": "https://images.example/chart.png"
            }},
        ]}]
    if case_name == "typed_png_bytes":
        return [
            UserMessage("Read the label.", files=[
                ImagePart(data=png_data, media_type="image/png")
            ])
        ], [{"role": "user", "content": [
            {"type": "text", "text": "Read the label."},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64,cG5nLWltYWdlLWZpeHR1cmU="
            }},
        ]}]
    if case_name == "typed_webp_bytes":
        return [
            UserMessage("Compare this image.", files=[
                ImagePart(data=webp_data, media_type="image/webp")
            ])
        ], [{"role": "user", "content": [
            {"type": "text", "text": "Compare this image."},
            {"type": "image_url", "image_url": {
                "url": "data:image/webp;base64,d2VicC1pbWFnZS1maXh0dXJl"
            }},
        ]}]
    if case_name == "typed_data_url":
        data_url = "data:image/jpeg;base64,/9j/2Q=="
        return [UserMessage("Identify this photo.", files=[
            ImagePart(url=data_url)
        ])], [{"role": "user", "content": [
            {"type": "text", "text": "Identify this photo."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}]
    if case_name == "mixed_typed_and_dictionary":
        data_url = "data:image/webp;base64,bWl4ZWQtZGF0YQ=="
        return [
            {"role": "system", "content": "Use visual evidence."},
            UserMessage("First image: ", files=[ImagePart(
                url="https://images.example/first.webp", media_type="image/webp"
            )]),
            {"role": "user", "content": [
                {"type": "text", "text": "Second image: "},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
            {"role": "assistant", "content": "I will compare both."},
        ], [
            {"role": "system", "content": "Use visual evidence."},
            {"role": "user", "content": [
                {"type": "text", "text": "First image: "},
                {"type": "image_url", "image_url": {
                    "url": "https://images.example/first.webp"
                }},
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "Second image: "},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
            {"role": "assistant", "content": "I will compare both."},
        ]
    raise AssertionError(f"Unknown image case: {case_name}")


@dataclass
class _ImageTransferRecordingTransport(RecordingTransport):
    transfer_attempts: list[str] = field(default_factory=list)

    def post_multipart(self, request, form, *, http_error_handler=None):
        self.transfer_attempts.append("sync multipart upload")
        raise AssertionError("image request attempted a multipart upload")


def _guard_image_transfers(monkeypatch, transport):
    """Record and block HTTP-library transfers while these tests use I/O doubles."""
    attempts = transport.transfer_attempts

    def blocked(label):
        def record_attempt(*args, **kwargs):
            attempts.append(label)
            raise AssertionError(f"unexpected local file transfer through {label}")
        return record_attempt

    import requests

    monkeypatch.setattr(
        requests.sessions.Session, "request", blocked("requests.Session.request")
    )
    monkeypatch.setattr(
        requests.sessions.Session, "send", blocked("requests.Session.send")
    )
    monkeypatch.setattr(urllib.request, "urlopen", blocked("urllib.request.urlopen"))
    try:
        import httpx
    except ImportError:
        pass
    else:
        monkeypatch.setattr(httpx.Client, "send", blocked("httpx.Client.send"))
        monkeypatch.setattr(
            httpx.AsyncClient, "send", blocked("httpx.AsyncClient.send")
        )
    return attempts


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
@pytest.mark.parametrize("case_name", _IMAGE_CASES)
def test_zai_images_keep_typed_dictionary_and_mixed_wire_semantics(
    zai_adapter_runtime, monkeypatch, model, method_kind, case_name
):
    """Every native execution path forwards supported images without local transfer."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    adapter._sync_transport = _ImageTransferRecordingTransport(
        body=_response_payload(model), events=[SSEEvent(None, data=_stream_payload(model))]
    )
    transfer_attempts = _guard_image_transfers(monkeypatch, adapter._sync_transport)
    async_calls = []

    async def async_request(url, **kwargs):
        async_calls.append((url, kwargs))
        return _response_payload(model)

    async def async_stream(url, **kwargs):
        async_calls.append((url, kwargs))
        yield SSEEvent(None, data=_stream_payload(model))

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    messages, expected_messages = _image_case_messages(case_name)
    _invoke_tool_method(adapter, method_kind, messages, {})

    if method_kind in {"chat", "stream_chat"}:
        assert len(adapter._sync_transport.requests) == 1
        sent_request = adapter._sync_transport.requests[0]
        assert sent_request.url == ZAI_CHAT_COMPLETIONS_URL
        payload = sent_request.payload
        assert async_calls == []
    else:
        assert len(async_calls) == 1
        assert async_calls[0][0] == ZAI_CHAT_COMPLETIONS_URL
        payload = async_calls[0][1]["payload"]
        assert adapter._sync_transport.requests == []
    expected = {"model": model, "messages": expected_messages}
    if method_kind in {"stream_chat", "astream_chat"}:
        expected.update({"stream": True, "stream_options": {"include_usage": True}})
    assert payload == expected
    assert transfer_attempts == []


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
@pytest.mark.parametrize(
    ("case_name", "expected_error", "expected_message"),
    [
        ("pdf_url", ValueError, "does not support DocumentPart yet"),
        ("pdf_bytes", ValueError, "does not support DocumentPart yet"),
        ("document_dict_url", ValueError, "does not support DocumentPart yet"),
        ("document_dict_bytes", ValueError, "does not support DocumentPart yet"),
        ("nonpdf_url", ValueError, "FilePart is not supported as OpenAI Chat Completions file part"),
        ("nonpdf_bytes", ValueError, "FilePart is not supported as OpenAI Chat Completions file part"),
        ("portable_schema", NotImplementedError, "does not support portable structured output"),
        ("response_model", NotImplementedError, "does not support portable structured output"),
    ],
)
def test_zai_unsupported_files_and_structured_output_fail_before_any_send(
    zai_adapter_runtime, monkeypatch, model, method_kind, case_name,
    expected_error, expected_message
):
    """Unsupported inputs retain their native error and are rejected before I/O."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    adapter._sync_transport = _ImageTransferRecordingTransport(
        body=_response_payload(model), events=[SSEEvent(None, data=_stream_payload(model))]
    )
    transfer_attempts = _guard_image_transfers(monkeypatch, adapter._sync_transport)
    async_calls = []

    async def async_request(url, **kwargs):
        async_calls.append((url, kwargs))
        return _response_payload(model)

    async def async_stream(url, **kwargs):
        async_calls.append((url, kwargs))
        yield SSEEvent(None, data=_stream_payload(model))

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    cases = {
        "pdf_url": (UserMessage("inspect", files=[DocumentPart(
            url="https://docs.example/report.pdf"
        )]), {}),
        "pdf_bytes": (UserMessage("inspect", files=[DocumentPart(
            data=b"%PDF-1.7 synthetic", media_type="application/pdf"
        )]), {}),
        "document_dict_url": ({"role": "user", "content": "inspect", "files": [{
            "type": "document", "source": {"type": "url",
                "url": "https://docs.example/report.pdf"}
        }]}, {}),
        "document_dict_bytes": ({"role": "user", "content": "inspect", "files": [{
            "type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                "data": "JVBERi0xLjcgc3ludGhldGlj"}
        }]}, {}),
        "nonpdf_url": (UserMessage("inspect", files=[FilePart(
            url="https://docs.example/report.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )]), {}),
        "nonpdf_bytes": (UserMessage("inspect", files=[FilePart(
            data=b"synthetic document", media_type="application/msword"
        )]), {}),
        "portable_schema": (UserMessage("answer in JSON", files=[ImagePart(
            url="https://images.example/valid.png"
        )]), {"json_schema": {"type": "object", "properties": {
            "answer": {"type": "string"}
        }}}),
        "response_model": (UserMessage("answer in JSON", files=[ImagePart(
            url="https://images.example/valid.png"
        )]), {"response_model": _PortableResponseModel}),
    }
    message, kwargs = cases[case_name]
    valid_image_and_text = UserMessage("valid image context", files=[ImagePart(
        url="https://images.example/context.png"
    )])
    messages = [valid_image_and_text, message]
    with pytest.raises(expected_error) as caught:
        _invoke_tool_method(adapter, method_kind, messages, kwargs)
    assert type(caught.value) is expected_error
    assert expected_message in str(caught.value)
    assert adapter._sync_transport.requests == []
    assert async_calls == []
    assert transfer_attempts == []


def _outcome_payload(model, case_name, *, streamed=False):
    """Build a controlled native completed-response or SSE outcome."""
    message = {"role": "assistant", "content": "A partial answer"}
    finish_reason = "stop"
    if case_name == "tool_only":
        message["content"] = None
        message["tool_calls"] = [{
            "id": "call-weather", "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
        }]
        finish_reason = "tool_calls"
    elif case_name == "length":
        finish_reason = "length"
    elif case_name == "refusal":
        message["content"] = None
        message["refusal"] = "synthetic refusal"
    elif case_name == "content_filter":
        message["content"] = None
        finish_reason = "content_filter"

    choice = {"index": 0, "finish_reason": finish_reason}
    if streamed:
        delta = {"content": message["content"]}
        if "tool_calls" in message:
            delta["tool_calls"] = [
                {"index": 0, **message["tool_calls"][0]}
            ]
        if "refusal" in message:
            # Exercise fragmented provider refusal text in the native assembler.
            delta["refusal"] = "synthetic "
            choice["finish_reason"] = None
        choice["delta"] = delta
        if case_name == "refusal":
            return [
                {"id": f"response-{model}", "model": model, "created": 1730000000,
                 "choices": [choice]},
                {"id": f"response-{model}", "model": model,
                 "choices": [{"index": 0, "delta": {"refusal": "refusal"},
                              "finish_reason": "stop"}]},
            ]
    else:
        choice["message"] = message

    return [{
        "id": f"response-{model}", "model": model, "created": 1730000000,
        "choices": [choice],
    }]


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["chat", "achat", "stream_chat", "astream_chat"])
@pytest.mark.parametrize("case_name", ["visible", "tool_only", "length", "refusal", "content_filter"])
def test_zai_native_outcomes_keep_identity_tools_and_terminal_state(
    zai_adapter_runtime, monkeypatch, model, method_kind, case_name
):
    """All registered models retain reported completed and stream outcomes."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    completed_callbacks = []
    tool_callbacks = []
    body = _outcome_payload(model, case_name)[0]
    events = [SSEEvent(None, data=item) for item in _outcome_payload(
        model, case_name, streamed=True
    )]
    adapter._sync_transport = RecordingTransport(body=body, events=events)
    async_calls = []

    async def async_request(url, **kwargs):
        async_calls.append(("chat", url, kwargs))
        return body

    async def async_stream(url, **kwargs):
        async_calls.append(("stream", url, kwargs))
        for event in events:
            yield event

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    kwargs = {
        "on_done": completed_callbacks.append,
        "on_tool_call": tool_callbacks.append,
    }
    visible = None
    if method_kind == "chat":
        response = adapter.chat([UserMessage("outcome")])
    elif method_kind == "achat":
        response = asyncio.run(adapter.achat([UserMessage("outcome")]))
    elif method_kind == "stream_chat":
        visible = list(adapter.stream_chat([UserMessage("outcome")], **kwargs))
        response = completed_callbacks[0]
    else:
        async def consume():
            return [item async for item in adapter.astream_chat(
                [UserMessage("outcome")], **kwargs
            )]
        visible = asyncio.run(consume())
        response = completed_callbacks[0]

    assert response.response_id == f"response-{model}"
    assert response.model == model
    assert response.timestamp == 1730000000
    assert len(completed_callbacks) == (1 if method_kind in {"stream_chat", "astream_chat"} else 0)
    if case_name == "visible":
        assert response.content == "A partial answer"
        assert response.finish_reason == "stop"
        assert response.refusal is None and response.incomplete_reason is None
        assert response.tool_calls is None
        expected_visible = ["A partial answer"]
    elif case_name == "tool_only":
        assert response.content is None
        assert response.finish_reason == "tool_calls"
        assert [(call.name, call.arguments, call.call_id) for call in response.tool_calls] == [
            ("weather", {"city": "Paris"}, "call-weather")
        ]
        assert response.refusal is None and response.incomplete_reason is None
        expected_visible = []
    elif case_name == "length":
        assert response.content == "A partial answer"
        assert response.finish_reason == "length"
        assert response.incomplete_reason == "length"
        assert response.refusal is None and response.tool_calls is None
        expected_visible = ["A partial answer"]
    elif case_name == "refusal":
        assert response.content is None
        assert response.finish_reason == "stop"
        assert response.refusal == "synthetic refusal"
        assert response.incomplete_reason is None and response.tool_calls is None
        expected_visible = []
    else:
        assert response.content is None
        assert response.finish_reason == "content_filter"
        assert response.refusal == "content_filter"
        assert response.incomplete_reason is None and response.tool_calls is None
        expected_visible = []
    if visible is not None:
        assert visible == expected_visible
    if case_name == "tool_only":
        assert [(call.name, call.arguments, call.call_id) for call in tool_callbacks] == [
            ("weather", {"city": "Paris"}, "call-weather")
        ] if method_kind in {"stream_chat", "astream_chat"} else tool_callbacks == []
    else:
        assert tool_callbacks == []
    assert response.usage is None
    assert response.parsed_json is None and response.parsed_model is None
    assert response.reasoning_events == []
    if method_kind in {"chat", "stream_chat"}:
        assert len(adapter._sync_transport.requests) == 1
        assert async_calls == []
    else:
        assert len(async_calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["stream_chat", "astream_chat"])
def test_zai_terminal_usage_is_reported_without_estimation(
    zai_adapter_runtime, monkeypatch, model, method_kind
):
    """Usage-only terminal events contribute reported counts; absent usage stays absent."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    events = [
        SSEEvent(None, data={"id": "usage-tail", "model": model, "created": 1730000001,
                             "choices": [{"index": 0, "delta": {"content": "answer"},
                                          "finish_reason": "stop"}]}),
        SSEEvent(None, data={"id": "usage-tail", "model": model,
                             "choices": [], "usage": {"prompt_tokens": 7,
                             "completion_tokens": 3, "total_tokens": 10}}),
    ]
    adapter._sync_transport = RecordingTransport(events=events)
    async_calls = []

    async def async_stream(url, **kwargs):
        async_calls.append((url, kwargs))
        for event in events:
            yield event

    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    completed = []
    kwargs = {"on_done": completed.append}
    if method_kind == "stream_chat":
        visible = list(adapter.stream_chat([UserMessage("usage")], **kwargs))
    else:
        async def consume():
            return [item async for item in adapter.astream_chat(
                [UserMessage("usage")], **kwargs
            )]
        visible = asyncio.run(consume())
    assert visible == ["answer"]
    assert len(completed) == 1
    response = completed[0]
    assert response.response_id == "usage-tail" and response.model == model
    assert response.content == "answer" and response.finish_reason == "stop"
    assert (response.usage.input_tokens, response.usage.output_tokens,
            response.usage.total_tokens) == (7, 3, 10)
    assert response.parsed_json is None and response.parsed_model is None


@pytest.mark.unit
@pytest.mark.parametrize("raw, expected", [
    (None, None), ({}, None),
    ({"prompt_tokens": 4}, (4, None, None)),
    ({"prompt_tokens": 4, "completion_tokens": 2}, (4, 2, 6)),
    ({"prompt_tokens": -1}, None),
    ({"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 9}, None),
])
def test_zai_native_usage_parser_keeps_unknown_data_unset(raw, expected):
    parsed = ZaiAdapter._parse_usage(raw)
    if expected is None:
        assert parsed is None
    else:
        assert (parsed.input_tokens, parsed.output_tokens, parsed.total_tokens) == expected


@pytest.mark.unit
@pytest.mark.parametrize("method_kind", ["chat", "achat"])
@pytest.mark.parametrize("case_name", [
    "top_level", "missing_choices", "empty_choices", "nonlist_choices",
    "nonmapping_choice", "message", "content", "tool_arguments",
])
def test_zai_completed_malformed_outputs_keep_public_error_categories(
    zai_adapter_runtime, monkeypatch, caplog, method_kind, case_name
):
    """Malformed native envelopes and JSON arguments retain existing public errors."""
    import logging

    caplog.set_level(logging.DEBUG)
    api_key = "SYNTHETIC_COMPLETED_TOOL_CREDENTIAL_89"
    tool_marker = "SYNTHETIC_COMPLETED_MALFORMED_TOOL_ARGUMENTS_94"
    adapter = ZaiAdapter(api_key=api_key, model=ZAI_MODELS[0])
    valid = _response_payload(ZAI_MODELS[0])
    if case_name == "top_level":
        body = []
        expected_error = LLMAPIClientError
    elif case_name in {"missing_choices", "empty_choices", "nonlist_choices", "nonmapping_choice"}:
        body = copy.deepcopy(valid)
        if case_name == "missing_choices":
            body.pop("choices")
        elif case_name == "empty_choices":
            body["choices"] = []
        elif case_name == "nonlist_choices":
            body["choices"] = {"index": 0}
        else:
            body["choices"] = ["not-a-choice"]
        expected_error = LLMAPIClientError
    else:
        body = copy.deepcopy(valid)
        if case_name == "message":
            body["choices"][0]["message"] = None
            expected_error = LLMAPIClientError
        elif case_name == "content":
            body["choices"][0]["message"]["content"] = ["not", "text"]
            expected_error = LLMAPIClientError
        else:
            body["choices"][0]["message"]["tool_calls"] = [{
                "id": "bad-json", "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": f"{{bad json {api_key} {tool_marker}",
                },
            }]
            expected_error = InvalidToolArgumentsError
    adapter._sync_transport = RecordingTransport(body=body)
    async_calls = []

    async def async_request(url, **kwargs):
        async_calls.append((url, kwargs))
        return body

    monkeypatch.setattr(async_client_module, "async_request", async_request)
    with pytest.raises(expected_error) as caught:
        if method_kind == "chat":
            adapter.chat([UserMessage("malformed")])
        else:
            asyncio.run(adapter.achat([UserMessage("malformed")]))
    assert type(caught.value) is expected_error
    _assert_sdk_records_omit_markers(caplog, (api_key, tool_marker))
    if method_kind == "chat":
        assert len(adapter._sync_transport.requests) == 1
        assert async_calls == []
    else:
        assert adapter._sync_transport.requests == []
        assert len(async_calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["stream_chat", "astream_chat"])
def test_zai_stream_bad_tool_arguments_fail_before_done_callback(
    zai_adapter_runtime, monkeypatch, caplog, model, method_kind
):
    import logging

    caplog.set_level(logging.DEBUG)
    api_key = "SYNTHETIC_STREAM_TOOL_CREDENTIAL_42"
    tool_marker = "SYNTHETIC_STREAM_TOOL_ARGUMENTS_57"
    adapter = ZaiAdapter(api_key=api_key, model=model)
    payload = {"id": "bad-stream-tool", "model": model, "choices": [{
        "index": 0, "delta": {"tool_calls": [{"index": 0, "id": "bad-json",
        "type": "function", "function": {
            "name": "lookup", "arguments": f"{{bad {api_key} {tool_marker}",
        }}]},
        "finish_reason": "tool_calls",
    }]}
    adapter._sync_transport = RecordingTransport(events=[SSEEvent(None, data=payload)])
    async_calls = []

    async def async_stream(url, **kwargs):
        async_calls.append((url, kwargs))
        yield SSEEvent(None, data=payload)

    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    completed = []
    tool_callbacks = []
    with pytest.raises(InvalidToolArgumentsError) as caught:
        if method_kind == "stream_chat":
            list(adapter.stream_chat(
                [UserMessage("bad tool")], on_done=completed.append,
                on_tool_call=tool_callbacks.append,
            ))
        else:
            async def consume():
                return [item async for item in adapter.astream_chat(
                    [UserMessage("bad tool")], on_done=completed.append,
                    on_tool_call=tool_callbacks.append,
                )]
            asyncio.run(consume())
    assert type(caught.value) is InvalidToolArgumentsError
    _assert_sdk_records_omit_markers(
        caplog, (api_key, tool_marker),
    )
    assert completed == []
    assert tool_callbacks == []
    if method_kind == "stream_chat":
        assert len(adapter._sync_transport.requests) == 1
        assert async_calls == []
    else:
        assert adapter._sync_transport.requests == []
        assert len(async_calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize("method_kind", ["stream_chat", "astream_chat"])
def test_zai_stream_malformed_payload_and_missing_finish_keep_native_permissiveness(
    zai_adapter_runtime, monkeypatch, method_kind
):
    """Malformed events and EOF without finish retain the native empty outcome."""
    model = ZAI_MODELS[0]
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    events = [
        SSEEvent(None, data="not-a-mapping"),
        SSEEvent(None, data={"choices": [{"index": 1, "delta": {"content": "ignored"}}]}),
        SSEEvent(None, data={"choices": [{"index": 0, "delta": {"content": 7}}]}),
    ]
    adapter._sync_transport = RecordingTransport(events=events)

    async def async_stream(url, **kwargs):
        for event in events:
            yield event

    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    completed = []
    with pytest.warns(UserWarning, match="OpenAI returned empty content"):
        if method_kind == "stream_chat":
            visible = list(adapter.stream_chat(
                [UserMessage("empty" )], on_done=completed.append
            ))
        else:
            async def consume():
                return [item async for item in adapter.astream_chat(
                    [UserMessage("empty")], on_done=completed.append
                )]
            visible = asyncio.run(consume())
    assert visible == [] and len(completed) == 1
    response = completed[0]
    assert response.content is None and response.tool_calls is None
    assert response.response_id is None and response.finish_reason is None
    assert response.usage is None and response.refusal is None
    assert response.parsed_json is None and response.parsed_model is None


@pytest.mark.unit
@pytest.mark.parametrize("method_kind", ["stream_chat", "astream_chat"])
def test_zai_stream_partial_visible_eof_keeps_native_outcome_without_finish(
    zai_adapter_runtime, monkeypatch, method_kind
):
    """EOF after valid partial content retains text and identity without inventing state."""
    model = ZAI_MODELS[0]
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    events = [
        SSEEvent(None, data="malformed keepalive-like payload"),
        SSEEvent(None, data={"choices": [{"index": 1, "delta": {"content": "ignored"}}]}),
        SSEEvent(None, data={
            "id": "partial-eof", "model": model, "created": 1730000002,
            "choices": [{"index": 0, "delta": {"content": "partial answer"}}],
        }),
    ]
    adapter._sync_transport = RecordingTransport(events=events)
    async_calls = []

    async def async_stream(url, **kwargs):
        async_calls.append((url, kwargs))
        for event in events:
            yield event

    monkeypatch.setattr(async_client_module, "async_stream_request", async_stream)
    completed = []
    if method_kind == "stream_chat":
        visible = list(adapter.stream_chat(
            [UserMessage("partial EOF")], on_done=completed.append
        ))
        assert len(adapter._sync_transport.requests) == 1
        assert async_calls == []
    else:
        async def consume():
            return [item async for item in adapter.astream_chat(
                [UserMessage("partial EOF")], on_done=completed.append
            )]
        visible = asyncio.run(consume())
        assert adapter._sync_transport.requests == []
        assert len(async_calls) == 1
    assert visible == ["partial answer"]
    assert len(completed) == 1
    response = completed[0]
    assert response.response_id == "partial-eof" and response.model == model
    assert response.timestamp == 1730000002 and response.content == "partial answer"
    assert response.finish_reason is None and response.incomplete_reason is None
    assert response.refusal is None and response.tool_calls is None
    assert response.usage is None
    assert response.parsed_json is None and response.parsed_model is None


class _TrackedSyncSource:
    """Retain the raw SSE iterator so lifecycle ownership is observed directly."""

    def __init__(self, events, *, error=None, close_error=None):
        self.events = iter(events)
        self.error = error
        self.close_error = close_error
        self.close_count = 0
        self.next_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.next_count += 1
        try:
            return next(self.events)
        except StopIteration:
            if self.error is not None:
                raise self.error
            raise

    def close(self):
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


class _TrackedAsyncSource:
    """Retain an async SSE source and expose explicit closure/cancellation."""

    def __init__(self, events=(), *, error=None, close_error=None, gate=None):
        self.events = iter(events)
        self.error = error
        self.close_error = close_error
        self.gate = gate
        self.close_count = 0
        self.next_count = 0
        self.cancelled = None
        self.gate_started = asyncio.Event()
        self.closed_event = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.next_count += 1
        if self.gate is not None and self.next_count > 1:
            self.gate_started.set()
            try:
                await self.gate.wait()
            except asyncio.CancelledError as error:
                self.cancelled = error
                raise
        try:
            return next(self.events)
        except StopIteration:
            if self.error is not None:
                raise self.error
            raise StopAsyncIteration

    async def aclose(self):
        self.close_count += 1
        self.closed_event.set()
        if self.close_error is not None:
            raise self.close_error


class _TrackedSourceTransport(RecordingTransport):
    """Provide one retained raw SSE iterator through the real native client."""

    def __init__(self, source):
        super().__init__(events=[])
        self.source = source

    def post_sse(self, request, *, http_error_handler=None, stream_error_handler=None):
        self.requests.append(request)
        return self.source


def _bind_tracked_async_source(monkeypatch, source):
    """Patch only async transport I/O and return a list of observed sends."""
    calls = []

    def stream_factory(url, **kwargs):
        calls.append((url, kwargs))
        return source

    monkeypatch.setattr(async_client_module, "async_stream_request", stream_factory)
    return calls


def _life_event(*, text="", reasoning="", tool_fragment=None, finish=None):
    """Build one controlled Z.ai event with independent visible/reasoning/tool data."""
    delta = {}
    if text:
        delta["content"] = text
    if reasoning:
        delta["reasoning_content"] = reasoning
    if tool_fragment is not None:
        delta["tool_calls"] = [tool_fragment]
    return SSEEvent(None, data={"id": "life-id", "choices": [{
        "index": 0, "delta": delta, "finish_reason": finish,
    }]})


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["stream_chat", "astream_chat"])
@pytest.mark.parametrize("buffer_chars", [None, 1, 3])
def test_zai_native_stream_success_orders_bounded_text_reasoning_and_tools(
    zai_adapter_runtime, monkeypatch, model, method_kind, buffer_chars
):
    """Native streams flush bounded visible text before tools and final success."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    adapter._sync_transport = RecordingTransport()
    tool_head = {"index": 0, "id": "call-1", "type": "function", "function": {
        "name": "lookup", "arguments": '{"key":',
    }}
    tool_tail = {"index": 0, "function": {"arguments": '"v"}'}}
    events = [
        _life_event(text="he", reasoning="think"),
        _life_event(text="llo ", tool_fragment=tool_head),
        _life_event(text="world", tool_fragment=tool_tail, finish="tool_calls"),
    ]
    callbacks = []
    chunks = []
    deltas = []
    reasoning = []
    tools = []
    done = []
    async_calls = _bind_tracked_async_source(
        monkeypatch, _TrackedAsyncSource([])
    )

    if method_kind == "stream_chat":
        source = _TrackedSyncSource(events)
        adapter._sync_transport = _TrackedSourceTransport(source)
        visible = list(adapter.stream_chat(
            [UserMessage("hello")], buffer_chars=buffer_chars, capture_reasoning=True,
            on_chunk=lambda item: (callbacks.append("chunk"), chunks.append(item.text)),
            on_delta=lambda item: (callbacks.append("delta"), deltas.append(item)),
            on_reasoning=lambda item: (callbacks.append("reasoning"), reasoning.append(item.text)),
            on_tool_call=lambda item: (callbacks.append("tool"), tools.append(item)),
            on_done=lambda item: (callbacks.append("done"), done.append(item)),
        ))
        assert len(adapter._sync_transport.requests) == 1
        assert async_calls == []
    else:
        source = _TrackedAsyncSource(events)
        async_calls = _bind_tracked_async_source(monkeypatch, source)

        async def consume():
            async def chunk_callback(item):
                callbacks.append("chunk")
                chunks.append(item.text)

            def delta_callback(item):
                async def record():
                    await asyncio.sleep(0)
                    callbacks.append("delta")
                    deltas.append(item)
                return record()

            async def reasoning_callback(item):
                callbacks.append("reasoning")
                reasoning.append(item.text)

            def tool_callback(item):
                callbacks.append("tool")
                tools.append(item)

            async def done_callback(item):
                callbacks.append("done")
                done.append(item)

            stream = adapter.astream_chat(
                [UserMessage("hello")], buffer_chars=buffer_chars, capture_reasoning=True,
                on_chunk=chunk_callback, on_delta=delta_callback,
                on_reasoning=reasoning_callback, on_tool_call=tool_callback,
                on_done=done_callback,
            )
            visible = [item async for item in stream]
            assert source.close_count == 1
            assert len(async_calls) == 1
            assert adapter._sync_transport.requests == []
            return visible

        visible = asyncio.run(consume())

    expected_visible = (
        ["he", "llo ", "world"] if buffer_chars is None
        else list("hello world") if buffer_chars == 1
        else ["hel", "lo ", "wor", "ld"]
    )
    assert visible == expected_visible
    assert chunks == visible and deltas == visible
    assert all(0 < len(chunk) <= (buffer_chars or 6) for chunk in visible)
    assert reasoning == ["think"]
    assert len(tools) == 1 and tools[0].name == "lookup"
    assert tools[0].arguments == {"key": "v"}
    last_text_callback = max(
        index for index, callback in enumerate(callbacks)
        if callback in {"chunk", "delta"}
    )
    assert last_text_callback < callbacks.index("tool") < callbacks.index("done")
    assert callbacks.count("tool") == callbacks.count("done") == 1
    assert callbacks[-1] == "done" and len(done) == 1
    assert done[0].content == "hello world"
    assert done[0].reasoning_events[0].text == "think"
    assert source.close_count == 1


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["stream_chat", "astream_chat"])
def test_zai_native_stream_failure_preserves_exception_and_skips_success_flush(
    zai_adapter_runtime, monkeypatch, model, method_kind
):
    """Provider failure closes the raw source without flushing a pending tail."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    adapter._sync_transport = RecordingTransport()
    failure = RuntimeError("provider source failed")
    events = [_life_event(text="abc"), _life_event(tool_fragment={
        "index": 0, "id": "late", "type": "function", "function": {
            "name": "lookup", "arguments": "{}",
        },
    })]
    completed = []
    tools = []
    chunk_trace = []
    delta_trace = []
    yielded = []
    async_calls = _bind_tracked_async_source(
        monkeypatch, _TrackedAsyncSource([])
    )
    if method_kind == "stream_chat":
        source = _TrackedSyncSource(events[:1], error=failure)
        adapter._sync_transport = _TrackedSourceTransport(source)
        stream = adapter.stream_chat(
            [UserMessage("failure")], buffer_chars=2,
            on_done=completed.append, on_tool_call=tools.append,
            on_chunk=lambda chunk: chunk_trace.append(chunk.text),
            on_delta=delta_trace.append,
        )
        yielded.append(next(stream))
        assert yielded == ["ab"]
        with pytest.raises(RuntimeError) as caught:
            yielded.append(next(stream))
        assert caught.value is failure
        assert source.close_count == 1
        assert len(adapter._sync_transport.requests) == 1
        assert async_calls == []
    else:
        source = _TrackedAsyncSource(events[:1], error=failure)
        async_calls = _bind_tracked_async_source(monkeypatch, source)

        async def consume():
            stream = adapter.astream_chat(
                [UserMessage("failure")], buffer_chars=2,
                on_done=completed.append, on_tool_call=tools.append,
                on_chunk=lambda chunk: chunk_trace.append(chunk.text),
                on_delta=delta_trace.append,
            )
            yielded.append(await anext(stream))
            assert yielded == ["ab"]
            with pytest.raises(RuntimeError) as caught:
                yielded.append(await anext(stream))
            assert caught.value is failure
            assert source.close_count == 1
            assert len(async_calls) == 1
            assert adapter._sync_transport.requests == []

        asyncio.run(consume())
    assert chunk_trace == delta_trace == ["ab"]
    assert yielded == ["ab"]
    assert completed == [] and tools == []


@pytest.mark.unit
@pytest.mark.parametrize("method_kind", ["stream_chat", "astream_chat"])
@pytest.mark.parametrize("buffer_chars", [0, -1, True, 1.5, "2"])
def test_zai_native_stream_rejects_invalid_buffer_before_io(
    zai_adapter_runtime, monkeypatch, method_kind, buffer_chars
):
    """Invalid stream bounds retain the exact local ValueError before transport."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=ZAI_MODELS[0])
    sync_source = _TrackedSyncSource([])
    adapter._sync_transport = _TrackedSourceTransport(sync_source)
    async_calls = []

    def stream_factory(*args, **kwargs):
        async_calls.append((args, kwargs))
        return _TrackedAsyncSource([])

    monkeypatch.setattr(async_client_module, "async_stream_request", stream_factory)
    with pytest.raises(ValueError, match="buffer_chars must be None or a positive integer") as caught:
        if method_kind == "stream_chat":
            list(adapter.stream_chat([UserMessage("bad bound")], buffer_chars=buffer_chars))
        else:
            async def consume():
                return [item async for item in adapter.astream_chat(
                    [UserMessage("bad bound")], buffer_chars=buffer_chars
                )]
            asyncio.run(consume())
    assert type(caught.value) is ValueError
    assert adapter._sync_transport.requests == []
    assert async_calls == []


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("method_kind", ["stream_chat", "astream_chat"])
def test_zai_native_stream_caller_close_does_not_take_success_route(
    zai_adapter_runtime, monkeypatch, model, method_kind
):
    """Explicit caller closure owns the raw source and suppresses pending success."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    adapter._sync_transport = RecordingTransport()
    completed = []
    chunk_trace = []
    delta_trace = []
    yielded = []
    event = _life_event(text="abc")
    if method_kind == "stream_chat":
        source = _TrackedSyncSource([event, _life_event(text="never")])
        adapter._sync_transport = _TrackedSourceTransport(source)
        async_calls = _bind_tracked_async_source(
            monkeypatch, _TrackedAsyncSource([])
        )
        stream = adapter.stream_chat(
            [UserMessage("close")], buffer_chars=2, on_done=completed.append,
            on_chunk=lambda chunk: chunk_trace.append(chunk.text),
            on_delta=delta_trace.append,
        )
        yielded.append(next(stream))
        assert yielded == ["ab"]
        stream.close()
        assert source.close_count == 1
        assert len(adapter._sync_transport.requests) == 1
        assert async_calls == []
    else:
        source = _TrackedAsyncSource([event, _life_event(text="never")])
        async_calls = _bind_tracked_async_source(monkeypatch, source)

        async def consume():
            stream = adapter.astream_chat(
                [UserMessage("close")], buffer_chars=2, on_done=completed.append,
                on_chunk=lambda chunk: chunk_trace.append(chunk.text),
                on_delta=delta_trace.append,
            )
            yielded.append(await anext(stream))
            assert yielded == ["ab"]
            await stream.aclose()
            assert source.close_count == 1
            assert len(async_calls) == 1
            assert adapter._sync_transport.requests == []

        asyncio.run(consume())
    assert chunk_trace == delta_trace == yielded == ["ab"]
    assert completed == []


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
def test_zai_native_async_stream_cancellation_closes_raw_source_without_success(
    zai_adapter_runtime, monkeypatch, model
):
    """Real consumer cancellation closes the owned async source before loop exit."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    adapter._sync_transport = RecordingTransport()
    gate = asyncio.Event()
    source = _TrackedAsyncSource([_life_event(text="abc")], gate=gate)
    monkeypatch.setattr(async_client_module, "async_stream_request", lambda *a, **k: source)
    completed = []
    chunk_trace = []
    delta_trace = []
    yielded = []
    public_cancellations = []

    async def consume():
        stream = adapter.astream_chat(
            [UserMessage("cancel")], buffer_chars=2, on_done=completed.append,
            on_chunk=lambda chunk: chunk_trace.append(chunk.text),
            on_delta=delta_trace.append,
        )
        yielded.append(await asyncio.wait_for(anext(stream), timeout=2))
        assert yielded == ["ab"]

        async def consume_next():
            """Capture cancellation where the public stream receives it."""
            try:
                return await anext(stream)
            except asyncio.CancelledError as error:
                public_cancellations.append(error)
                raise

        waiter = asyncio.create_task(consume_next())
        await asyncio.wait_for(source.gate_started.wait(), timeout=2)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await waiter
        assert len(public_cancellations) == 1
        assert public_cancellations[0].args == ()
        assert source.cancelled is public_cancellations[0]
        assert isinstance(caught.value, asyncio.CancelledError)
        assert caught.value.args == public_cancellations[0].args
        assert source.close_count == 1
        assert len(async_calls) == 1
        assert adapter._sync_transport.requests == []
        assert chunk_trace == delta_trace == yielded == ["ab"]
        assert completed == []

    async_calls = _bind_tracked_async_source(monkeypatch, source)
    asyncio.run(asyncio.wait_for(consume(), timeout=5))
    assert len(public_cancellations) == 1


@pytest.mark.unit
@pytest.mark.parametrize("method_kind", ["stream_chat", "astream_chat"])
def test_zai_native_stream_close_failure_prevents_success_completion(
    zai_adapter_runtime, monkeypatch, method_kind
):
    """A raw-source close fault on normal EOF cannot flush a pending success tail."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=ZAI_MODELS[0])
    adapter._sync_transport = RecordingTransport()
    close_failure = RuntimeError("raw source close failed")
    completed = []
    chunk_trace = []
    delta_trace = []
    yielded = []
    if method_kind == "stream_chat":
        source = _TrackedSyncSource([_life_event(text="tail")], close_error=close_failure)
        adapter._sync_transport = _TrackedSourceTransport(source)
        async_calls = _bind_tracked_async_source(
            monkeypatch, _TrackedAsyncSource([])
        )
        stream = adapter.stream_chat(
            [UserMessage("close fault")], buffer_chars=5, on_done=completed.append,
            on_chunk=lambda chunk: chunk_trace.append(chunk.text),
            on_delta=delta_trace.append,
        )
        with pytest.raises(RuntimeError) as caught:
            yielded.extend(stream)
        assert caught.value is close_failure
        assert source.close_count == 1
        assert len(adapter._sync_transport.requests) == 1
        assert async_calls == []
    else:
        source = _TrackedAsyncSource([_life_event(text="tail")], close_error=close_failure)
        async_calls = _bind_tracked_async_source(monkeypatch, source)

        async def consume():
            stream = adapter.astream_chat(
                [UserMessage("close fault")], buffer_chars=5, on_done=completed.append,
                on_chunk=lambda chunk: chunk_trace.append(chunk.text),
                on_delta=delta_trace.append,
            )
            with pytest.raises(RuntimeError) as caught:
                async for item in stream:
                    yielded.append(item)
            assert caught.value is close_failure
            assert source.close_count == 1
            assert len(async_calls) == 1
            assert adapter._sync_transport.requests == []
            assert yielded == chunk_trace == delta_trace == []

        asyncio.run(consume())
    assert yielded == chunk_trace == delta_trace == []
    assert completed == []


@pytest.mark.unit
@pytest.mark.parametrize("method_kind", ["stream_chat", "astream_chat"])
def test_zai_native_callback_error_wins_over_raw_source_close_failure(
    zai_adapter_runtime, monkeypatch, method_kind
):
    """Closing after a caller callback failure preserves its identity over close faults."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=ZAI_MODELS[0])
    adapter._sync_transport = RecordingTransport()
    callback_failure = RuntimeError("caller callback failed")
    close_failure = RuntimeError("secondary raw source close failed")
    event = _life_event(text="abc")
    completed = []
    chunk_trace = []
    delta_attempts = []
    yielded = []
    if method_kind == "stream_chat":
        source = _TrackedSyncSource([event], close_error=close_failure)
        adapter._sync_transport = _TrackedSourceTransport(source)
        async_calls = _bind_tracked_async_source(
            monkeypatch, _TrackedAsyncSource([])
        )

        def fail_delta(_text):
            delta_attempts.append(_text)
            raise callback_failure

        stream = adapter.stream_chat(
            [UserMessage("callback close")], buffer_chars=2,
            on_delta=fail_delta, on_done=completed.append,
            on_chunk=lambda chunk: chunk_trace.append(chunk.text),
        )
        with pytest.raises(RuntimeError) as caught:
            yielded.extend(stream)
        assert caught.value is callback_failure
        assert source.close_count == 1
        assert len(adapter._sync_transport.requests) == 1
        assert async_calls == []
    else:
        source = _TrackedAsyncSource([event], close_error=close_failure)
        async_calls = _bind_tracked_async_source(monkeypatch, source)

        async def consume():
            async def fail_delta(_text):
                delta_attempts.append(_text)
                raise callback_failure

            stream = adapter.astream_chat(
                [UserMessage("callback close")], buffer_chars=2,
                on_delta=fail_delta, on_done=completed.append,
                on_chunk=lambda chunk: chunk_trace.append(chunk.text),
            )
            with pytest.raises(RuntimeError) as caught:
                async for item in stream:
                    yielded.append(item)
            assert caught.value is callback_failure
            assert source.close_count == 1
            assert len(async_calls) == 1
            assert adapter._sync_transport.requests == []
            assert yielded == [] and chunk_trace == ["ab"]
            assert delta_attempts == ["ab"]
            assert completed == []

        asyncio.run(consume())
    assert yielded == [] and chunk_trace == ["ab"]
    assert delta_attempts == ["ab"]
    assert completed == []


@pytest.mark.unit
@pytest.mark.parametrize("method_kind", ["stream_chat", "astream_chat"])
@pytest.mark.parametrize("callback_name", ["on_chunk", "on_delta", "on_reasoning", "on_tool_call", "on_done"])
def test_zai_native_stream_callback_errors_keep_identity_and_close_source(
    zai_adapter_runtime, monkeypatch, caplog, method_kind, callback_name
):
    """Early and completion-stage callback failures stop at their exact callback point."""
    import logging

    caplog.set_level(logging.DEBUG)
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=ZAI_MODELS[0])
    adapter._sync_transport = RecordingTransport()
    callback_marker = "CALLER_CALLBACK_PRIVATE_SENTINEL_68"
    failure = RuntimeError(
        f"{callback_name} failed {callback_marker}", "CALLER_OWNED_ARGUMENT",
    )
    original_args = failure.args
    events = [_life_event(text="abc"), _life_event(reasoning="reason"), _life_event(
        tool_fragment={"index": 0, "id": "c", "type": "function", "function": {
            "name": "lookup", "arguments": "{}",
        }}, finish="tool_calls",
    )]
    order = []
    yielded = []
    tools = []
    done_calls = []
    async_calls = _bind_tracked_async_source(
        monkeypatch, _TrackedAsyncSource([])
    )

    if method_kind == "stream_chat":
        source = _TrackedSyncSource(events)
        adapter._sync_transport = _TrackedSourceTransport(source)

        def make_callback(name, value_getter):
            def invoke(value):
                value = value_getter(value)
                order.append((name, value))
                if callback_name == {
                    "chunk": "on_chunk", "delta": "on_delta",
                    "reasoning": "on_reasoning", "tool": "on_tool_call",
                    "done": "on_done",
                }[name]:
                    raise failure
                if name == "tool":
                    tools.append(value)
                if name == "done":
                    done_calls.append(value)
            return invoke

        kwargs = {
            "on_chunk": make_callback("chunk", lambda item: item.text),
            "on_delta": make_callback("delta", lambda item: item),
            "on_reasoning": make_callback("reasoning", lambda item: item.text),
            "on_tool_call": make_callback("tool", lambda item: item.name),
            "on_done": make_callback("done", lambda _item: "done"),
        }
        stream = adapter.stream_chat(
            [UserMessage("callback")], buffer_chars=2, capture_reasoning=True, **kwargs
        )
        with pytest.raises(RuntimeError) as caught:
            while True:
                yielded.append(next(stream))
        assert caught.value is failure
        assert caught.value.args == original_args
        assert failure.args == original_args
        assert source.close_count == 1
        assert len(adapter._sync_transport.requests) == 1
        assert async_calls == []
    else:
        source = _TrackedAsyncSource(events)
        async_calls = _bind_tracked_async_source(monkeypatch, source)

        async def consume():
            def make_callback(name, value_getter):
                async def invoke(value):
                    value = value_getter(value)
                    order.append((name, value))
                    if callback_name == {
                        "chunk": "on_chunk", "delta": "on_delta",
                        "reasoning": "on_reasoning", "tool": "on_tool_call",
                        "done": "on_done",
                    }[name]:
                        raise failure
                    if name == "tool":
                        tools.append(value)
                    if name == "done":
                        done_calls.append(value)
                return invoke

            kwargs = {
                "on_chunk": make_callback("chunk", lambda item: item.text),
                "on_delta": make_callback("delta", lambda item: item),
                "on_reasoning": make_callback("reasoning", lambda item: item.text),
                "on_tool_call": make_callback("tool", lambda item: item.name),
                "on_done": make_callback("done", lambda _item: "done"),
            }
            with pytest.raises(RuntimeError) as caught:
                stream = adapter.astream_chat(
                    [UserMessage("callback")], buffer_chars=2,
                    capture_reasoning=True, **kwargs,
                )
                async for item in stream:
                    yielded.append(item)
            assert caught.value is failure
            assert caught.value.args == original_args
            assert failure.args == original_args
            assert source.close_count == 1
            assert len(async_calls) == 1
            assert adapter._sync_transport.requests == []

        asyncio.run(consume())
    _assert_sdk_records_omit_markers(caplog, (callback_marker,))
    prefix = [("chunk", "ab"), ("delta", "ab")]
    if callback_name == "on_chunk":
        expected_order = [("chunk", "ab")]
        expected_yielded = []
    elif callback_name == "on_delta":
        expected_order = prefix
        expected_yielded = []
    elif callback_name == "on_reasoning":
        expected_order = prefix + [("reasoning", "reason")]
        expected_yielded = ["ab"]
    else:
        expected_order = prefix + [
            ("reasoning", "reason"), ("chunk", "c"), ("delta", "c"),
            ("tool", "lookup"),
        ]
        expected_yielded = ["ab", "c"]
        if callback_name == "on_done":
            expected_order.append(("done", "done"))
    assert order == expected_order
    assert yielded == expected_yielded
    assert done_calls == []
    assert tools == ([] if callback_name in {"on_chunk", "on_delta", "on_reasoning", "on_tool_call"} else ["lookup"])


def _record_post_attempt(attempts, args, kwargs):
    """Normalize a fake backend POST to the externally observable request."""
    attempts.append({
        "url": args[0],
        "headers": kwargs["headers"],
        "payload": kwargs["json"],
        "timeout": kwargs["timeout"],
        "stream": bool(kwargs.get("stream", False)),
    })


def _record_send_attempt(attempts, request, kwargs):
    """Normalize a fake HTTPX send request and its transport streaming flag."""
    build_args, build_kwargs = request
    attempts.append({
        "url": build_args[1],
        "headers": build_kwargs["headers"],
        "payload": build_kwargs["json"],
        "timeout": build_kwargs["timeout"],
        "stream": bool(kwargs["stream"]),
    })


def _assert_native_error_attempt(attempts, *, model, api_key, timeout, streaming):
    """Assert one native request kept its official destination and wire identity."""
    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt["url"] == ZAI_CHAT_COMPLETIONS_URL
    assert attempt["headers"] == {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    assert attempt["timeout"] == timeout
    assert attempt["stream"] is streaming
    assert attempt["payload"]["model"] == model
    assert attempt["payload"].get("stream", False) is streaming


def _assert_sdk_records_omit_markers(caplog, markers):
    """Inspect every captured SDK log channel, including deferred record data."""
    records = [record for record in caplog.records if record.name.startswith("llm_api_adapter")]
    rendered = " ".join(
        record.getMessage() + repr(record.args) + repr(record.exc_info)
        for record in records
    )
    for marker in markers:
        assert marker not in rendered
    return records


_NATIVE_HTTP_FAILURES = (
    (401, {}, LLMAPIAuthorizationError),
    (403, {}, LLMAPIAuthorizationError),
    (429, {"error": {"type": "rate_limit_error"}}, LLMAPIRateLimitError),
    (402, {}, LLMAPIUsageLimitError),
    (400, {"error": {"type": "context_length_exceeded"}}, LLMAPITokenLimitError),
    (408, {}, LLMAPITimeoutError),
    (504, {}, LLMAPITimeoutError),
    (422, {}, LLMAPIClientError),
    (500, {}, LLMAPIServerError),
)


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("lane", ["requests", "httpx", "async"])
@pytest.mark.parametrize("method_kind", ["chat", "stream"])
@pytest.mark.parametrize("status,payload,expected_error", _NATIVE_HTTP_FAILURES)
def test_zai_native_http_failures_keep_categories_and_safe_transport_logs(
    zai_adapter_runtime, monkeypatch, caplog, model, lane, method_kind,
    status, payload, expected_error,
):
    """Real Core transports map native HTTP failures and omit private log text."""
    import json
    import logging
    import requests
    import httpx
    from llm_api_adapter.llms import requests_transport

    api_key = "SYNTHETIC_PRIVATE_CREDENTIAL_91"
    raw_event = "SYNTHETIC_RAW_PROVIDER_EVENT_72"
    reasoning = "SYNTHETIC_REASONING_TEXT_64"
    tool_args = "SYNTHETIC_TOOL_ARGUMENTS_38"
    timeout = 17.25
    message = f"{api_key} {raw_event} {reasoning} {tool_args}"
    body = copy.deepcopy(payload or {"error": {"message": message}})
    if isinstance(body.get("error"), dict):
        body["error"].setdefault("message", message)
    attempts = []

    class RequestsResponse:
        status_code = status
        def json(self): return body
        def raise_for_status(self):
            raise requests.HTTPError(message, response=self)
        def iter_lines(self, **kwargs):
            del kwargs
            yield ("data: " + json.dumps({"error": body.get("error", {})})).encode()
        def close(self): pass

    if lane == "requests":
        adapter = ZaiAdapter(api_key=api_key, model=model)
        def post(*args, **kwargs):
            _record_post_attempt(attempts, args, kwargs)
            return RequestsResponse()
        monkeypatch.setattr(requests_transport.requests, "post", post)
    elif lane == "httpx":
        adapter = ZaiAdapter(api_key=api_key, model=model, transport="httpx")
        class HttpxResponse:
            status_code = status
            def __init__(self):
                self.request = httpx.Request("POST", ZAI_CHAT_COMPLETIONS_URL)
            def json(self): return body
            def raise_for_status(self):
                raise httpx.HTTPStatusError(message, request=self.request, response=self)
            def iter_lines(self):
                yield "data: " + json.dumps({"error": body.get("error", {})})
            def close(self): pass
        class HttpxClient:
            def post(self, *args, **kwargs):
                _record_post_attempt(attempts, args, kwargs)
                return HttpxResponse()
            def build_request(self, *args, **kwargs): return args, kwargs
            def send(self, request, **kwargs):
                _record_send_attempt(attempts, request, kwargs)
                return HttpxResponse()
            def close(self): pass
        monkeypatch.setattr(httpx, "Client", HttpxClient)
    else:
        adapter = ZaiAdapter(api_key=api_key, model=model)
        class AsyncResponse:
            status_code = status
            def __init__(self):
                self.request = httpx.Request("POST", ZAI_CHAT_COMPLETIONS_URL)
            def json(self): return body
            def raise_for_status(self):
                raise httpx.HTTPStatusError(message, request=self.request, response=self)
            async def aread(self): return None
            async def aclose(self): return None
            async def aiter_lines(self):
                yield "data: " + json.dumps({"error": body.get("error", {})})
        class AsyncClient:
            async def post(self, *args, **kwargs):
                _record_post_attempt(attempts, args, kwargs)
                return AsyncResponse()
            def build_request(self, *args, **kwargs): return args, kwargs
            async def send(self, request, **kwargs):
                _record_send_attempt(attempts, request, kwargs)
                return AsyncResponse()
            async def aclose(self): return None
        monkeypatch.setattr(httpx, "AsyncClient", AsyncClient)

    caplog.set_level(logging.DEBUG)
    with pytest.raises(expected_error) as caught:
        if lane == "async":
            async def invoke():
                if method_kind == "chat":
                    return await adapter.achat([UserMessage("safe failure")], timeout_s=timeout)
                return [item async for item in adapter.astream_chat(
                    [UserMessage("safe failure")], timeout_s=timeout,
                )]
            asyncio.run(invoke())
        elif method_kind == "chat":
            adapter.chat([UserMessage("safe failure")], timeout_s=timeout)
        else:
            list(adapter.stream_chat([UserMessage("safe failure")], timeout_s=timeout))
    assert type(caught.value) is expected_error
    _assert_native_error_attempt(
        attempts, model=model, api_key=api_key, timeout=timeout,
        streaming=method_kind == "stream",
    )
    transport_records = [r for r in caplog.records if r.name.endswith(("requests_transport", "httpx_transport", "async_streaming"))]
    assert transport_records
    _assert_sdk_records_omit_markers(
        caplog, (api_key, raw_event, reasoning, tool_args),
    )
    diagnostics = " ".join(
        record.getMessage() + repr(record.args) + repr(record.exc_info)
        for record in transport_records
    )
    assert "Z.ai transport failure (details omitted)" in diagnostics


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("lane", ["requests", "httpx", "async"])
@pytest.mark.parametrize(
    "error_type,expected_error",
    [
        ("authentication_error", LLMAPIAuthorizationError),
        ("rate_limit_error", LLMAPIRateLimitError),
        ("insufficient_quota", LLMAPIUsageLimitError),
        ("context_length_exceeded", LLMAPITokenLimitError),
        ("timeout_error", LLMAPITimeoutError),
        ("api_error", LLMAPIServerError),
        ("invalid_request", LLMAPIClientError),
    ],
)
def test_zai_native_sse_errors_keep_categories_and_do_not_log_raw_events(
    zai_adapter_runtime, monkeypatch, caplog, model, lane, error_type, expected_error,
):
    """Real SSE decoders route native categories without logging event contents."""
    import json
    import logging
    import httpx
    from llm_api_adapter.llms import requests_transport

    api_key = "SYNTHETIC_SSE_CREDENTIAL_19"
    markers = (api_key, "SYNTHETIC_SSE_EVENT_28",
               "SYNTHETIC_SSE_REASONING_37", "SYNTHETIC_SSE_TOOL_ARGS_46")
    timeout = 23.5
    frame = "data: " + json.dumps({"error": {
        "type": error_type, "message": " ".join(markers),
    }})
    attempts = []

    class Response:
        status_code = 200
        def raise_for_status(self): return None
        def iter_lines(self, **kwargs):
            del kwargs
            yield "event: error".encode()
            yield frame.encode()
            yield b""
        def close(self): pass

    if lane == "requests":
        adapter = ZaiAdapter(api_key=api_key, model=model)
        def post(*args, **kwargs):
            _record_post_attempt(attempts, args, kwargs)
            return Response()
        monkeypatch.setattr(requests_transport.requests, "post", post)
    elif lane == "httpx":
        adapter = ZaiAdapter(api_key=api_key, model=model, transport="httpx")
        class HttpxResponse(Response):
            def iter_lines(self):
                yield "event: error"
                yield frame
                yield ""
        class HttpxClient:
            def build_request(self, *args, **kwargs): return args, kwargs
            def send(self, request, **kwargs):
                _record_send_attempt(attempts, request, kwargs)
                return HttpxResponse()
            def close(self): pass
        monkeypatch.setattr(httpx, "Client", HttpxClient)
    else:
        adapter = ZaiAdapter(api_key=api_key, model=model)
        class AsyncResponse(Response):
            async def aiter_lines(self):
                yield "event: error"
                yield frame
                yield ""
            async def aclose(self): return None
        class AsyncClient:
            def build_request(self, *args, **kwargs): return args, kwargs
            async def send(self, request, **kwargs):
                _record_send_attempt(attempts, request, kwargs)
                return AsyncResponse()
            async def aclose(self): return None
        monkeypatch.setattr(httpx, "AsyncClient", AsyncClient)

    caplog.set_level(logging.DEBUG)
    with pytest.raises(expected_error) as caught:
        if lane == "async":
            async def consume():
                return [item async for item in adapter.astream_chat(
                    [UserMessage("SSE error")], timeout_s=timeout,
                )]
            asyncio.run(consume())
        else:
            list(adapter.stream_chat([UserMessage("SSE error")], timeout_s=timeout))
    assert type(caught.value) is expected_error
    _assert_native_error_attempt(
        attempts, model=model, api_key=api_key, timeout=timeout, streaming=True,
    )
    _assert_sdk_records_omit_markers(caplog, markers)


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("lane", ["requests", "httpx", "async"])
@pytest.mark.parametrize("method_kind", ["chat", "stream"])
@pytest.mark.parametrize("failure_kind", ["timeout", "connection"])
def test_zai_native_transport_faults_keep_public_errors_and_safe_logs(
    zai_adapter_runtime, monkeypatch, caplog, model, lane, method_kind, failure_kind,
):
    """Native HTTPX and Requests faults retain mapped classes without echoing text."""
    import logging
    import requests
    import httpx
    from llm_api_adapter.llms import requests_transport

    api_key = "SYNTHETIC_FAILURE_CREDENTIAL_53"
    marker = f"{api_key} SYNTHETIC_FAILURE_EVENT_REASONING_TOOL_ARGS_53"
    timeout = 31.75
    expected_error = LLMAPITimeoutError if failure_kind == "timeout" else LLMAPIClientError
    attempts = []

    if lane == "requests":
        adapter = ZaiAdapter(api_key=api_key, model=model)
        failure = requests.Timeout(marker) if failure_kind == "timeout" else requests.ConnectionError(marker)
        def post(*args, **kwargs):
            _record_post_attempt(attempts, args, kwargs)
            raise failure
        monkeypatch.setattr(requests_transport.requests, "post", post)
    elif lane == "httpx":
        adapter = ZaiAdapter(api_key=api_key, model=model, transport="httpx")
        request = httpx.Request("POST", ZAI_CHAT_COMPLETIONS_URL)
        failure = httpx.TimeoutException(marker, request=request) if failure_kind == "timeout" else httpx.ConnectError(marker, request=request)
        class HttpxClient:
            def post(self, *args, **kwargs):
                _record_post_attempt(attempts, args, kwargs)
                raise failure
            def build_request(self, *args, **kwargs): return args, kwargs
            def send(self, request, **kwargs):
                _record_send_attempt(attempts, request, kwargs)
                raise failure
            def close(self): pass
        monkeypatch.setattr(httpx, "Client", HttpxClient)
    else:
        adapter = ZaiAdapter(api_key=api_key, model=model)
        request = httpx.Request("POST", ZAI_CHAT_COMPLETIONS_URL)
        failure = httpx.TimeoutException(marker, request=request) if failure_kind == "timeout" else httpx.ConnectError(marker, request=request)
        class AsyncClient:
            async def post(self, *args, **kwargs):
                _record_post_attempt(attempts, args, kwargs)
                raise failure
            def build_request(self, *args, **kwargs): return args, kwargs
            async def send(self, request, **kwargs):
                _record_send_attempt(attempts, request, kwargs)
                raise failure
            async def aclose(self): return None
        monkeypatch.setattr(httpx, "AsyncClient", AsyncClient)

    caplog.set_level(logging.DEBUG)
    with pytest.raises(expected_error) as caught:
        if lane == "async":
            async def invoke():
                if method_kind == "chat":
                    return await adapter.achat([UserMessage("transport fault")], timeout_s=timeout)
                return [item async for item in adapter.astream_chat(
                    [UserMessage("transport fault")], timeout_s=timeout,
                )]
            asyncio.run(invoke())
        elif method_kind == "chat":
            adapter.chat([UserMessage("transport fault")], timeout_s=timeout)
        else:
            list(adapter.stream_chat([UserMessage("transport fault")], timeout_s=timeout))
    assert type(caught.value) is expected_error
    assert caught.value.detail == marker
    _assert_native_error_attempt(
        attempts, model=model, api_key=api_key, timeout=timeout,
        streaming=method_kind == "stream",
    )
    records = [r for r in caplog.records if r.name.endswith(("requests_transport", "httpx_transport", "async_streaming"))]
    assert records
    assert "Z.ai transport failure (details omitted)" in " ".join(r.getMessage() for r in records)
    _assert_sdk_records_omit_markers(caplog, (api_key, marker))


_NO_USAGE = object()


_ACCOUNTING_USAGE_CASES = [
    pytest.param(_NO_USAGE, None, id="usage-omitted"),
    pytest.param(None, None, id="usage-null"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "prompt_tokens_details": {"cached_tokens": 3}}, (10, 4, 14, 3, None), id="full-with-cache"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4}, (10, 4, 14, None, None), id="components-infer-only-exact-sum"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": None}, (10, 4, 14, None, None), id="null-total-sums-components"),
    pytest.param({"total_tokens": 14}, (None, None, 14, None, None), id="total-only-no-inference"),
    pytest.param({"prompt_tokens": 10}, (10, None, None, None, None), id="partial-input-only"),
    pytest.param({"completion_tokens": 4}, (None, 4, None, None, None), id="partial-output-only"),
    pytest.param({"prompt_tokens": 10, "total_tokens": 14}, (10, None, 14, None, None), id="input-and-total-output-omitted"),
    pytest.param({"completion_tokens": 4, "total_tokens": 14}, (None, 4, 14, None, None), id="output-and-total-input-omitted"),
    pytest.param({"prompt_tokens": 0, "completion_tokens": 4, "total_tokens": 4}, (0, 4, 4, None, None), id="zero-input"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}, (10, 0, 10, None, None), id="zero-output"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": None, "total_tokens": 10}, (10, None, 10, None, None), id="null-output-count"),
    pytest.param({"prompt_tokens": None, "completion_tokens": 4, "total_tokens": 4}, (None, 4, 4, None, None), id="null-input-count"),
    pytest.param({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "prompt_tokens_details": {"cached_tokens": 0}}, (0, 0, 0, 0, None), id="explicit-zero"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": None, "total_tokens": None}, (10, None, None, None, None), id="null-counts-remain-unknown"),
    pytest.param({"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}, None, id="all-null-counts"),
    pytest.param({}, None, id="empty-usage"),
    pytest.param("not-an-object", None, id="malformed-usage-object"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 15}, None, id="contradictory-total"),
    pytest.param({"prompt_tokens": False, "completion_tokens": 4, "total_tokens": 4}, None, id="false-input-is-not-zero"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": False, "total_tokens": 10}, None, id="false-output-is-not-zero"),
    pytest.param({"prompt_tokens": 10, "total_tokens": False}, None, id="false-total-is-not-zero"),
    *[
        pytest.param(
            (
                {"prompt_tokens": value, "completion_tokens": 4}
                if field == "prompt_tokens"
                else {"prompt_tokens": 10, "completion_tokens": value}
                if field == "completion_tokens"
                else {"prompt_tokens": 10, "total_tokens": value}
            ),
            None,
            id=f"invalid-{field}-{label}",
        )
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        for value, label in ((True, "boolean"), (-1, "negative"), (1.5, "noninteger"), ("10", "string"))
    ],
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "prompt_tokens_details": {"cached_tokens": 0}}, (10, 4, 14, 0, None), id="cache-zero"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "prompt_tokens_details": {"cached_tokens": 5}}, (10, 4, 14, 5, None), id="cache-partial"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "prompt_tokens_details": {"cached_tokens": 10}}, (10, 4, 14, 10, None), id="cache-full"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "prompt_tokens_details": {}}, (10, 4, 14, None, None), id="cache-omitted"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "prompt_tokens_details": {"cached_tokens": None}}, (10, 4, 14, None, None), id="cache-null"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "prompt_tokens_details": {"cached_tokens": True}}, (10, 4, 14, None, None), id="cache-boolean"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "prompt_tokens_details": {"cached_tokens": "5"}}, (10, 4, 14, None, None), id="cache-string"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "prompt_tokens_details": {"cached_tokens": -1}}, (10, 4, 14, None, None), id="cache-negative"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "prompt_tokens_details": {"cached_tokens": 11}}, (10, 4, 14, None, None), id="cache-above-input"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "prompt_tokens_details": []}, (10, 4, 14, None, None), id="cache-nonobject-details"),
    pytest.param({"completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 2}}, (None, 4, None, None, None), id="cache-without-known-input"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "prompt_tokens_details": {"cached_tokens": False}}, (10, 4, 14, None, None), id="cache-false-is-not-zero"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "prompt_tokens_details": {"cached_tokens": 2.5}}, (10, 4, 14, None, None), id="cache-noninteger"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "cache_write_tokens": 2}, (10, 4, 14, None, None), id="top-level-cache-write-ignored"),
    pytest.param({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14, "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 2}}, (10, 4, 14, 3, None), id="nested-cache-write-ignored"),
]


def _accounting_reply(model, usage, *, streaming=False):
    payload = {
        "id": "accounting-response-id",
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "accounted"},
            "finish_reason": "stop",
        }],
    }
    if usage is not _NO_USAGE:
        payload["usage"] = usage
    if not streaming:
        return payload
    first = {
        "id": "accounting-stream-id",
        "model": model,
        "choices": [{"index": 0, "delta": {"content": "accounted"}, "finish_reason": "stop"}],
    }
    terminal = {"id": "accounting-stream-id", "model": model, "choices": []}
    if usage is not _NO_USAGE:
        terminal["usage"] = usage
    return [first, terminal]


def _usage_tuple(usage):
    if usage is None:
        return None
    assert isinstance(usage, ZaiUsage)
    return (
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        usage.cached_tokens,
        usage.cache_write_tokens,
    )


class _AccountingHTTPResponse:
    status_code = 200

    def __init__(self, payload, *, streaming, events):
        self.payload = payload
        self.streaming = streaming
        self.events = events
        self.closed = False

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload

    def iter_lines(self, **kwargs):
        del kwargs
        import json
        for event in self.events:
            yield ("data: " + json.dumps(event)).encode()
            yield b""
        yield b"data: [DONE]"
        yield b""

    async def aiter_lines(self):
        for line in self.iter_lines():
            yield line.decode()

    def close(self):
        self.closed = True

    async def aclose(self):
        self.closed = True


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("lane", ["chat-requests", "chat-httpx", "achat", "stream-requests", "stream-httpx", "astream"])
@pytest.mark.parametrize(("raw_usage", "expected"), _ACCOUNTING_USAGE_CASES)
def test_zai_native_public_accounting_across_transports_and_execution_methods(
    zai_adapter_runtime, monkeypatch, model, lane, raw_usage, expected,
):
    """Exercise native accounting through the real JSON/SSE and selected transport paths."""
    import httpx
    from llm_api_adapter.llms import requests_transport

    is_stream = lane.startswith("stream") or lane == "astream"
    is_async = lane in {"achat", "astream"}
    transport_name = "httpx" if lane.endswith("httpx") else "requests"
    calls = []

    def response_for(payload, streaming):
        events = _accounting_reply(model, raw_usage, streaming=True) if streaming else []
        return _AccountingHTTPResponse(_accounting_reply(model, raw_usage), streaming=streaming, events=events)

    def sync_post(url, *, headers, json, timeout=None, stream=False):
        payload = json
        calls.append(("requests", url, headers, payload, timeout, stream))
        return response_for(payload, stream)

    if not is_async and transport_name == "requests":
        monkeypatch.setattr(requests_transport.requests, "post", sync_post)
    elif not is_async:
        class SyncHTTPXClient:
            def post(self, url, *, headers, json, timeout):
                calls.append(("httpx-json", url, headers, json, timeout, False))
                return response_for(json, False)
            def build_request(self, method, url, *, headers, json, timeout):
                calls.append(("httpx-build", method, url, headers, json, timeout, True))
                return (method, url, headers, json, timeout)
            def send(self, request, *, stream):
                method, url, headers, payload, timeout = request
                calls.append(("httpx-send", method, url, headers, payload, timeout, stream))
                return response_for(payload, stream)
            def close(self):
                return None
        monkeypatch.setattr(httpx, "Client", SyncHTTPXClient)
    else:
        class AsyncHTTPXClient:
            async def post(self, url, *, headers, json, timeout):
                calls.append(("async-json", url, headers, json, timeout, False))
                return response_for(json, False)
            def build_request(self, method, url, *, headers, json, timeout):
                calls.append(("async-build", method, url, headers, json, timeout, True))
                return (method, url, headers, json, timeout)
            async def send(self, request, *, stream):
                method, url, headers, payload, timeout = request
                calls.append(("async-send", method, url, headers, payload, timeout, stream))
                return response_for(payload, stream)
            async def aclose(self):
                return None
        monkeypatch.setattr(httpx, "AsyncClient", AsyncHTTPXClient)

    caller, _ = _native_adapter("direct", model, "httpx" if transport_name == "httpx" else "requests")
    request_kwargs = {"timeout_s": 5.25}
    completed = []
    if lane == "chat-requests" or lane == "chat-httpx":
        response = caller.chat([UserMessage("usage")], **request_kwargs)
    elif lane == "achat":
        response = asyncio.run(caller.achat([UserMessage("usage")], **request_kwargs))
    elif lane in {"stream-requests", "stream-httpx"}:
        visible = list(caller.stream_chat([UserMessage("usage")], on_done=completed.append, **request_kwargs))
        assert visible == ["accounted"]
        response = completed[0]
    else:
        async def collect():
            visible = [item async for item in caller.astream_chat(
                [UserMessage("usage")], on_done=completed.append, **request_kwargs,
            )]
            return visible
        assert asyncio.run(collect()) == ["accounted"]
        response = completed[0]

    assert response.content == "accounted"
    assert response.model == model
    assert response.finish_reason == "stop"
    assert _usage_tuple(response.usage) == expected
    assert response.usage is None if expected is None else isinstance(response.usage, ZaiUsage)
    if is_stream:
        assert response.response_id == "accounting-stream-id"
        assert len(completed) == 1
    else:
        assert response.response_id == "accounting-response-id"
    expected_kinds = {
        "chat-requests": ["requests"],
        "chat-httpx": ["httpx-json"],
        "achat": ["async-json"],
        "stream-requests": ["requests"],
        "stream-httpx": ["httpx-build", "httpx-send"],
        "astream": ["async-build", "async-send"],
    }[lane]
    assert [call[0] for call in calls] == expected_kinds
    for call in calls:
        kind = call[0]
        if kind in {"httpx-build", "httpx-send", "async-build", "async-send"}:
            _, method, url, headers, payload, timeout, streaming_flag = call
            assert method == "POST"
            assert streaming_flag is True
        else:
            _, url, headers, payload, timeout, streaming_flag = call
        assert url == ZAI_CHAT_COMPLETIONS_URL
        assert headers == {"Authorization": "Bearer synthetic-zai-key", "Content-Type": "application/json"}
        assert timeout == 5.25
        assert payload["model"] == model
        if is_stream:
            assert payload["stream"] is True
            assert payload["stream_options"] == {"include_usage": True}
            if kind == "requests":
                assert streaming_flag is True
        else:
            assert "stream" not in payload
            assert streaming_flag is False


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("lane", ["stream-requests", "stream-httpx", "astream"])
def test_zai_native_stream_usage_merges_cumulative_snapshots_without_doubling(
    zai_adapter_runtime, monkeypatch, model, lane,
):
    """Retain terminal-only and cumulative usage snapshots as reported, without summing them."""
    import httpx
    from llm_api_adapter.llms import requests_transport

    snapshots = [
        {"prompt_tokens": 10},
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        {"prompt_tokens_details": {"cached_tokens": 3}},
    ]
    events = [
        {"id": "merge-stream-id", "model": model, "choices": [{"index": 0, "delta": {"content": "merged"}, "finish_reason": "stop"}], "usage": snapshots[0]},
        {"id": "merge-stream-id", "model": model, "choices": [], "usage": snapshots[1]},
        {"id": "merge-stream-id", "model": model, "choices": [], "usage": snapshots[2]},
        {"id": "merge-stream-id", "model": model, "choices": [], "usage": snapshots[3]},
    ]
    calls = []

    class Response(_AccountingHTTPResponse):
        def __init__(self):
            super().__init__({}, streaming=True, events=events)

    if lane == "stream-requests":
        def post(url, *, headers, json, timeout, stream=False):
            calls.append(("requests", url, headers, json, timeout, stream))
            return Response()
        monkeypatch.setattr(requests_transport.requests, "post", post)
    elif lane == "stream-httpx":
        class Client:
            def build_request(self, method, url, *, headers, json, timeout):
                calls.append(("httpx-build", method, url, headers, json, timeout, True))
                return (method, url, headers, json, timeout)
            def send(self, request, *, stream):
                method, url, headers, payload, timeout = request
                calls.append(("httpx-send", method, url, headers, payload, timeout, stream))
                return Response()
            def close(self):
                return None
        monkeypatch.setattr(httpx, "Client", Client)
    else:
        class AsyncClient:
            def build_request(self, method, url, *, headers, json, timeout):
                calls.append(("async-build", method, url, headers, json, timeout, True))
                return (method, url, headers, json, timeout)
            async def send(self, request, *, stream):
                method, url, headers, payload, timeout = request
                calls.append(("async-send", method, url, headers, payload, timeout, stream))
                return Response()
            async def aclose(self):
                return None
        monkeypatch.setattr(httpx, "AsyncClient", AsyncClient)

    caller, _ = _native_adapter("direct", model, "httpx" if lane == "stream-httpx" else "requests")
    completed = []
    timeout = 9.75
    if lane == "astream":
        async def consume():
            return [item async for item in caller.astream_chat(
                [UserMessage("merge usage")], on_done=completed.append, timeout_s=timeout,
            )]
        assert asyncio.run(consume()) == ["merged"]
    else:
        assert list(caller.stream_chat(
            [UserMessage("merge usage")], on_done=completed.append, timeout_s=timeout,
        )) == ["merged"]
    assert len(completed) == 1
    assert completed[0].content == "merged"
    assert completed[0].response_id == "merge-stream-id"
    assert completed[0].model == model
    assert completed[0].finish_reason == "stop"
    assert _usage_tuple(completed[0].usage) == (10, 4, 14, 3, None)
    expected_kinds = {
        "stream-requests": ["requests"],
        "stream-httpx": ["httpx-build", "httpx-send"],
        "astream": ["async-build", "async-send"],
    }[lane]
    assert [call[0] for call in calls] == expected_kinds
    for call in calls:
        if call[0] == "requests":
            _, url, headers, payload, actual_timeout, stream = call
            assert stream is True
        else:
            _, method, url, headers, payload, actual_timeout, stream = call
            assert method == "POST"
            assert stream is True
        assert url == ZAI_CHAT_COMPLETIONS_URL
        assert headers == {"Authorization": "Bearer synthetic-zai-key", "Content-Type": "application/json"}
        assert payload["model"] == model
        assert payload["stream"] is True
        assert payload["stream_options"] == {"include_usage": True}
        assert actual_timeout == timeout


_PRICING_USAGE_CASES = [
    pytest.param(
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
         "cache_write_tokens": 5,
         "prompt_tokens_details": {"cached_tokens": 3, "cache_write_tokens": 7}},
        (10, 4, 14, 3, None), id="complete-cache-subset",
    ),
    pytest.param(
        {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9,
         "prompt_tokens_details": {"cached_tokens": 0}},
        (7, 2, 9, 0, None), id="mid-input-cache-zero",
    ),
    pytest.param(
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
         "prompt_tokens_details": {"cached_tokens": 10}},
        (10, 4, 14, 10, None), id="full-cache-hit",
    ),
    pytest.param(
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
         "prompt_tokens_details": {"cached_tokens": 0}},
        (0, 0, 0, 0, None), id="reported-zeroes",
    ),
    pytest.param(
        {"prompt_tokens": 0, "completion_tokens": 4, "total_tokens": 4},
        (0, 4, 4, None, None), id="zero-input-unknown-cache",
    ),
    pytest.param(_NO_USAGE, None, id="usage-missing"),
    pytest.param(
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 15},
        None, id="usage-contradictory-total",
    ),
    pytest.param({"prompt_tokens": 10}, (10, None, None, None, None), id="input-only"),
    pytest.param(
        {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 3}},
        (10, None, None, 3, None), id="partial-input-with-confirmed-cache",
    ),
    pytest.param({"completion_tokens": 4}, (None, 4, None, None, None), id="output-only"),
    pytest.param({"total_tokens": 14}, (None, None, 14, None, None), id="total-only"),
    pytest.param(
        {"prompt_tokens": 10, "completion_tokens": True},
        None, id="invalid-output-with-valid-input",
    ),
    pytest.param(
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
         "prompt_tokens_details": {}},
        (10, 4, 14, None, None), id="missing-cache-evidence",
    ),
    pytest.param(
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
         "prompt_tokens_details": {"cached_tokens": True}},
        (10, 4, 14, None, None), id="invalid-cache-evidence",
    ),
]


def _zai_raw_model_metadata(model):
    """Read exact Z.ai model tariff and limits from the registered metadata."""
    return PLUGIN.model_metadata.organization_data["models"][model]


def _zai_expected_costs(raw_usage, normalized, rates, currency):
    """Calculate expected costs independently from the reported counts and raw rates."""
    if normalized is None or normalized[0] is None:
        return None, None, None, None
    input_tokens, output_tokens, _, cache_tokens, _ = normalized
    input_rate, output_rate, cache_rate = rates
    if cache_tokens is None and cache_rate is not None:
        input_cost = None
    elif cache_tokens and cache_rate is None:
        input_cost = None
    else:
        cached = cache_tokens or 0
        input_cost = (input_tokens - cached) * input_rate + cached * (cache_rate or 0)
    output_cost = None if output_tokens is None else output_tokens * output_rate
    total_cost = None if input_cost is None or output_cost is None else input_cost + output_cost
    return currency, input_cost, output_cost, total_cost


def _install_pricing_accounting_io(monkeypatch, *, model, lane, raw_usage):
    """Stub only backend I/O while leaving Core JSON/SSE and finalization active."""
    import httpx
    from llm_api_adapter.llms import requests_transport

    calls = []

    def make_response(streaming):
        events = _accounting_reply(model, raw_usage, streaming=True) if streaming else []
        return _AccountingHTTPResponse(
            _accounting_reply(model, raw_usage), streaming=streaming, events=events,
        )

    def requests_post(url, *, headers, json, timeout=None, stream=False):
        calls.append(("requests", "POST", url, headers, json, timeout, stream))
        return make_response(stream)

    is_async = lane in {"achat", "astream"}
    is_httpx = lane.endswith("httpx")
    is_stream = lane.startswith("stream") or lane == "astream"
    if not is_async and not is_httpx:
        monkeypatch.setattr(requests_transport.requests, "post", requests_post)
    elif not is_async:
        class SyncClient:
            def post(self, url, *, headers, json, timeout):
                calls.append(("httpx-json", "POST", url, headers, json, timeout, False))
                return make_response(False)

            def build_request(self, method, url, *, headers, json, timeout):
                calls.append(("httpx-build", method, url, headers, json, timeout, True))
                return (method, url, headers, json, timeout)

            def send(self, request, *, stream):
                method, url, headers, payload, timeout = request
                calls.append(("httpx-send", method, url, headers, payload, timeout, stream))
                return make_response(stream)

            def close(self):
                return None

        monkeypatch.setattr(httpx, "Client", SyncClient)
    else:
        class AsyncClient:
            async def post(self, url, *, headers, json, timeout):
                calls.append(("async-json", "POST", url, headers, json, timeout, False))
                return make_response(False)

            def build_request(self, method, url, *, headers, json, timeout):
                calls.append(("async-build", method, url, headers, json, timeout, True))
                return (method, url, headers, json, timeout)

            async def send(self, request, *, stream):
                method, url, headers, payload, timeout = request
                calls.append(("async-send", method, url, headers, payload, timeout, stream))
                return make_response(stream)

            async def aclose(self):
                return None

        monkeypatch.setattr(httpx, "AsyncClient", AsyncClient)
    return calls, is_async, is_stream, is_httpx


def _run_pricing_lane(caller, lane, completed):
    """Invoke one public native execution method and return its completed response."""
    kwargs = {"timeout_s": 6.75}
    messages = [UserMessage("price reported usage")]
    if lane in {"chat-requests", "chat-httpx"}:
        return caller.chat(messages, **kwargs)
    if lane == "achat":
        return asyncio.run(caller.achat(messages, **kwargs))
    if lane in {"stream-requests", "stream-httpx"}:
        assert list(caller.stream_chat(messages, on_done=completed.append, **kwargs)) == ["accounted"]
        return completed[0]

    async def collect():
        assert [item async for item in caller.astream_chat(
            messages, on_done=completed.append, **kwargs,
        )] == ["accounted"]
    asyncio.run(collect())
    return completed[0]


def _assert_pricing_lane_trace(calls, lane, model, *, invocations=1, timeout=6.75):
    """Verify each public invocation made exactly its expected official request sequence."""
    expected_once = {
        "chat-requests": ["requests"], "chat-httpx": ["httpx-json"],
        "achat": ["async-json"], "stream-requests": ["requests"],
        "stream-httpx": ["httpx-build", "httpx-send"],
        "astream": ["async-build", "async-send"],
    }[lane]
    expected = expected_once * invocations
    assert [call[0] for call in calls] == expected
    streaming = lane.startswith("stream") or lane == "astream"
    for _, method, url, headers, payload, actual_timeout, stream in calls:
        assert method == "POST"
        assert url == ZAI_CHAT_COMPLETIONS_URL
        assert headers == {"Authorization": "Bearer synthetic-zai-key", "Content-Type": "application/json"}
        assert actual_timeout == timeout
        assert payload["model"] == model
        assert stream is streaming
        if streaming:
            assert payload["stream"] is True
            assert payload["stream_options"] == {"include_usage": True}
        else:
            assert "stream" not in payload


def _assert_response_costs(response, currency, input_cost, output_cost, total_cost):
    """Check each known price approximately and each unknown price explicitly."""
    assert response.currency == currency
    for actual, expected in (
        (response.cost_input, input_cost),
        (response.cost_output, output_cost),
        (response.cost_total, total_cost),
    ):
        if expected is None:
            assert actual is None
        else:
            assert actual == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize(
    "lane", ["chat-requests", "chat-httpx", "achat", "stream-requests", "stream-httpx", "astream"],
)
@pytest.mark.parametrize("pricing_mode", ["registry", "caller-setters"])
@pytest.mark.parametrize(("raw_usage", "normalized"), _PRICING_USAGE_CASES)
def test_zai_native_registry_and_caller_tariffs_price_reported_usage(
    zai_adapter_runtime, monkeypatch, model, lane, pricing_mode, raw_usage, normalized,
):
    """Price native usage through all public methods without estimating unknown components."""
    raw_model = _zai_raw_model_metadata(model)
    raw_org = PLUGIN.model_metadata.organization_data
    native_rates = (
        raw_model["pricing_tiers"][0]["input_per_1m"] / 1_000_000,
        raw_model["pricing_tiers"][0]["output_per_1m"] / 1_000_000,
        raw_model["pricing_tiers"][0]["cache_read_input_per_1m"] / 1_000_000,
    )
    caller_rates = (native_rates[0] * 3, native_rates[1] * 2, native_rates[2])
    currency = raw_org["currency"]
    caller_currency = "TST"
    calls, is_async, is_stream, is_httpx = _install_pricing_accounting_io(
        monkeypatch, model=model, lane=lane, raw_usage=raw_usage,
    )
    caller, _ = _native_adapter("direct", model, "httpx" if is_httpx else "requests")
    if pricing_mode == "caller-setters":
        caller.pricing.set_in_per_1m(raw_model["pricing_tiers"][0]["input_per_1m"] * 3)
        caller.pricing.set_out_per_1m(raw_model["pricing_tiers"][0]["output_per_1m"] * 2)
        caller.pricing.set_currency(caller_currency)
        rates, expected_currency = caller_rates, caller_currency
    else:
        rates, expected_currency = native_rates, currency

    completed = []
    response = _run_pricing_lane(caller, lane, completed)
    assert response.content == "accounted"
    assert response.model == model
    assert response.finish_reason == "stop"
    assert _usage_tuple(response.usage) == normalized
    expected_costs = _zai_expected_costs(raw_usage, normalized, rates, expected_currency)
    _assert_response_costs(response, *expected_costs)
    assert response.cost_breakdown is None
    if is_stream:
        assert response.response_id == "accounting-stream-id"
        assert completed == [response]
    else:
        assert response.response_id == "accounting-response-id"
        assert completed == []
    expected_kinds = {
        "chat-requests": ["requests"], "chat-httpx": ["httpx-json"],
        "achat": ["async-json"], "stream-requests": ["requests"],
        "stream-httpx": ["httpx-build", "httpx-send"],
        "astream": ["async-build", "async-send"],
    }[lane]
    assert [call[0] for call in calls] == expected_kinds
    for _, method, url, headers, payload, timeout, stream_flag in calls:
        assert method == "POST"
        assert url == ZAI_CHAT_COMPLETIONS_URL
        assert headers == {"Authorization": "Bearer synthetic-zai-key", "Content-Type": "application/json"}
        assert timeout == 6.75
        assert payload["model"] == model
        assert stream_flag is is_stream
        if is_stream:
            assert payload["stream"] is True
            assert payload["stream_options"] == {"include_usage": True}
        else:
            assert "stream" not in payload


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize(
    "lane", ["chat-requests", "chat-httpx", "achat", "stream-requests", "stream-httpx", "astream"],
)
def test_zai_native_caller_tier_boundary_uses_full_reported_input(
    zai_adapter_runtime, monkeypatch, model, lane,
):
    """Select synthetic caller tiers at inclusive boundaries, including cache-heavy prompts."""
    import httpx
    from llm_api_adapter.llms import requests_transport

    raw_model = _zai_raw_model_metadata(model)
    raw_tier = raw_model["pricing_tiers"][0]
    boundary = raw_model["limits"]["context_window_tokens"] // 2
    low_rates = (
        raw_tier["input_per_1m"] / 1_000_000,
        raw_tier["output_per_1m"] / 1_000_000,
        raw_tier["cache_read_input_per_1m"] / 1_000_000,
    )
    high_rates = tuple(rate * 2 for rate in low_rates)
    pricing = Pricing((
        PricingTier(boundary, low_rates[0], low_rates[1], low_rates[2]),
        PricingTier(None, high_rates[0], high_rates[1], high_rates[2]),
    ), currency="SYN")
    cases = ((boundary - 1, 0), (boundary, 0), (boundary + 1, 0), (boundary + 1, 2))
    calls = []
    index = 0
    response_usage = []

    def make_response(streaming):
        usage = response_usage[index]
        events = _accounting_reply(model, usage, streaming=True) if streaming else []
        return _AccountingHTTPResponse(_accounting_reply(model, usage), streaming=streaming, events=events)

    def requests_post(url, *, headers, json, timeout=None, stream=False):
        calls.append(("requests", "POST", url, headers, json, timeout, stream))
        return make_response(stream)

    is_async = lane in {"achat", "astream"}
    is_httpx = lane.endswith("httpx")
    if not is_async and not is_httpx:
        monkeypatch.setattr(requests_transport.requests, "post", requests_post)
    elif not is_async:
        class SyncClient:
            def post(self, url, *, headers, json, timeout):
                calls.append(("httpx-json", "POST", url, headers, json, timeout, False))
                return make_response(False)
            def build_request(self, method, url, *, headers, json, timeout):
                calls.append(("httpx-build", method, url, headers, json, timeout, True))
                return (method, url, headers, json, timeout)
            def send(self, request, *, stream):
                method, url, headers, payload, timeout = request
                calls.append(("httpx-send", method, url, headers, payload, timeout, stream))
                return make_response(stream)
            def close(self):
                return None
        monkeypatch.setattr(httpx, "Client", SyncClient)
    else:
        class AsyncClient:
            async def post(self, url, *, headers, json, timeout):
                calls.append(("async-json", "POST", url, headers, json, timeout, False))
                return make_response(False)
            def build_request(self, method, url, *, headers, json, timeout):
                calls.append(("async-build", method, url, headers, json, timeout, True))
                return (method, url, headers, json, timeout)
            async def send(self, request, *, stream):
                method, url, headers, payload, timeout = request
                calls.append(("async-send", method, url, headers, payload, timeout, stream))
                return make_response(stream)
            async def aclose(self):
                return None
        monkeypatch.setattr(httpx, "AsyncClient", AsyncClient)

    caller, _ = _native_adapter("direct", model, "httpx" if is_httpx else "requests")
    caller.pricing = pricing
    for input_tokens, cached_tokens in cases:
        usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": 4,
            "total_tokens": input_tokens + 4,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        }
        response_usage.append(usage)
        completed = []
        response = _run_pricing_lane(caller, lane, completed)
        selected_rates = low_rates if input_tokens <= boundary else high_rates
        expected_input = ((input_tokens - cached_tokens) * selected_rates[0]
                          + cached_tokens * selected_rates[2])
        expected_output = 4 * selected_rates[1]
        assert response.content == "accounted"
        assert response.model == model
        assert response.finish_reason == "stop"
        assert _usage_tuple(response.usage) == (input_tokens, 4, input_tokens + 4, cached_tokens, None)
        _assert_response_costs(
            response, "SYN", expected_input, expected_output,
            expected_input + expected_output,
        )
        assert response.cost_breakdown is None
        if lane.startswith("stream") or lane == "astream":
            assert response.response_id == "accounting-stream-id"
            assert completed == [response]
        else:
            assert response.response_id == "accounting-response-id"
        index += 1
    _assert_pricing_lane_trace(calls, lane, model, invocations=4)


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
def test_zai_native_pricing_mutation_and_assignment_references_are_live(
    zai_adapter_runtime, monkeypatch, model,
):
    """Prove registry copies, setters, and caller-assigned Pricing references via real sends."""
    from llm_api_adapter.llms import requests_transport

    raw_model = _zai_raw_model_metadata(model)
    raw_tier = raw_model["pricing_tiers"][0]
    registry_model = zai_adapter_runtime.organizations["zai"].models[model]
    registry_snapshot = copy.deepcopy(registry_model.pricing_tiers)
    adapter, _ = _native_adapter("direct", model)
    control, _ = _native_adapter("direct", model)
    facade, _ = _native_adapter("facade", model)
    adapter_pricing_reference = adapter.pricing
    assert adapter.pricing is not control.pricing
    assert adapter.pricing is not registry_model.pricing_tiers
    assert adapter.pricing.tiers is not control.pricing.tiers
    calls = []
    usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
             "prompt_tokens_details": {"cached_tokens": 3}}

    def post(url, *, headers, json, timeout=None, stream=False):
        calls.append((url, headers, json, timeout, stream))
        return _AccountingHTTPResponse(_accounting_reply(model, usage), streaming=False, events=[])

    monkeypatch.setattr(requests_transport.requests, "post", post)
    native_rates = (raw_tier["input_per_1m"] / 1_000_000,
                    raw_tier["output_per_1m"] / 1_000_000,
                    raw_tier["cache_read_input_per_1m"] / 1_000_000)

    def check(response, rates, currency):
        expected = _zai_expected_costs(usage, (10, 4, 14, 3, None), rates, currency)
        assert response.content == "accounted"
        assert response.model == model
        assert response.response_id == "accounting-response-id"
        assert response.finish_reason == "stop"
        assert _usage_tuple(response.usage) == (10, 4, 14, 3, None)
        assert response.currency == currency
        assert response.cost_input == pytest.approx(expected[1])
        assert response.cost_output == pytest.approx(expected[2])
        assert response.cost_total == pytest.approx(expected[3])
        assert response.cost_breakdown is None

    adapter_pricing_reference.set_in_per_1m(raw_tier["input_per_1m"] * 2)
    adapter_pricing_reference.set_out_per_1m(raw_tier["output_per_1m"] * 3)
    adapter_pricing_reference.set_currency("SET")
    assert adapter.pricing is adapter_pricing_reference
    first = adapter.chat([UserMessage("pricing before reference assignment")], timeout_s=3.5)
    check(first, (native_rates[0] * 2, native_rates[1] * 3, native_rates[2]), "SET")

    control_response = control.chat([UserMessage("independent registry copy")], timeout_s=3.5)
    check(control_response, native_rates, PLUGIN.model_metadata.organization_data["currency"])

    shared = Pricing((PricingTier(None, *native_rates),), currency="RAW")
    adapter.pricing = shared
    facade.adapter.pricing = shared
    assert adapter.pricing is shared and facade.adapter.pricing is shared
    assert facade.pricing is shared
    assigned_before_mutation = adapter.chat(
        [UserMessage("assigned direct per-token pricing before setter mutation")],
        timeout_s=3.5,
    )
    check(assigned_before_mutation, native_rates, "RAW")
    shared.set_in_per_1m(raw_tier["input_per_1m"] * 4)
    shared.set_out_per_1m(raw_tier["output_per_1m"] * 5)
    shared.set_currency("REF")
    assigned_response = adapter.chat([UserMessage("pricing after reference mutation")], timeout_s=3.5)
    facade_response = facade.chat([UserMessage("facade delegated pricing")], timeout_s=3.5)
    assigned_rates = (native_rates[0] * 4, native_rates[1] * 5, native_rates[2])
    check(assigned_response, assigned_rates, "REF")
    check(facade_response, assigned_rates, "REF")
    control_after = control.chat([UserMessage("registry copy remains independent")], timeout_s=3.5)
    check(control_after, native_rates, PLUGIN.model_metadata.organization_data["currency"])
    fresh, _ = _native_adapter("direct", model)
    fresh_response = fresh.chat([UserMessage("fresh adapter keeps registry defaults")], timeout_s=3.5)
    check(fresh_response, native_rates, PLUGIN.model_metadata.organization_data["currency"])

    assert len(calls) == 7
    for url, headers, payload, timeout, stream in calls:
        assert url == ZAI_CHAT_COMPLETIONS_URL
        assert headers == {"Authorization": "Bearer synthetic-zai-key", "Content-Type": "application/json"}
        assert payload["model"] == model and "stream" not in payload
        assert timeout == 3.5 and stream is False
    current_registry_pricing = zai_adapter_runtime.organizations["zai"].models[model].pricing_tiers
    assert current_registry_pricing == registry_snapshot
    assert current_registry_pricing is not adapter.pricing
    assert current_registry_pricing is not control.pricing
    assert current_registry_pricing is not fresh.pricing


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize(
    "lane", ["chat-requests", "chat-httpx", "achat", "stream-requests", "stream-httpx", "astream"],
)
def test_zai_native_positive_cache_without_caller_read_rate_stays_unpriced(
    zai_adapter_runtime, monkeypatch, model, lane,
):
    """A caller tariff without a cache rate cannot price confirmed cached input."""
    raw_tier = _zai_raw_model_metadata(model)["pricing_tiers"][0]
    usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
             "prompt_tokens_details": {"cached_tokens": 3}}
    calls, _, is_stream, is_httpx = _install_pricing_accounting_io(
        monkeypatch, model=model, lane=lane, raw_usage=usage,
    )
    caller, _ = _native_adapter("direct", model, "httpx" if is_httpx else "requests")
    caller.pricing = Pricing((PricingTier(
        None, raw_tier["input_per_1m"] / 1_000_000,
        raw_tier["output_per_1m"] / 1_000_000,
    ),), currency="CALLER")
    completed = []
    response = _run_pricing_lane(caller, lane, completed)
    assert response.content == "accounted"
    assert response.model == model
    assert response.finish_reason == "stop"
    assert _usage_tuple(response.usage) == (10, 4, 14, 3, None)
    _assert_response_costs(
        response, "CALLER", None,
        4 * raw_tier["output_per_1m"] / 1_000_000, None,
    )
    assert response.cost_breakdown is None
    if is_stream:
        assert response.response_id == "accounting-stream-id"
        assert completed == [response]
    else:
        assert response.response_id == "accounting-response-id"
        assert completed == []
    _assert_pricing_lane_trace(calls, lane, model)


_ZERO_CALLER_PRICING_CASES = [
    pytest.param(
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
         "prompt_tokens_details": {"cached_tokens": 3}},
        (10, 4, 14, 3, None), (0.0, 0.0, 0.0), id="complete-positive-cache-zero-rates",
    ),
    pytest.param(_NO_USAGE, None, (None, None, None), id="omitted-usage-remains-unknown"),
    pytest.param(
        {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
         "prompt_tokens_details": {}},
        (10, 4, 14, None, None), (None, 0.0, None), id="missing-cache-evidence-remains-unknown",
    ),
]


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize(
    "lane", ["chat-requests", "chat-httpx", "achat", "stream-requests", "stream-httpx", "astream"],
)
@pytest.mark.parametrize(("raw_usage", "normalized", "expected_costs"), _ZERO_CALLER_PRICING_CASES)
def test_zai_native_zero_rate_caller_tariff_preserves_unknown_evidence(
    zai_adapter_runtime, monkeypatch, model, lane, raw_usage, normalized, expected_costs,
):
    """Explicit zero rates price reported components as zero without filling missing evidence."""
    raw_tier = _zai_raw_model_metadata(model)["pricing_tiers"][0]
    pricing = Pricing((PricingTier(
        None,
        raw_tier["input_per_1m"] / 1_000_000,
        raw_tier["output_per_1m"] / 1_000_000,
        0.0,
    ),), currency="ZERO")
    pricing.set_in_per_1m(0)
    pricing.set_out_per_1m(0)
    calls, _, is_stream, is_httpx = _install_pricing_accounting_io(
        monkeypatch, model=model, lane=lane, raw_usage=raw_usage,
    )
    caller, _ = _native_adapter("direct", model, "httpx" if is_httpx else "requests")
    caller.pricing = pricing
    completed = []
    response = _run_pricing_lane(caller, lane, completed)
    assert response.content == "accounted"
    assert response.model == model
    assert response.finish_reason == "stop"
    assert _usage_tuple(response.usage) == normalized
    expected_currency = "ZERO" if normalized is not None and normalized[0] is not None else None
    _assert_response_costs(response, expected_currency, *expected_costs)
    assert response.cost_breakdown is None
    if is_stream:
        assert response.response_id == "accounting-stream-id"
        assert completed == [response]
    else:
        assert response.response_id == "accounting-response-id"
        assert completed == []
    _assert_pricing_lane_trace(calls, lane, model)


_T021_GENERIC_ROOTS = {
    "chat_completions": "https://generic-chat.example.invalid/v3",
    "responses": "https://generic-responses.example.invalid/proxy",
}
_T021_GENERIC_KEYS = {
    "chat_completions": "generic-chat-isolation-key",
    "responses": "generic-responses-isolation-key",
}
_T021_LANES = (
    "chat-requests", "chat-httpx", "achat", "stream-requests", "stream-httpx", "astream",
)


def _t021_chat_reply(model, label, input_tokens, output_tokens, cached_tokens=0):
    """Make a distinct Chat Completions result with explicit usage components."""
    return {
        "id": f"{label}-chat-id", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": f"{label} text"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": input_tokens, "completion_tokens": output_tokens,
                  "total_tokens": input_tokens + output_tokens,
                  "prompt_tokens_details": {"cached_tokens": cached_tokens}},
    }


def _t021_responses_reply(model, label, input_tokens, output_tokens):
    """Make a completed Responses result with distinct identity and usage."""
    return {
        "id": f"{label}-responses-id", "object": "response", "model": model,
        "status": "completed", "output": [{
            "id": f"{label}-message-id", "type": "message", "role": "assistant",
            "status": "completed", "content": [{"type": "output_text", "text": f"{label} text"}],
        }],
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens,
                  "total_tokens": input_tokens + output_tokens},
    }


class _T021HTTPResponse:
    """Minimal Requests/HTTPX reply supporting their real JSON and SSE decoders."""

    status_code = 200

    def __init__(self, body):
        self.body = body
        self.closed = False

    def raise_for_status(self):
        return None

    def json(self):
        return self.body

    def iter_lines(self):
        import json
        for event in self._events():
            yield ("data: " + json.dumps(event)).encode()
            yield b""
        yield b"data: [DONE]"
        yield b""

    async def aiter_lines(self):
        for line in self.iter_lines():
            yield line.decode()

    def _events(self):
        body = self.body
        if body.get("object") == "response":
            content = body["output"][0]["content"][0]["text"]
            yield {"type": "response.output_text.delta", "delta": content,
                   "item_id": body["output"][0]["id"], "output_index": 0,
                   "content_index": 0, "sequence_number": 0}
            yield {"type": "response.completed", "response": copy.deepcopy(body),
                   "sequence_number": 1}
            return
        event = {
            "id": body["id"], "model": body["model"],
            "choices": [{"index": 0, "delta": {"content": body["choices"][0]["message"]["content"]},
                         "finish_reason": None}],
        }
        yield event
        yield {"id": body["id"], "model": body["model"], "choices": [{"index": 0,
                "delta": {}, "finish_reason": "stop"}]}
        yield {"id": body["id"], "model": body["model"], "choices": [], "usage": body["usage"]}

    async def aclose(self):
        self.closed = True

    def close(self):
        self.closed = True


def _t021_routes(model, generic_api, *, include_native=True):
    """Return immutable route expectations and separate provider reply envelopes."""
    native_usage = (12, 4, 3)
    generic_usage = (23, 5, 0)
    native_url = ZAI_CHAT_COMPLETIONS_URL
    generic_url = f"{_T021_GENERIC_ROOTS[generic_api]}/{'responses' if generic_api == 'responses' else 'chat/completions'}"
    routes = {}
    if include_native:
        routes[(native_url, "Bearer synthetic-zai-key")] = {
            "label": "native-zai", "timeout": 2.25,
            "body": _t021_chat_reply(model, "native-zai", *native_usage),
        }
    routes[(generic_url, f"Bearer {_T021_GENERIC_KEYS[generic_api]}")] = {
        "label": f"generic-{generic_api}", "timeout": 3.75,
        "body": (_t021_responses_reply(model, f"generic-{generic_api}", 23, 5)
                 if generic_api == "responses" else
                 _t021_chat_reply(model, f"generic-{generic_api}", *generic_usage)),
    }
    return routes


def _t021_install_io(monkeypatch, lane, routes, *, async_barrier=None):
    """Mock only the backend client while leaving shared transports and decoders active."""
    import httpx
    import requests
    from llm_api_adapter.llms import requests_transport

    calls = []

    def routed(url, headers, payload, timeout, streaming):
        snapshot = copy.deepcopy(payload)
        calls.append(("send", "POST", url, dict(headers), snapshot, timeout, streaming))
        route = routes[(url, headers["Authorization"])]
        assert timeout == route["timeout"]
        if route.get("error_status"):
            response = requests.Response()
            response.status_code = route["error_status"]
            response._content = b'{"error":{"message":"synthetic unavailable"}}'
            request = requests.Request("POST", url).prepare()
            response.request = request
            error = requests.HTTPError("synthetic unavailable", response=response, request=request)
            raise error
        return _T021HTTPResponse(copy.deepcopy(route["body"]))

    def requests_post(url, *, headers, json, timeout=None, stream=False):
        return routed(url, headers, json, timeout, stream)

    if lane.endswith("requests"):
        monkeypatch.setattr(requests_transport.requests, "post", requests_post)
        return calls

    if lane in {"chat-httpx", "stream-httpx"}:
        class SyncClient:
            def post(self, url, *, headers, json, timeout):
                return routed(url, headers, json, timeout, False)

            def build_request(self, method, url, *, headers, json, timeout):
                calls.append(("build", method, url, dict(headers), copy.deepcopy(json), timeout, True))
                return (method, url, headers, json, timeout)

            def send(self, request, *, stream):
                method, url, headers, payload, timeout = request
                assert method == "POST"
                return routed(url, headers, payload, timeout, stream)

            def close(self):
                return None

        monkeypatch.setattr(httpx, "Client", SyncClient)
        return calls

    class AsyncClient:
        async def post(self, url, *, headers, json, timeout):
            if async_barrier is not None:
                await async_barrier(url, headers, json, timeout, False)
            return routed(url, headers, json, timeout, False)

        def build_request(self, method, url, *, headers, json, timeout):
            calls.append(("build", method, url, dict(headers), copy.deepcopy(json), timeout, True))
            return (method, url, headers, json, timeout)

        async def send(self, request, *, stream):
            method, url, headers, payload, timeout = request
            assert method == "POST"
            if async_barrier is not None:
                await async_barrier(url, headers, payload, timeout, True)
            return routed(url, headers, payload, timeout, stream)

        async def aclose(self):
            return None

    monkeypatch.setattr(httpx, "AsyncClient", AsyncClient)
    return calls


def _t021_generic_adapter(model, generic_api, *, pricing=None):
    return OpenAICompatibleAdapter(
        company="openai_compatible", api_key=_T021_GENERIC_KEYS[generic_api], model=model,
        base_url=_T021_GENERIC_ROOTS[generic_api], api_variant=generic_api, pricing=pricing,
    )


def _t021_expected_history(user_text, assistant_text):
    """Return exact ordered generic/native-compatible history for either wire protocol."""
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def _t021_call(adapter, lane, label, *, callbacks=None, previous_response=None):
    callbacks = callbacks if callbacks is not None else []
    messages = [UserMessage(f"{label} user history"), AIMessage(f"{label} assistant history")]
    timeout = 2.25 if label in {"native-zai", "native-control"} else 3.75
    if lane.startswith("chat-"):
        return adapter.chat(messages, timeout_s=timeout, previous_response=previous_response)
    if lane == "achat":
        return asyncio.run(adapter.achat(messages, timeout_s=timeout, previous_response=previous_response))
    if lane.startswith("stream-"):
        chunks = list(adapter.stream_chat(
            messages, timeout_s=timeout, previous_response=previous_response,
            on_done=callbacks.append,
        ))
    else:
        async def collect():
            return [item async for item in adapter.astream_chat(
                messages, timeout_s=timeout, previous_response=previous_response,
                on_done=callbacks.append,
            )]
        chunks = asyncio.run(collect())
    assert chunks == [f"{label} text"]
    assert len(callbacks) == 1
    return callbacks[0]


def _t021_expected_native_cost(model, input_tokens, output_tokens, cached_tokens):
    """Calculate exact metadata-derived native cost independently of production code."""
    tier = _zai_raw_model_metadata(model)["pricing_tiers"][0]
    input_cost = ((input_tokens - cached_tokens) * tier["input_per_1m"]
                  + cached_tokens * tier["cache_read_input_per_1m"]) / 1_000_000
    output_cost = output_tokens * tier["output_per_1m"] / 1_000_000
    return input_cost, output_cost, input_cost + output_cost


def _t021_assert_native_result(response, model, label, *, pricing=None):
    input_tokens, output_tokens, cached_tokens = (12, 4, 3)
    expected_costs = (
        _t021_expected_native_cost(model, input_tokens, output_tokens, cached_tokens)
        if pricing is None else (
            (input_tokens - cached_tokens) * pricing[0] + cached_tokens * pricing[2],
            output_tokens * pricing[1],
            (input_tokens - cached_tokens) * pricing[0] + cached_tokens * pricing[2]
            + output_tokens * pricing[1],
        )
    )
    assert (response.content, response.response_id, response.model, response.finish_reason) == (
        f"{label} text", f"{label}-chat-id", model, "stop",
    )
    assert isinstance(response.usage, ZaiUsage)
    assert _usage_tuple(response.usage) == (12, 4, 16, 3, None)
    assert response.currency == ("OWN" if pricing is not None else PLUGIN.model_metadata.organization_data["currency"])
    assert (response.cost_input, response.cost_output, response.cost_total) == pytest.approx(expected_costs)


def _t021_assert_generic_result(response, model, generic_api, label, *, pricing=None):
    assert (response.content, response.response_id, response.model, response.finish_reason) == (
        f"{label} text",
        f"{label}-responses-id" if generic_api == "responses" else f"{label}-chat-id",
        model,
        "completed" if generic_api == "responses" else "stop",
    )
    assert type(response.usage) is not ZaiUsage
    counts = (23, 5, 28) if generic_api == "chat_completions" else (23, 5, 28)
    assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens) == counts
    if pricing is None:
        assert response.currency is None
        assert (response.cost_input, response.cost_output, response.cost_total) == (None, None, None)
    else:
        input_cost, output_cost = 23 * pricing[0], 5 * pricing[1]
        assert response.currency == "CALL"
        assert (response.cost_input, response.cost_output, response.cost_total) == pytest.approx(
            (input_cost, output_cost, input_cost + output_cost)
        )


def _t021_assert_trace(calls, lane, records):
    """Verify every build/send observation, request route, identity and timeout."""
    expected_kinds = {
        "chat-requests": ["send"], "chat-httpx": ["send"], "achat": ["send"],
        "stream-requests": ["send"], "stream-httpx": ["build", "send"],
        "astream": ["build", "send"],
    }[lane]
    expected = {(record["url"], f"Bearer {record['key']}", kind): 0
                for record in records for kind in expected_kinds}
    assert len(calls) == len(records) * len(expected_kinds)
    seen = []
    for call in calls:
        _, method, url, headers, payload, timeout, streaming = call
        key = headers["Authorization"]
        record = next(item for item in records if item["url"] == url and f"Bearer {item['key']}" == key)
        kind = call[0]
        assert (url, key, kind) in expected and method == "POST"
        assert headers == {"Authorization": key, "Content-Type": "application/json"}
        expected[(url, key, kind)] += 1
        seen.append((url, key, kind))
        assert timeout == record["timeout"] and streaming is record["stream"]
        assert payload["model"] == record["model"]
        assert payload.get("stream") is (True if record["stream"] else None)
        history_key = "input" if url.endswith("/responses") else "messages"
        assert payload[history_key] == _t021_expected_history(
            record["user_history"], record["assistant_history"],
        )
        assert "previous_response_id" not in payload
        assert "foreign-opaque-id" not in repr(payload)
        assert "foreign-secret" not in repr(payload)
        assert "opaque-foreign-data" not in repr(payload)
        if url.endswith("/responses"):
            assert payload["store"] is False
            assert not ({"stream_options", "max_tokens", "max_completion_tokens", "reasoning_effort"}
                        & payload.keys())
        elif record["stream"]:
            assert payload["stream_options"] == {"include_usage": True}
    assert set(expected.values()) == {1}
    if lane in {"stream-httpx", "astream"}:
        for record in records:
            identity = (record["url"], f"Bearer {record['key']}")
            assert seen.index((*identity, "build")) < seen.index((*identity, "send"))


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
def test_zai_native_profile_tariff_and_generic_caller_pricing_remain_isolated(
    zai_adapter_runtime, monkeypatch, model,
):
    """Real native sends retain registry/override rates while generic caller rates stay local."""
    from llm_api_adapter.llms import requests_transport

    native, _ = _native_adapter("direct", model)
    control, _ = _native_adapter("direct", model)
    control.api_key = "native-control-key"
    generic_chat = _t021_generic_adapter(model, "chat_completions")
    generic_responses = _t021_generic_adapter(model, "responses")
    raw_model = copy.deepcopy(_zai_raw_model_metadata(model))
    raw_org = copy.deepcopy(PLUGIN.model_metadata.organization_data)
    registry_snapshot = copy.deepcopy(zai_adapter_runtime.organizations["zai"].models[model].pricing_tiers)
    native_profile = native.model_spec
    native_override = Pricing((PricingTier(None, 3e-6, 5e-6, 1e-6),), currency="OWN")
    caller_rates = (7e-6, 11e-6)
    caller_pricing = Pricing((PricingTier(None, *caller_rates),), currency="CALL")
    generic_responses.pricing = caller_pricing
    foreign_previous = ChatResponse(
        content="foreign response", model="foreign-model", response_id="foreign-opaque-id",
        provider_data={"credential": "foreign-secret", "opaque": "opaque-foreign-data"},
    )
    native_usage = {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16,
                    "prompt_tokens_details": {"cached_tokens": 3}}
    routes = {}
    records = []

    def route(url, key, label, body, timeout, user, assistant):
        routes[(url, f"Bearer {key}")] = {"label": label, "body": body, "timeout": timeout}
        records.append({"url": url, "key": key, "label": label, "model": model,
                        "timeout": timeout, "stream": False,
                        "user_history": user, "assistant_history": assistant})

    route(ZAI_CHAT_COMPLETIONS_URL, "synthetic-zai-key", "native-zai",
          _t021_chat_reply(model, "native-zai", 12, 4, 3), 2.25,
          "native-zai user history", "native-zai assistant history")
    route(ZAI_CHAT_COMPLETIONS_URL, "native-control-key", "native-zai",
          _t021_chat_reply(model, "native-zai", 12, 4, 3), 2.25,
          "native-zai user history", "native-zai assistant history")
    route(f"{_T021_GENERIC_ROOTS['chat_completions']}/chat/completions",
          _T021_GENERIC_KEYS["chat_completions"],
          "generic-chat_completions", _t021_chat_reply(model, "generic-chat_completions", 23, 5),
          3.75, "generic-chat_completions user history", "generic-chat_completions assistant history")
    route(f"{_T021_GENERIC_ROOTS['responses']}/responses",
          _T021_GENERIC_KEYS["responses"],
          "generic-responses", _t021_responses_reply(model, "generic-responses", 23, 5),
          3.75, "generic-responses user history", "generic-responses assistant history")
    calls = []

    def send(url, headers, payload, timeout, stream):
        calls.append(("send", "POST", url, copy.deepcopy(headers), copy.deepcopy(payload), timeout, stream))
        route_spec = routes[(url, headers["Authorization"])]
        assert timeout == route_spec["timeout"]
        return _T021HTTPResponse(copy.deepcopy(route_spec["body"]))

    monkeypatch.setattr(requests_transport.requests, "post", lambda url, *, headers, json, timeout=None, stream=False:
                        send(url, headers, json, timeout, stream))
    native.pricing = native_override
    r_native_1 = native.chat([UserMessage("native-zai user history"), AIMessage("native-zai assistant history")],
                             timeout_s=2.25, previous_response=foreign_previous)
    r_control_1 = control.chat([UserMessage("native-zai user history"), AIMessage("native-zai assistant history")],
                               timeout_s=2.25, previous_response=foreign_previous)
    r_generic_chat_1 = generic_chat.chat([UserMessage("generic-chat_completions user history"),
                                          AIMessage("generic-chat_completions assistant history")],
                                        timeout_s=3.75, previous_response=foreign_previous)
    r_generic_response_1 = generic_responses.chat([UserMessage("generic-responses user history"),
                                                    AIMessage("generic-responses assistant history")],
                                                  timeout_s=3.75, previous_response=foreign_previous)
    _t021_assert_native_result(r_native_1, model, "native-zai",
                               pricing=(3e-6, 5e-6, 1e-6))
    _t021_assert_native_result(r_control_1, model, "native-zai")
    _t021_assert_generic_result(r_generic_chat_1, model, "chat_completions", "generic-chat_completions")
    _t021_assert_generic_result(r_generic_response_1, model, "responses", "generic-responses",
                                pricing=caller_rates)

    caller_pricing.set_in_per_1m(14.0)
    caller_pricing.set_out_per_1m(22.0)
    r_generic_response_2 = generic_responses.chat([UserMessage("generic-responses user history"),
                                                    AIMessage("generic-responses assistant history")],
                                                  timeout_s=3.75, previous_response=foreign_previous)
    r_native_2 = native.chat([UserMessage("native-zai user history"), AIMessage("native-zai assistant history")],
                             timeout_s=2.25, previous_response=foreign_previous)
    r_control_2 = control.chat([UserMessage("native-zai user history"), AIMessage("native-zai assistant history")],
                               timeout_s=2.25, previous_response=foreign_previous)
    r_generic_chat_2 = generic_chat.chat([UserMessage("generic-chat_completions user history"),
                                          AIMessage("generic-chat_completions assistant history")],
                                        timeout_s=3.75, previous_response=foreign_previous)
    mutated_rates = (14e-6, 22e-6)
    _t021_assert_generic_result(r_generic_response_2, model, "responses", "generic-responses",
                                pricing=mutated_rates)
    _t021_assert_native_result(r_native_2, model, "native-zai", pricing=(3e-6, 5e-6, 1e-6))
    _t021_assert_native_result(r_control_2, model, "native-zai")
    _t021_assert_generic_result(r_generic_chat_2, model, "chat_completions", "generic-chat_completions")

    assert native.model_spec is native_profile
    assert native.model_spec.limits.context_window_tokens == raw_model["limits"]["context_window_tokens"]
    assert native.model_spec.limits.max_output_tokens == raw_model["limits"]["max_output_tokens"]
    native_profile_tier = native.model_spec.pricing_tiers.tiers[0]
    assert native_profile_tier.in_per_token == pytest.approx(
        raw_model["pricing_tiers"][0]["input_per_1m"] / 1_000_000
    )
    assert native_profile_tier.out_per_token == pytest.approx(
        raw_model["pricing_tiers"][0]["output_per_1m"] / 1_000_000
    )
    assert native_profile_tier.cache_read_in_per_token == pytest.approx(
        raw_model["pricing_tiers"][0]["cache_read_input_per_1m"] / 1_000_000
    )
    assert generic_chat.model_spec is None and generic_chat.pricing is None
    assert generic_responses.model_spec is None and generic_responses.pricing is caller_pricing
    assert control.pricing.currency == raw_org["currency"]
    assert zai_adapter_runtime.organizations["zai"].models[model].pricing_tiers == registry_snapshot
    assert len(calls) == 8
    labels = ["native-zai", "native-zai", "generic-chat_completions", "generic-responses",
              "generic-responses", "native-zai", "native-zai", "generic-chat_completions"]
    expected_keys = ["synthetic-zai-key", "native-control-key",
                     _T021_GENERIC_KEYS["chat_completions"],
                     _T021_GENERIC_KEYS["responses"], _T021_GENERIC_KEYS["responses"],
                     "synthetic-zai-key", "native-control-key",
                     _T021_GENERIC_KEYS["chat_completions"]]
    for call, label, key in zip(calls, labels, expected_keys):
        _, method, url, headers, payload, timeout, stream = call
        assert method == "POST" and stream is False
        assert headers["Authorization"] == f"Bearer {key}"
        expected_url = {
            "native-zai": ZAI_CHAT_COMPLETIONS_URL,
            "generic-chat_completions": f"{_T021_GENERIC_ROOTS['chat_completions']}/chat/completions",
            "generic-responses": f"{_T021_GENERIC_ROOTS['responses']}/responses",
        }[label]
        assert url == expected_url
        assert timeout == (2.25 if key in {"synthetic-zai-key", "native-control-key"} else 3.75)
        assert payload["model"] == model
        assert "foreign-opaque-id" not in repr(payload)
        assert "foreign-secret" not in repr(payload)
        assert "opaque-foreign-data" not in repr(payload)
        assert "previous_response_id" not in payload
        record = next(record for record in records if record["url"] == url and record["key"] == key)
        history_key = "input" if url.endswith("/responses") else "messages"
        assert payload[history_key] == _t021_expected_history(
            record["user_history"], record["assistant_history"],
        )
        if url.endswith("/responses"):
            assert payload["store"] is False and "previous_response_id" not in payload
            assert not ({"stream_options", "max_tokens", "max_completion_tokens", "reasoning_effort"}
                        & payload.keys())
        elif stream:
            assert payload["stream_options"] == {"include_usage": True}


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("lane", _T021_LANES)
@pytest.mark.parametrize("generic_api", ["chat_completions", "responses"])
@pytest.mark.parametrize("pricing_mode", ["unpriced", "caller-priced"])
def test_zai_native_and_generic_profiles_are_isolated_across_execution_lanes(
    zai_adapter_runtime, monkeypatch, model, lane, generic_api, pricing_mode,
):
    """Distinct native and generic results stay isolated across all methods/transports."""
    native, _ = _native_adapter("direct", model, "httpx" if lane.endswith("httpx") else "requests")
    caller_rates = (13e-6, 17e-6)
    generic = OpenAICompatibleAdapter(
        company="openai_compatible", api_key=_T021_GENERIC_KEYS[generic_api], model=model,
        base_url=_T021_GENERIC_ROOTS[generic_api], api_variant=generic_api,
        pricing=(Pricing((PricingTier(None, *caller_rates),), currency="CALL")
                 if pricing_mode == "caller-priced" else None),
        transport="httpx" if lane.endswith("httpx") else "requests",
    )
    native_profile = native.model_spec
    native_records = [{"url": ZAI_CHAT_COMPLETIONS_URL, "key": "synthetic-zai-key", "model": model,
                       "timeout": 2.25, "stream": lane.startswith("stream") or lane == "astream",
                       "user_history": "native-zai user history",
                       "assistant_history": "native-zai assistant history"}]
    generic_url = f"{_T021_GENERIC_ROOTS[generic_api]}/{'responses' if generic_api == 'responses' else 'chat/completions'}"
    generic_records = [{"url": generic_url, "key": _T021_GENERIC_KEYS[generic_api], "model": model,
                        "timeout": 3.75, "stream": lane.startswith("stream") or lane == "astream",
                        "user_history": f"generic-{generic_api} user history",
                        "assistant_history": f"generic-{generic_api} assistant history"}]
    routes = _t021_routes(model, generic_api)
    barrier_state = {"arrivals": [], "event": None, "release": None}

    async def overlap_barrier(url, headers, payload, timeout, streaming):
        barrier_state["arrivals"].append((url, headers["Authorization"], copy.deepcopy(payload), timeout, streaming))
        if len(barrier_state["arrivals"]) == 2:
            barrier_state["event"].set()
        await asyncio.wait_for(barrier_state["release"].wait(), timeout=2)

    barrier = overlap_barrier if lane in {"achat", "astream"} else None
    barrier_state["event"] = asyncio.Event()
    barrier_state["release"] = asyncio.Event()
    calls = _t021_install_io(monkeypatch, lane, routes, async_barrier=barrier)
    native_callbacks, generic_callbacks = [], []
    previous = ChatResponse(
        content="foreign", model="foreign-model", response_id="foreign-opaque-id",
        provider_data={"key": "foreign-secret", "sentinel": "opaque-foreign-data"},
    )

    if barrier is None:
        native_result = _t021_call(native, lane, "native-zai", callbacks=native_callbacks,
                                   previous_response=previous)
        generic_result = _t021_call(generic, lane, f"generic-{generic_api}", callbacks=generic_callbacks,
                                    previous_response=previous)
    else:
        async def overlapping_calls():
            if lane == "astream":
                async def collect(adapter, label, callbacks, timeout):
                    messages = [UserMessage(f"{label} user history"), AIMessage(f"{label} assistant history")]
                    return [item async for item in adapter.astream_chat(
                        messages, timeout_s=timeout, previous_response=previous, on_done=callbacks.append,
                    )]
                native_task = asyncio.create_task(collect(native, "native-zai", native_callbacks, 2.25))
                generic_task = asyncio.create_task(collect(generic, f"generic-{generic_api}", generic_callbacks, 3.75))
            else:
                native_task = asyncio.create_task(native.achat(
                    [UserMessage("native-zai user history"), AIMessage("native-zai assistant history")],
                    timeout_s=2.25, previous_response=previous,
                ))
                generic_task = asyncio.create_task(generic.achat(
                    [UserMessage(f"generic-{generic_api} user history"),
                     AIMessage(f"generic-{generic_api} assistant history")],
                    timeout_s=3.75, previous_response=previous,
                ))
            await asyncio.wait_for(barrier_state["event"].wait(), timeout=2)
            assert len(barrier_state["arrivals"]) == 2
            assert not native_task.done() and not generic_task.done()
            assert native_callbacks == generic_callbacks == []
            arrivals = {(entry[0], entry[1]): entry for entry in barrier_state["arrivals"]}
            expected_arrivals = {
                (native_records[0]["url"], "Bearer synthetic-zai-key"): (
                    native_records[0], "native-zai", 2.25,
                ),
                (generic_records[0]["url"], f"Bearer {_T021_GENERIC_KEYS[generic_api]}"): (
                    generic_records[0], f"generic-{generic_api}", 3.75,
                ),
            }
            assert set(arrivals) == set(expected_arrivals)
            for identity, (record, label, expected_timeout) in expected_arrivals.items():
                url, key, payload, actual_timeout, streaming = arrivals[identity]
                assert url == record["url"] and key == f"Bearer {record['key']}"
                assert actual_timeout == expected_timeout
                assert streaming is (lane == "astream")
                assert payload["model"] == model
                history_key = "input" if url.endswith("/responses") else "messages"
                assert payload[history_key] == _t021_expected_history(
                    f"{label} user history", f"{label} assistant history",
                )
                assert "previous_response_id" not in payload
                assert "foreign-opaque-id" not in repr(payload)
                assert "foreign-secret" not in repr(payload)
                assert "opaque-foreign-data" not in repr(payload)
                assert payload.get("stream") is (True if lane == "astream" else None)
                if url.endswith("/responses"):
                    assert payload["store"] is False
                elif lane == "astream":
                    assert payload["stream_options"] == {"include_usage": True}
            barrier_state["release"].set()
            if lane == "astream":
                native_chunks, generic_chunks = await asyncio.gather(native_task, generic_task)
                assert native_chunks == ["native-zai text"] and generic_chunks == [f"generic-{generic_api} text"]
                return native_callbacks[0], generic_callbacks[0]
            return await asyncio.gather(native_task, generic_task)
        native_result, generic_result = asyncio.run(overlapping_calls())

    _t021_assert_native_result(native_result, model, "native-zai")
    _t021_assert_generic_result(
        generic_result, model, generic_api, f"generic-{generic_api}",
        pricing=caller_rates if pricing_mode == "caller-priced" else None,
    )
    assert native.model_spec is native_profile and generic.model_spec is None
    assert generic.is_reasoning is False and generic.is_adaptive_thinking is False
    if pricing_mode == "caller-priced":
        assert generic.pricing is not None
    else:
        assert generic.pricing is None
    if lane.startswith("stream") or lane == "astream":
        assert native_callbacks == [native_result]
        assert generic_callbacks == [generic_result]
    else:
        assert native_callbacks == generic_callbacks == []
    _t021_assert_trace(calls, lane, native_records + generic_records)


@pytest.mark.unit
@pytest.mark.parametrize("model", ZAI_MODELS)
@pytest.mark.parametrize("generic_api", ["chat_completions", "responses"])
def test_generic_failure_recovers_without_fallback_or_native_policy_changes(
    zai_adapter_runtime, monkeypatch, model, generic_api,
):
    """A mapped generic 503 is one attempt; native and same generic instance then succeed."""
    from llm_api_adapter.errors.llm_api_error import LLMAPIServerError

    generic = _t021_generic_adapter(model, generic_api)
    native, _ = _native_adapter("direct", model)
    native_profile = native.model_spec
    routes = _t021_routes(model, generic_api)
    generic_key = (f"{_T021_GENERIC_ROOTS[generic_api]}/"
                   f"{'responses' if generic_api == 'responses' else 'chat/completions'}",
                   f"Bearer {_T021_GENERIC_KEYS[generic_api]}")
    routes[generic_key]["error_status"] = 503
    calls = _t021_install_io(monkeypatch, "chat-requests", routes)
    messages = [UserMessage("generic recovery history"), AIMessage("generic prior turn")]
    with pytest.raises(LLMAPIServerError):
        generic.chat(messages, timeout_s=3.75)
    routes[generic_key].pop("error_status")
    native_result = native.chat(
        [UserMessage("native recovery history"), AIMessage("native prior turn")], timeout_s=2.25,
    )
    response = generic.chat(messages, timeout_s=3.75)
    _t021_assert_native_result(native_result, model, "native-zai")
    _t021_assert_generic_result(response, model, generic_api, f"generic-{generic_api}")
    assert native.model_spec is native_profile
    assert generic.model_spec is None and generic.pricing is None
    assert response.currency is None
    assert len(calls) == 3
    expected = [
        (generic_key[0], _T021_GENERIC_KEYS[generic_api], 3.75, "generic recovery history", "generic prior turn"),
        (ZAI_CHAT_COMPLETIONS_URL, "synthetic-zai-key", 2.25, "native recovery history", "native prior turn"),
        (generic_key[0], _T021_GENERIC_KEYS[generic_api], 3.75, "generic recovery history", "generic prior turn"),
    ]
    assert [(call[2], call[3]["Authorization"], call[5]) for call in calls] == [
        (url, f"Bearer {key}", timeout) for url, key, timeout, _, _ in expected
    ]
    for call, (expected_url, expected_key, expected_timeout, expected_user, expected_assistant) in zip(calls, expected):
        _, method, url, headers, payload, timeout, stream = call
        assert method == "POST" and url == expected_url
        assert headers["Authorization"] == f"Bearer {expected_key}"
        assert payload["model"] == model and timeout == expected_timeout and stream is False
        history_key = "input" if url.endswith("/responses") else "messages"
        assert payload[history_key] == _t021_expected_history(expected_user, expected_assistant)
        assert "foreign-opaque-id" not in repr(payload)
        assert "foreign-secret" not in repr(payload)
        assert "previous_response_id" not in payload
        if url.endswith("/responses"):
            assert payload["store"] is False and "previous_response_id" not in payload
