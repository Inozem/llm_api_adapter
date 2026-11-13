"""Credential-free legacy and shared-engine contracts for Z.ai clients."""

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
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[4]
CORE_SOURCE = ROOT / "src"
PACKAGE_SOURCE = ROOT / "packages/organizations/zai/src"
for source in (str(PACKAGE_SOURCE), str(CORE_SOURCE)):
    if source not in sys.path:
        sys.path.insert(0, source)

from llm_api_adapter.errors.llm_api_error import (
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
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent, SyncTransport, TransportRequest
from llm_api_adapter.llms.requests_transport import RequestsSyncTransport
from llm_api_adapter.llms.httpx_transport import HttpxSyncTransport
from llm_api_adapter.llms.openai_compatible.sync_client import OpenAICompatibleSyncClient
from llm_api_adapter.llms.openai_compatible.async_client import _AsyncExecutionMixin
from llm_api_adapter_zai.clients import (
    ZAI_BASE_URL,
    ZAI_CHAT_COMPLETIONS_URL,
    ZaiAsyncClient,
    ZaiChatCompletionsAsyncClient,
    ZaiChatCompletionsSyncClient,
    ZaiSyncClient,
)
from llm_api_adapter_zai import clients as clients_package
from llm_api_adapter_zai.clients import async_client as async_client_module
from llm_api_adapter_zai.clients import sync_client as sync_client_module
from llm_api_adapter_zai.adapter import ZaiAdapter
from llm_api_adapter_zai.plugin import PLUGIN
import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.models.messages.file_parts import DocumentPart
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.models.tools.tool_spec import ToolSpec
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
def test_zai_clients_keep_exports_keyword_forms_and_raw_payload_semantics():
    assert ZaiChatCompletionsSyncClient is ZaiSyncClient
    assert ZaiChatCompletionsAsyncClient is ZaiAsyncClient
    assert clients_package.__all__ == [
        "ZAI_BASE_URL", "ZAI_CHAT_COMPLETIONS_URL", "ZaiAsyncClient",
        "ZaiChatCompletionsAsyncClient", "ZaiChatCompletionsSyncClient", "ZaiSyncClient",
    ]
    assert clients_package.ZAI_BASE_URL == ZAI_BASE_URL
    assert clients_package.ZAI_CHAT_COMPLETIONS_URL == ZAI_CHAT_COMPLETIONS_URL
    assert sync_client_module.ZAI_BASE_URL == ZAI_BASE_URL
    assert sync_client_module.ZAI_CHAT_COMPLETIONS_URL == ZAI_CHAT_COMPLETIONS_URL
    transport = RecordingTransport()
    client = ZaiSyncClient(transport)
    payload = {"max_tokens": 19, "temperature": 1.0}
    original_payload = copy.deepcopy(payload)
    assert client.chat(api_key="unit-key", payload=payload, timeout_s=7) == {"ok": True}
    request = transport.requests[-1]
    assert request.url == ZAI_CHAT_COMPLETIONS_URL
    assert request.headers_dict() == {
        "Authorization": "Bearer unit-key", "Content-Type": "application/json"
    }
    assert request.timeout == 7 and request.payload == original_payload
    assert payload == original_payload

    replacement = RecordingTransport()
    client.transport = replacement
    client.chat(api_key="replacement key", payload={}, timeout_s=9)
    assert replacement.requests[0].headers_dict()["Authorization"] == "Bearer replacement key"
    assert replacement.requests[0].timeout == 9

    stream_cases = [
        ({"max_tokens": 11, "temperature": 1.0}, {"include_usage": True}),
        ({"stream_options": {"include_usage": False}, "max_tokens": 11, "temperature": 1.0}, {"include_usage": False}),
        ({"stream_options": {}, "max_tokens": 11, "temperature": 1.0}, {}),
        ({"stream_options": None, "max_tokens": 11, "temperature": 1.0}, None),
    ]
    for raw, expected_options in stream_cases:
        snapshot = copy.deepcopy(raw)
        expected_options = copy.deepcopy(expected_options)
        list(client.stream(api_key="other-key", payload=raw, timeout_s=None))
        streamed = replacement.requests[-1].payload
        assert streamed["stream"] is True and streamed["max_tokens"] == 11
        assert streamed["temperature"] == 1.0
        assert streamed["stream_options"] == expected_options
        assert raw == snapshot


@pytest.mark.unit
def test_zai_legacy_async_client_preserves_keyword_forms_and_live_patch_points(monkeypatch):
    seen = []
    async def fake_request(url, **kwargs):
        seen.append(("chat", url, kwargs))
        return {"ok": True}
    async def fake_stream(url, **kwargs):
        seen.append(("stream", url, kwargs))
        yield SSEEvent(None, data={"ok": True})
    client = ZaiAsyncClient()
    monkeypatch.setattr(async_client_module, "async_request", fake_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", fake_stream)
    payload = {"temperature": 1.0, "max_tokens": 12}
    payload_snapshot = copy.deepcopy(payload)
    stream_payloads = [
        {},
        {"stream_options": {"include_usage": False}},
        {"stream_options": {}},
        {"stream_options": None},
    ]
    stream_snapshots = [copy.deepcopy(item) for item in stream_payloads]
    stream_expected_options = [copy.deepcopy(item.get("stream_options", {"include_usage": True})) for item in stream_snapshots]
    async def exercise():
        assert await client.chat(api_key="async legacy", payload=payload, timeout_s=6) == {"ok": True}
        for stream_payload in stream_payloads:
            assert [event async for event in client.stream(
                api_key="async legacy", payload=stream_payload, timeout_s=8
            )]
    asyncio.run(exercise())
    assert payload == payload_snapshot and stream_payloads == stream_snapshots
    assert seen[0][1] == ZAI_CHAT_COMPLETIONS_URL
    assert seen[0][2]["headers"]["Authorization"] == "Bearer async legacy"
    assert seen[0][2]["payload"] == payload_snapshot and seen[0][2]["timeout"] == 6
    assert [item[2]["payload"]["stream_options"] for item in seen[1:]] == [
        *stream_expected_options
    ]
    assert all(item[2]["payload"]["stream"] is True for item in seen[1:])
    assert all(item[2]["headers"]["Authorization"] == "Bearer async legacy" for item in seen[1:])
    assert all(item[2]["timeout"] == 8 for item in seen[1:])


@pytest.mark.unit
@pytest.mark.parametrize("body", [None, [], "text", 3])
def test_zai_legacy_client_rejects_non_object_json(body):
    with pytest.raises(LLMAPIClientError):
        ZaiSyncClient(RecordingTransport(body=body)).chat(
            api_key="synthetic", payload={}, timeout_s=None
        )


@pytest.mark.unit
@pytest.mark.parametrize("body", [None, [], "text", 3])
def test_zai_legacy_async_client_rejects_non_object_json(monkeypatch, body):
    async def fake_request(url, **kwargs):
        return body
    monkeypatch.setattr(async_client_module, "async_request", fake_request)
    with pytest.raises(LLMAPIClientError):
        asyncio.run(
            ZaiAsyncClient().chat(api_key="synthetic", payload={}, timeout_s=None)
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
        ZaiSyncClient(RecordingTransport(http_error=HTTPError("synthetic"))).chat(
            api_key="synthetic", payload={}, timeout_s=None
        )


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
    client = ZaiSyncClient(RecordingTransport(events=[event]))
    with pytest.raises(error):
        list(client.stream(api_key="synthetic", payload={}, timeout_s=None))


@pytest.mark.unit
def test_zai_factories_bind_credentials_and_reuse_the_sync_engine(monkeypatch):
    """One adapter call or legacy client invocation; never mutated for another call."""
    transport = RecordingTransport()
    wrapper = ZaiSyncClient(transport)
    calls = []
    original = OpenAICompatibleSyncClient._send_request
    original_stream = OpenAICompatibleSyncClient._stream_request
    stream_calls = []
    def observed(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)
    def observed_stream(self, *args, **kwargs):
        stream_calls.append(self)
        return original_stream(self, *args, **kwargs)
    monkeypatch.setattr(OpenAICompatibleSyncClient, "_send_request", observed)
    monkeypatch.setattr(OpenAICompatibleSyncClient, "_stream_request", observed_stream)

    first = wrapper._for_request("first synthetic")
    second = wrapper._for_request("second synthetic")
    first.complete(model="glm-5.3-flash", timeout=1, max_tokens=13, temperature=1.0)
    list(second.stream(model="glm-5.3-flash", timeout=2, max_tokens=13))
    first.complete(model="glm-5.3-flash", timeout=3, max_tokens=14)
    replacement = RecordingTransport()
    wrapper.transport = replacement
    wrapper.chat(
        api_key="legacy synthetic", payload={"max_tokens": 9}, timeout_s=6
    )
    list(wrapper.stream(
        api_key="legacy synthetic", payload={"max_tokens": 9}, timeout_s=7
    ))
    third = wrapper._for_request("replacement factory")
    third.complete(model="glm-5.3-flash", timeout=8, max_tokens=10)
    assert calls and len(calls) == 4 and len(stream_calls) == 2
    assert [r.headers_dict()["Authorization"] for r in transport.requests] == [
        "Bearer first synthetic", "Bearer second synthetic", "Bearer first synthetic"
    ]
    assert [r.headers_dict()["Authorization"] for r in replacement.requests] == [
        "Bearer legacy synthetic", "Bearer legacy synthetic", "Bearer replacement factory"
    ]
    assert all(r.url == ZAI_CHAT_COMPLETIONS_URL for r in transport.requests + replacement.requests)
    assert [r.timeout for r in transport.requests] == [1, 2, 3]
    assert [r.payload.get("max_tokens") for r in transport.requests] == [13, 13, 14]
    assert [r.timeout for r in replacement.requests] == [6, 7, 8]
    assert all(engine._sync_transport is transport for engine in calls[:2])
    assert all(engine._sync_transport is replacement for engine in calls[2:])

    invalid = ZaiSyncClient(RecordingTransport(body=[]))
    with pytest.raises(LLMAPIClientError):
        invalid._for_request("invalid bound response").complete(
            model="glm-5.3-flash", max_tokens=1
        )
    with pytest.raises(LLMAPIClientError):
        invalid.chat(api_key="invalid legacy response", payload={}, timeout_s=None)


@pytest.mark.unit
def test_zai_async_factory_uses_shared_async_execution_and_live_patch_points(monkeypatch):
    wrapper = ZaiAsyncClient()
    first = wrapper._for_request("async synthetic")
    second = wrapper._for_request("second async synthetic")
    calls = []
    response_body = [{"ok": True}]
    async def fake_request(url, **kwargs):
        calls.append(("json", url, kwargs))
        return response_body[0]
    async def fake_stream(url, **kwargs):
        calls.append(("sse", url, kwargs))
        yield SSEEvent(None, data={"ok": True})
    monkeypatch.setattr(async_client_module, "async_request", fake_request)
    monkeypatch.setattr(async_client_module, "async_stream_request", fake_stream)

    class GuardAsyncClient:
        async def post(self, *args, **kwargs):
            raise AssertionError("stale async patch point attempted provider I/O")
        def build_request(self, *args, **kwargs):
            return object()
        async def send(self, *args, **kwargs):
            raise AssertionError("stale async stream patch point attempted provider I/O")
        async def aclose(self):
            return None
    class GuardHTTPX:
        AsyncClient = GuardAsyncClient
        TimeoutException = type("TimeoutException", (Exception,), {})
        HTTPStatusError = type("HTTPStatusError", (Exception,), {})
        RequestError = type("RequestError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "httpx", GuardHTTPX)
    mixin_send = _AsyncExecutionMixin._send_request
    mixin_stream = _AsyncExecutionMixin._stream_request
    send_calls, stream_calls = [], []
    async def send_seen(self, *a, **kw):
        send_calls.append(self)
        return await mixin_send(self, *a, **kw)
    async def stream_seen(self, *a, **kw):
        stream_calls.append(self)
        async for event in mixin_stream(self, *a, **kw): yield event
    monkeypatch.setattr(_AsyncExecutionMixin, "_send_request", send_seen)
    monkeypatch.setattr(_AsyncExecutionMixin, "_stream_request", stream_seen)

    async def exercise():
        assert await first.complete(model="glm-5.3-flash", timeout=4, max_tokens=7) == {"ok": True}
        assert [e async for e in second.stream(model="glm-5.3-flash", timeout=5, max_tokens=8)]
        assert await first.complete(model="glm-5.3-flash", timeout=6, max_tokens=9) == {"ok": True}
        assert [e async for e in first.stream(model="glm-5.3-flash", timeout=7, max_tokens=10)]
        assert await wrapper.chat(
            api_key="legacy async", payload={"max_tokens": 6}, timeout_s=9
        ) == {"ok": True}
        legacy_stream_payload = {"stream_options": {"include_usage": False}}
        legacy_stream_snapshot = dict(legacy_stream_payload)
        assert [event async for event in wrapper.stream(
            api_key="legacy async", payload=legacy_stream_payload, timeout_s=10
        )]
        assert legacy_stream_payload == legacy_stream_snapshot
        assert [item[2]["headers"]["Authorization"] for item in calls] == [
            "Bearer async synthetic", "Bearer second async synthetic",
            "Bearer async synthetic", "Bearer async synthetic",
            "Bearer legacy async", "Bearer legacy async",
        ]
        assert all(item[1] == ZAI_CHAT_COMPLETIONS_URL for item in calls)
        assert [item[2]["timeout"] for item in calls] == [4, 5, 6, 7, 9, 10]
        assert [item[2]["payload"].get("max_tokens") for item in calls] == [
            7, 8, 9, 10, 6, None
        ]
        assert calls[3][2]["payload"]["stream_options"] == {"include_usage": True}
        assert calls[5][2]["payload"]["stream_options"] == {"include_usage": False}
        response_body[0] = []
        with pytest.raises(LLMAPIClientError):
            await first.complete(model="glm-5.3-flash", timeout=11, max_tokens=1)
        with pytest.raises(LLMAPIClientError):
            await wrapper.chat(api_key="legacy async", payload={}, timeout_s=12)
    asyncio.run(exercise())
    assert len(send_calls) == 5 and len(stream_calls) == 3 and len(calls) == 8
    assert [item[0] for item in calls] == [
        "json", "sse", "json", "sse", "json", "sse", "json", "json"
    ]
    assert [item[2]["headers"]["Authorization"] for item in calls] == [
        "Bearer async synthetic", "Bearer second async synthetic",
        "Bearer async synthetic", "Bearer async synthetic",
        "Bearer legacy async", "Bearer legacy async",
        "Bearer async synthetic", "Bearer legacy async",
    ]
    assert all(item[1] == ZAI_CHAT_COMPLETIONS_URL for item in calls)
    assert [item[2]["timeout"] for item in calls] == [4, 5, 6, 7, 9, 10, 11, 12]
    assert [item[2]["payload"].get("max_tokens") for item in calls] == [
        7, 8, 9, 10, 6, None, 1, None
    ]
    assert calls[3][2]["payload"]["stream_options"] == {"include_usage": True}
    assert calls[5][2]["payload"]["stream_options"] == {"include_usage": False}


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
    second._client = ZaiSyncClient(RecordingTransport())
    second._async_client = ZaiAsyncClient()
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
