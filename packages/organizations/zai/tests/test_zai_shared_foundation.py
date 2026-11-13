"""Credential-free legacy and shared-engine contracts for Z.ai clients."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
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
)
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent, SyncTransport, TransportRequest
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
