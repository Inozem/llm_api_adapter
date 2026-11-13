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
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llm_registry.reasoning import (
    CANONICAL_REASONING_LEVELS,
    resolve_reasoning_level,
)
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
    zai_adapter_runtime, monkeypatch, method_kind, case_name
):
    """Malformed native envelopes and JSON arguments retain existing public errors."""
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=ZAI_MODELS[0])
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
                "function": {"name": "lookup", "arguments": "{bad json"},
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
    zai_adapter_runtime, monkeypatch, model, method_kind
):
    adapter = ZaiAdapter(api_key="synthetic-zai-key", model=model)
    payload = {"id": "bad-stream-tool", "model": model, "choices": [{
        "index": 0, "delta": {"tool_calls": [{"index": 0, "id": "bad-json",
        "type": "function", "function": {"name": "lookup", "arguments": "{bad"}}]},
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
