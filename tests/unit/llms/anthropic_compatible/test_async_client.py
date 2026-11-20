"""Raw asynchronous Anthropic Messages connection client contracts."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import requests

from src.llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from src.llm_api_adapter.errors.config_errors import LLMConfigError
from src.llm_api_adapter.llms.transports import SSEEvent
from tests.fixtures.anthropic_compatible import messages_response


ROOT = "http://127.0.0.1:8040/local/proxy/v2/"
KEY = "synthetic-compatible-key"
MODEL = "claude-opus-4-8"
MESSAGES = [{"role": "user", "content": "hello"}]
MESSAGE_URL = ROOT.rstrip("/") + "/messages"
REAL_HTTPX_ASYNC_CLIENT = httpx.AsyncClient


def _client(**kwargs):
    # Deliberately lazy: these tests collect before the client module exists.
    from src.llm_api_adapter.llms.anthropic_compatible.async_client import (
        AnthropicCompatibleAsyncClient,
    )

    endpoint = kwargs.pop("endpoint", ROOT)
    return AnthropicCompatibleAsyncClient(api_key=KEY, endpoint=endpoint, **kwargs)


@pytest.fixture(autouse=True)
def _isolate_network_and_native_policy(monkeypatch):
    import socket

    from src.llm_api_adapter.adapters import base_adapter
    from src.llm_api_adapter.llm_registry import llm_registry
    from src.llm_api_adapter.llms import request_rules
    from src.llm_api_adapter.llms.anthropic import sync_client as native_sync
    from src.llm_api_adapter.organization_registry import OrganizationPluginDiscovery

    attempts = []

    def forbidden(name):
        def fail(*args, **kwargs):
            attempts.append(name)
            raise AssertionError("forbidden network/discovery/native-policy path: " + name)

        return fail

    monkeypatch.setattr("requests.post", forbidden("Requests post"))
    monkeypatch.setattr(requests.Session, "send", forbidden("Requests Session.send"))
    monkeypatch.setattr(socket, "getaddrinfo", forbidden("DNS"))
    monkeypatch.setattr(httpx, "Client", forbidden("HTTPX sync factory"))
    monkeypatch.setattr(httpx, "AsyncClient", forbidden("HTTPX async factory"))
    monkeypatch.setattr(OrganizationPluginDiscovery, "discover", forbidden("plugin discovery"))
    for name in ("model_spec_for", "model_api_variant", "apply_model_request_rules", "resolve_model_spec"):
        monkeypatch.setattr(request_rules, name, forbidden("request_rules." + name))
    monkeypatch.setattr(llm_registry, "resolve_model_spec", forbidden("registry.resolve_model_spec"))
    monkeypatch.setattr(base_adapter, "resolve_model_spec", forbidden("base_adapter.resolve_model_spec"))
    monkeypatch.setattr(native_sync, "apply_model_request_rules", forbidden("native Claude rules"))
    return {"attempts": attempts}


def _install_async_httpx(monkeypatch, handler):
    clients = []
    def make_client(**kwargs):
        kwargs.update(transport=httpx.MockTransport(handler), trust_env=False)
        client = REAL_HTTPX_ASYNC_CLIENT(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    return clients


def _assert_request(request, *, model=MODEL, messages=MESSAGES, stream=False, endpoint=ROOT, timeout=6.5):
    assert request.method == "POST"
    assert request.url == httpx.URL(endpoint.rstrip("/") + "/messages")
    headers = {key.lower(): value for key, value in request.headers.items()}
    assert headers["x-api-key"] == KEY
    assert headers["content-type"] == "application/json"
    assert headers["anthropic-version"] == "2023-06-01"
    if timeout is not None:
        assert request.extensions["timeout"]["read"] == timeout
    body = json.loads(request.content)
    assert body == {
        "model": model,
        "messages": messages,
        "max_tokens": 37,
        **({"stream": True} if stream else {}),
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_json_uses_async_httpx_dispatch_and_returns_raw_payload(monkeypatch, _isolate_network_and_native_policy):
    payload = messages_response(model=MODEL, custom_future_field={"kept": True})
    seen, closed = [], []

    def forbidden_sync(*args, **kwargs):
        raise AssertionError("true async path must not dispatch through sync clients")

    monkeypatch.setattr("requests.post", forbidden_sync)

    class JsonSource(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield json.dumps(payload).encode()

        async def aclose(self):
            closed.append(True)

    def handler(request):
        seen.append(request)
        return httpx.Response(200, stream=JsonSource(), request=request)

    clients = _install_async_httpx(monkeypatch, handler)
    caller_messages = json.loads(json.dumps(MESSAGES))
    original_messages = json.loads(json.dumps(MESSAGES))
    result = await _client().chat_completion(
        MODEL, timeout_s=6.5, messages=caller_messages, max_tokens=37
    )
    assert len(seen) == 1
    _assert_request(seen[0], messages=original_messages, stream=False)
    assert caller_messages == original_messages
    assert result == payload
    assert result["custom_future_field"] == {"kept": True}
    assert closed == [True]
    assert clients[0].is_closed
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_sse_preserves_raw_event_order_and_closes_source_client(monkeypatch, _isolate_network_and_native_policy):
    lines = [
        'event: content_block_delta',
        'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"raw"}}',
        "",
        'event: future_event',
        'data: {"type":"future_event","opaque":7}',
        "",
        'data: [DONE]',
        "",
    ]
    seen, closed = [], []

    class Source(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield "\n".join(lines).encode()

        async def aclose(self):
            closed.append(True)

    def handler(request):
        seen.append(request)
        return httpx.Response(200, stream=Source(), request=request)

    clients = _install_async_httpx(monkeypatch, handler)
    caller_messages = json.loads(json.dumps(MESSAGES))
    original_messages = json.loads(json.dumps(MESSAGES))
    events = [event async for event in _client().stream(
        MODEL, timeout_s=6.5, messages=caller_messages, max_tokens=37
    )]
    assert len(seen) == 1
    _assert_request(seen[0], messages=original_messages, stream=True)
    assert caller_messages == original_messages
    assert events == [
        SSEEvent("content_block_delta", {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "raw"}}),
        SSEEvent("future_event", {"type": "future_event", "opaque": 7}),
    ]
    assert closed == [True]
    assert clients[0].is_closed
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    "status,error_type,error_class",
    [
        (401, "authentication_error", LLMAPIAuthorizationError),
        (429, "rate_limit_error", LLMAPIRateLimitError),
        (400, "invalid_request_error", LLMAPIClientError),
        (503, "overloaded_error", LLMAPIServerError),
        (504, "api_error", LLMAPITimeoutError),
    ],
)
async def test_async_http_errors_map_without_retry(status, error_type, error_class, monkeypatch, _isolate_network_and_native_policy):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            status,
            json={"error": {"type": error_type, "message": "controlled"}},
            request=request,
        )

    clients = _install_async_httpx(monkeypatch, handler)
    with pytest.raises(error_class):
        await _client().chat_completion(MODEL, messages=MESSAGES, max_tokens=37)
    assert len(seen) == 1
    assert seen[0].url == httpx.URL(MESSAGE_URL)
    assert {key.lower(): value for key, value in seen[0].headers.items()}["x-api-key"] == KEY
    assert clients[0].is_closed
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_stream_cancellation_closes_source_and_client(monkeypatch, _isolate_network_and_native_policy):
    started = asyncio.Event()
    closed = []

    class HangingSource(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            await asyncio.Event().wait()
            yield b"data: {}\n\n"

        async def aclose(self):
            closed.append("source")

    def handler(request):
        return httpx.Response(200, stream=HangingSource(), request=request)

    clients = _install_async_httpx(monkeypatch, handler)
    events = _client().stream(MODEL, messages=MESSAGES, max_tokens=37)
    waiting = asyncio.create_task(events.__anext__())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert closed == ["source"]
        assert clients[0].is_closed
    finally:
        if not waiting.done():
            waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await events.aclose()
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_early_aclose_releases_source_and_client(monkeypatch, _isolate_network_and_native_policy):
    closed, seen = [], []

    class Source(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'event: future_event\ndata: {"type":"future_event","value":1}\n\n'
            await asyncio.Event().wait()

        async def aclose(self):
            closed.append("source")

    def handler(request):
        seen.append(request)
        return httpx.Response(200, stream=Source(), request=request)

    clients = _install_async_httpx(monkeypatch, handler)
    events = _client().stream(MODEL, messages=json.loads(json.dumps(MESSAGES)), max_tokens=37)
    assert await events.__anext__() == SSEEvent("future_event", {"type": "future_event", "value": 1})
    await events.aclose()
    assert len(seen) == 1
    assert closed == ["source"]
    assert clients[0].is_closed
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_stream_sse_error_and_malformed_data_close_owned_resources(monkeypatch, _isolate_network_and_native_policy):
    for raw, error_class in (
        (b'event: error\ndata: {"type":"error","error":{"type":"rate_limit_error","message":"controlled"}}\n\n', LLMAPIRateLimitError),
        (b"data: {bad json\n\n", LLMAPIClientError),
    ):
        closed, seen = [], []

        class Source(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield raw

            async def aclose(self):
                closed.append(True)

        def handler(request):
            seen.append(request)
            return httpx.Response(200, stream=Source(), request=request)

        clients = _install_async_httpx(monkeypatch, handler)
        with pytest.raises(error_class):
            async for _ in _client().stream(MODEL, messages=MESSAGES, max_tokens=37):
                pass
        assert len(seen) == 1
        assert closed == [True]
        assert clients[0].is_closed
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_http_error_closes_observable_unbuffered_source(monkeypatch, _isolate_network_and_native_policy):
    closed, seen = [], []

    class ErrorSource(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"error":{"type":"rate_limit_error","message":"controlled"}}'

        async def aclose(self):
            closed.append(True)

    def handler(request):
        seen.append(request)
        return httpx.Response(429, stream=ErrorSource(), request=request)

    clients = _install_async_httpx(monkeypatch, handler)
    with pytest.raises(LLMAPIRateLimitError):
        await _client().chat_completion(MODEL, messages=json.loads(json.dumps(MESSAGES)), max_tokens=37)
    assert len(seen) == 1
    assert closed == [True]
    assert clients[0].is_closed
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    "error,error_class",
    [
        (httpx.ReadTimeout("controlled"), LLMAPITimeoutError),
        (httpx.ConnectError("controlled"), LLMAPIClientError),
    ],
)
async def test_async_transport_failures_map_once_and_close_client(error, error_class, monkeypatch, _isolate_network_and_native_policy):
    seen = []

    def handler(request):
        seen.append(request)
        raise type(error)("controlled", request=request)

    clients = _install_async_httpx(monkeypatch, handler)
    with pytest.raises(error_class):
        await _client().chat_completion(MODEL, timeout_s=6.5, messages=json.loads(json.dumps(MESSAGES)), max_tokens=37)
    assert len(seen) == 1
    assert seen[0].url == httpx.URL(MESSAGE_URL)
    assert {key.lower(): value for key, value in seen[0].headers.items()}["x-api-key"] == KEY
    assert json.loads(seen[0].content)["model"] == MODEL
    _assert_request(seen[0], model=MODEL, messages=MESSAGES)
    assert clients[0].is_closed
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_connection_root_is_required_and_invalid_roots_fail_before_io(_isolate_network_and_native_policy):
    from src.llm_api_adapter.llms.anthropic_compatible.async_client import AnthropicCompatibleAsyncClient

    with pytest.raises((TypeError, LLMConfigError)):
        AnthropicCompatibleAsyncClient(api_key=KEY)
    for endpoint in (None, "ftp://gateway.example/v1", "http://127.0.0.1:8040/v2/messages"):
        with pytest.raises(LLMConfigError):
            AnthropicCompatibleAsyncClient(api_key=KEY, endpoint=endpoint)
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_prepared_async_execution_forwards_request_local_contexts():
    from src.llm_api_adapter.llms.anthropic_compatible.async_client import _AsyncMessagesExecutionMixin
    from src.llm_api_adapter.llms.transports import TransportRequest

    bridge = object.__new__(_AsyncMessagesExecutionMixin)
    captured = []
    callbacks = []

    async def json_request(url, *, headers, payload, timeout, http_error_handler):
        captured.append((url, headers, payload, timeout))
        callbacks.append((http_error_handler,))
        return {"url": url, "model": payload["model"]}

    class StreamSource:
        def __init__(self):
            self.closed = 0
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return SSEEvent("future", {"opaque": True})

        async def aclose(self):
            self.closed += 1

    source = StreamSource()

    def stream_request(url, *, headers, payload, timeout, http_error_handler, stream_error_handler):
        captured.append((url, headers, payload, timeout))
        callbacks.append((http_error_handler, stream_error_handler))
        return source

    bridge._get_async_request = json_request
    bridge._get_async_stream_request = stream_request
    first = TransportRequest("https://one.example/prefix/messages", {"x-api-key": "one"}, {"model": "m1"}, 3)
    second = TransportRequest("http://two.example/root/messages", {"x-workspace": "two"}, {"model": "m2"}, 5)

    assert await bridge._execute_prepared_request(first) == {"url": first.url, "model": "m1"}
    events = bridge._execute_prepared_stream(second)
    assert await events.__anext__() == SSEEvent("future", {"opaque": True})
    await events.aclose()

    assert captured == [
        (first.url, {"x-api-key": "one"}, {"model": "m1"}, 3),
        (second.url, {"x-workspace": "two"}, {"model": "m2"}, 5),
    ]
    assert source.closed == 1
    assert callbacks[0][0].__self__ is bridge
    assert callbacks[1][0].__self__ is bridge
    assert callbacks[1][1].__self__ is bridge
    assert not hasattr(bridge, "api_key")
    assert not hasattr(bridge, "endpoint")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_configuration_preserves_whitespace_and_overlapping_request_contexts():
    from src.llm_api_adapter.llms.anthropic_compatible.async_client import AnthropicCompatibleAsyncClient

    client = AnthropicCompatibleAsyncClient(api_key=" ", endpoint=ROOT)
    headers_calls, captured = [], []
    ready = asyncio.Event()

    def headers():
        headers_calls.append(len(headers_calls) + 1)
        return {"x-call": str(headers_calls[-1])}

    async def async_json(url, *, headers, payload, timeout, http_error_handler):
        captured.append((headers, payload))
        if len(captured) == 2:
            ready.set()
        await ready.wait()
        return {"model": payload["model"]}

    client._headers = headers
    client._get_async_request = async_json
    one, two = await asyncio.gather(
        client.chat_completion("model one", messages=[{"role": "user", "content": "one"}]),
        client.chat_completion("model two", messages=[{"role": "user", "content": "two"}]),
    )

    assert {one["model"], two["model"]} == {"model one", "model two"}
    assert headers_calls == [1, 2]
    assert sorted((header["x-call"], body["model"]) for header, body in captured) == [
        ("1", "model one"), ("2", "model two")
    ]
    assert client.api_key == " "


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_diagnostics_redact_escaped_key_and_encoded_endpoint_key():
    from urllib.parse import quote

    from src.llm_api_adapter.llms.anthropic_compatible.async_client import AnthropicCompatibleAsyncClient

    key = 'quoted"\\line\nkey'
    encoded = quote(key, safe="")
    endpoint = f"https://gateway.example/prefix/{encoded}"
    client = AnthropicCompatibleAsyncClient(api_key=key, endpoint=endpoint)
    representation = repr(client)

    assert client.api_key == key
    assert client.endpoint == endpoint
    assert key not in representation
    assert encoded not in representation


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_execution_sanitizes_message_and_detail_without_mutating_original():
    from urllib.parse import quote

    from src.llm_api_adapter.errors.llm_api_error import LLMAPIClientError

    client = _client()
    encoded_key = quote(KEY, safe="")
    original = LLMAPIClientError(
        message=f"failed {KEY} {encoded_key} https://user:pass@gateway.example/?token=x",
        detail=f"detail {KEY} {encoded_key} https://user:pass@gateway.example/#fragment",
    )

    async def fail_json(url, **kwargs):
        raise original

    client._get_async_request = fail_json
    with pytest.raises(LLMAPIClientError) as raised:
        await client.chat_completion(MODEL, messages=MESSAGES)

    assert raised.value is not original
    assert original.message.startswith("failed " + KEY)
    assert original.detail.startswith("detail " + KEY)
    assert raised.value.__cause__ is None
    for visible in (str(raised.value), raised.value.message, raised.value.detail or "", repr(raised.value)):
        assert KEY not in visible
        assert encoded_key not in visible
        assert "user:pass" not in visible
        assert "token=x" not in visible
        assert "#fragment" not in visible


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_stream_source_creation_and_normal_cleanup_errors_use_generic_policy():
    from src.llm_api_adapter.errors.llm_api_error import LLMAPIClientError

    client = _client()
    eager = LLMAPIClientError(message=f"source {KEY}")

    def eager_source(url, **kwargs):
        raise eager

    client._get_async_stream_request = eager_source
    with pytest.raises(LLMAPIClientError) as raised_eager:
        async for _ in client.stream(MODEL, messages=MESSAGES):
            pass
    assert raised_eager.value is not eager
    assert KEY not in str(raised_eager.value)

    cleanup = LLMAPIClientError(message=f"cleanup {KEY}")

    class Source:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def aclose(self):
            raise cleanup

    client._get_async_stream_request = lambda url, **kwargs: Source()
    with pytest.raises(LLMAPIClientError) as raised_cleanup:
        async for _ in client.stream(MODEL, messages=MESSAGES):
            pass
    assert raised_cleanup.value is not cleanup
    assert KEY not in str(raised_cleanup.value)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_keyless_async_bridge_preserves_native_error_identity_and_primary_close_failure():
    from src.llm_api_adapter.errors.llm_api_error import LLMAPIClientError
    from src.llm_api_adapter.llms.anthropic_compatible.async_client import _AsyncMessagesExecutionMixin
    from src.llm_api_adapter.llms.transports import TransportRequest

    bridge = object.__new__(_AsyncMessagesExecutionMixin)
    native_error = LLMAPIClientError(detail="native")

    async def fail_json(url, **kwargs):
        raise native_error

    bridge._get_async_request = fail_json
    with pytest.raises(LLMAPIClientError) as raised:
        await bridge._execute_prepared_request(TransportRequest("https://one.example/messages"))
    assert raised.value is native_error

    decode_error = json.JSONDecodeError("bad JSON", "secret response", 0)

    async def fail_decode(url, **kwargs):
        raise decode_error

    bridge._get_async_request = fail_decode
    with pytest.raises(json.JSONDecodeError) as decode_raised:
        await bridge._execute_prepared_request(TransportRequest("https://one.example/messages"))
    assert decode_raised.value is decode_error
    assert not hasattr(bridge, "api_key")
    assert not hasattr(bridge, "endpoint")

    primary = RuntimeError("iteration")

    class BrokenSource:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise primary

        async def aclose(self):
            raise ValueError("secondary close")

    bridge._get_async_stream_request = lambda *args, **kwargs: BrokenSource()
    events = bridge._execute_prepared_stream(TransportRequest("https://two.example/messages"))
    with pytest.raises(RuntimeError) as iter_error:
        async for _ in events:
            pass
    assert iter_error.value is primary


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_success_status_malformed_json_maps_to_client_error_and_closes(monkeypatch, _isolate_network_and_native_policy):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, text="not-json", request=request)

    clients = _install_async_httpx(monkeypatch, handler)
    with pytest.raises(LLMAPIClientError, match="Malformed JSON response"):
        await _client().chat_completion(MODEL, messages=MESSAGES, max_tokens=37)

    assert len(seen) == 1
    assert seen[0].url == httpx.URL(MESSAGE_URL)
    assert clients[0].is_closed


