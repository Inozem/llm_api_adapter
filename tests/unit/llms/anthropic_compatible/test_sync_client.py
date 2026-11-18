"""Raw synchronous Anthropic Messages connection client contracts."""

from __future__ import annotations

from unittest.mock import Mock
import json

import httpx
import pytest
import requests
import requests_mock

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


ROOT = "https://gateway.example/prefix/v9/"
KEY = "synthetic-compatible-key"
MODEL = "unregistered-messages-model"
MESSAGES = [{"role": "user", "content": "hello"}]
MESSAGE_URL = ROOT.rstrip("/") + "/messages"
REAL_HTTPX_CLIENT = httpx.Client
_UNSET = object()


def _client(transport="requests", **kwargs):
    # Deliberately lazy: these tests collect before the client module exists.
    from src.llm_api_adapter.llms.anthropic_compatible.sync_client import (
        AnthropicCompatibleSyncClient,
    )

    endpoint = kwargs.pop("endpoint", ROOT)
    return AnthropicCompatibleSyncClient(
        api_key=KEY, endpoint=endpoint, transport=transport, **kwargs
    )


@pytest.fixture(autouse=True)
def _isolate_network_and_native_policy(monkeypatch):
    """Block non-mocked I/O and policy before the lazy client import."""
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

    # requests_mock is the Requests send barrier: unmatched URLs cannot use the network.
    mocker = requests_mock.Mocker(real_http=False)
    mocker.start()
    closed_responses = []
    real_response_close = requests.Response.close

    def record_close(response):
        closed_responses.append(response)
        return real_response_close(response)

    monkeypatch.setattr(socket, "getaddrinfo", forbidden("DNS"))
    monkeypatch.setattr(httpx, "Client", forbidden("HTTPX sync factory"))
    monkeypatch.setattr(httpx, "AsyncClient", forbidden("HTTPX async factory"))
    monkeypatch.setattr(requests.Response, "close", record_close)
    monkeypatch.setattr(OrganizationPluginDiscovery, "discover", forbidden("plugin discovery"))
    for name in ("model_spec_for", "model_api_variant", "apply_model_request_rules", "resolve_model_spec"):
        monkeypatch.setattr(request_rules, name, forbidden("request_rules." + name))
    monkeypatch.setattr(llm_registry, "resolve_model_spec", forbidden("registry.resolve_model_spec"))
    monkeypatch.setattr(base_adapter, "resolve_model_spec", forbidden("base_adapter.resolve_model_spec"))
    monkeypatch.setattr(native_sync, "apply_model_request_rules", forbidden("native Claude rules"))
    try:
        yield {"requests": mocker, "attempts": attempts, "closed_responses": closed_responses}
    finally:
        mocker.stop()


def _install_sync_httpx(monkeypatch, handler):
    clients = []

    def make_client(**kwargs):
        kwargs.update(transport=httpx.MockTransport(handler), trust_env=False)
        client = REAL_HTTPX_CLIENT(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", make_client)
    return clients


class _RequestsResponse:
    def __init__(self, *, lines=()):
        self.lines = list(lines)
        self.close = Mock()

    def iter_lines(self, decode_unicode=True):
        return iter(self.lines)

    def raise_for_status(self):
        return None


def _capture_requests(monkeypatch, *, lines=()):
    seen = []
    response = _RequestsResponse(lines=lines)

    def post(url, **kwargs):
        prepared = requests.Request(
            "POST", url, headers=kwargs.get("headers"), json=kwargs.get("json")
        ).prepare()
        seen.append((prepared, kwargs))
        return response

    monkeypatch.setattr("src.llm_api_adapter.llms.requests_transport.requests.post", post)
    return seen, response


def _assert_wire(request, *, model=MODEL, messages=MESSAGES, stream=False, expected_timeout=_UNSET):
    assert request.method == "POST"
    assert str(request.url) == MESSAGE_URL
    headers = {key.lower(): value for key, value in request.headers.items()}
    assert headers["x-api-key"] == KEY
    assert headers["content-type"] == "application/json"
    assert headers["anthropic-version"] == "2023-06-01"
    raw_body = request.content if isinstance(request, httpx.Request) else request.body
    body = json.loads(raw_body)
    expected = {"model": model, "messages": messages, "max_tokens": 37}
    if stream:
        expected["stream"] = True
    assert body == expected
    if isinstance(request, httpx.Request):
        assert request.extensions["timeout"]["read"] == 6.5
    else:
        assert request.stream is stream
        if expected_timeout is not _UNSET:
            assert request.timeout == expected_timeout


def _assert_requests_history(mock, *, model=MODEL, messages=MESSAGES, stream=False, expected_timeout=None):
    assert mock.call_count == 1
    _assert_wire(
        mock.request_history[0], model=model, messages=messages, stream=stream,
        expected_timeout=expected_timeout,
    )


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_json_uses_selected_transport_and_preserves_raw_messages_request(transport, monkeypatch, _isolate_network_and_native_policy):
    body = messages_response(model=MODEL, custom_future_field={"kept": True})
    original_messages = json.loads(json.dumps(MESSAGES))
    caller_messages = json.loads(json.dumps(MESSAGES))
    if transport == "requests":
        requests_http = _isolate_network_and_native_policy["requests"]
        requests_http.post(MESSAGE_URL, json=body)
        result = _client(transport).chat_completion(
            MODEL, timeout_s=6.5, messages=caller_messages, max_tokens=37
        )
        assert requests_http.call_count == 1
        _assert_wire(requests_http.request_history[0], model=MODEL, messages=original_messages, expected_timeout=6.5)
        assert len(_isolate_network_and_native_policy["closed_responses"]) == 1
    else:
        seen, sources = [], []

        class JsonSource(httpx.SyncByteStream):
            def __iter__(self):
                yield json.dumps(body).encode()

            def close(self):
                sources.append("closed")

        def handler(request):
            seen.append(request)
            return httpx.Response(200, stream=JsonSource(), request=request)

        clients = _install_sync_httpx(monkeypatch, handler)
        result = _client(transport).chat_completion(
            MODEL, timeout_s=6.5, messages=caller_messages, max_tokens=37
        )
        assert len(seen) == 1
        _assert_wire(seen[0], model=MODEL, messages=original_messages)
        assert sources == ["closed"]
        assert clients[0].is_closed
    assert caller_messages == original_messages
    assert _isolate_network_and_native_policy["attempts"] == []
    assert result == body
    assert result["custom_future_field"] == {"kept": True}


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_sse_returns_raw_ordered_events_and_closes_owned_response(transport, monkeypatch, _isolate_network_and_native_policy):
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
    caller_messages = json.loads(json.dumps(MESSAGES))
    original_messages = json.loads(json.dumps(MESSAGES))
    if transport == "requests":
        requests_http = _isolate_network_and_native_policy["requests"]
        requests_http.post(MESSAGE_URL, content="\n".join(lines).encode())
        events = list(_client(transport).stream(
            MODEL, timeout_s=6.5, messages=caller_messages, max_tokens=37
        ))
        assert requests_http.call_count == 1
        _assert_wire(requests_http.request_history[0], model=MODEL, messages=original_messages, stream=True, expected_timeout=6.5)
        assert len(_isolate_network_and_native_policy["closed_responses"]) == 1
    else:
        seen, clients, streams = [], [], []

        class Source(httpx.SyncByteStream):
            def __iter__(self):
                yield "\n".join(lines).encode()

            def close(self):
                streams.append("closed")

        def handler(request):
            seen.append(request)
            return httpx.Response(200, stream=Source(), request=request)

        clients = _install_sync_httpx(monkeypatch, handler)
        events = list(_client(transport).stream(
            MODEL, timeout_s=6.5, messages=caller_messages, max_tokens=37
        ))
        assert len(seen) == 1
        _assert_wire(seen[0], model=MODEL, messages=original_messages, stream=True)
        assert streams == ["closed"]
        assert clients[0].is_closed
    assert caller_messages == original_messages
    assert events == [
        SSEEvent("content_block_delta", {"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "raw"}}),
        SSEEvent("future_event", {"type": "future_event", "opaque": 7}),
    ]
    assert _isolate_network_and_native_policy["attempts"] == []


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
def test_http_failures_map_to_public_categories_without_retry(status, error_type, error_class, _isolate_network_and_native_policy):
    requests_http = _isolate_network_and_native_policy["requests"]
    requests_http.post(
        MESSAGE_URL, status_code=status,
        json={"error": {"type": error_type, "message": "controlled"}},
    )
    with pytest.raises(error_class):
        _client().chat_completion(MODEL, messages=MESSAGES, max_tokens=37)
    _assert_requests_history(requests_http)
    assert len(_isolate_network_and_native_policy["closed_responses"]) == 1
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.unit
def test_known_messages_sse_errors_and_malformed_json_keep_public_errors(_isolate_network_and_native_policy):
    requests_http = _isolate_network_and_native_policy["requests"]
    for error_type, error_class in (
        ("authentication_error", LLMAPIAuthorizationError),
        ("rate_limit_error", LLMAPIRateLimitError),
        ("api_error", LLMAPIServerError),
        ("overloaded_error", LLMAPIServerError),
    ):
        lines = ["event: error", "data: " + json.dumps({"type": "error", "error": {"type": error_type, "message": "controlled"}}), ""]
        requests_http.post(MESSAGE_URL, content="\n".join(lines).encode())
        with pytest.raises(error_class):
            list(_client().stream(MODEL, messages=MESSAGES, max_tokens=37))
        _assert_requests_history(requests_http, stream=True)
        assert len(_isolate_network_and_native_policy["closed_responses"]) == 1
        _isolate_network_and_native_policy["closed_responses"].clear()
        requests_http.reset()

    requests_http.post(MESSAGE_URL, content=b"data: {bad json\n\n")
    with pytest.raises(LLMAPIClientError):
        list(_client().stream(MODEL, messages=MESSAGES, max_tokens=37))
    _assert_requests_history(requests_http, stream=True)
    assert len(_isolate_network_and_native_policy["closed_responses"]) == 1
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "endpoint",
    ["", "ftp://gateway.example", "https://user:pass@gateway.example/v1", "https://gateway.example/v1?x=1", "https://gateway.example/v1#fragment", "https://gateway.example/v1/messages"],
)
def test_invalid_explicit_roots_fail_before_transport(endpoint, monkeypatch, _isolate_network_and_native_policy):
    from src.llm_api_adapter.llms.requests_transport import RequestsSyncTransport

    send = Mock(side_effect=AssertionError("invalid root reached provider I/O"))
    monkeypatch.setattr(RequestsSyncTransport, "post_json", send)
    with pytest.raises(LLMConfigError):
        _client(endpoint=endpoint)
    send.assert_not_called()
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.unit
def test_root_is_required(_isolate_network_and_native_policy):
    from src.llm_api_adapter.llms.anthropic_compatible.sync_client import AnthropicCompatibleSyncClient

    with pytest.raises((TypeError, LLMConfigError)):
        AnthropicCompatibleSyncClient(api_key=KEY)
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_early_consumer_close_releases_owned_source_and_client(transport, monkeypatch, _isolate_network_and_native_policy):
    lines = ['data: {"type":"future_event","value":1}', "", 'data: {"type":"future_event","value":2}', ""]
    if transport == "requests":
        seen, response = _capture_requests(monkeypatch, lines=lines)
        stream = _client(transport).stream(MODEL, messages=json.loads(json.dumps(MESSAGES)), max_tokens=37)
        next(stream)
        stream.close()
        assert len(seen) == 1
        assert response.close.call_count == 1
    else:
        sources, seen = [], []

        class Source(httpx.SyncByteStream):
            def __iter__(self):
                yield "\n".join(lines).encode()

            def close(self):
                sources.append("closed")

        def handler(request):
            seen.append(request)
            return httpx.Response(200, stream=Source(), request=request)

        clients = _install_sync_httpx(monkeypatch, handler)
        stream = _client(transport).stream(MODEL, messages=json.loads(json.dumps(MESSAGES)), max_tokens=37)
        next(stream)
        stream.close()
        assert len(seen) == 1
        assert sources == ["closed"]
        assert clients[0].is_closed
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.unit
def test_httpx_malformed_stream_closes_source_and_client(monkeypatch, _isolate_network_and_native_policy):
    sources, seen = [], []

    class Source(httpx.SyncByteStream):
        def __iter__(self):
            yield b"data: {broken json\n\n"

        def close(self):
            sources.append("closed")

    def handler(request):
        seen.append(request)
        return httpx.Response(200, stream=Source(), request=request)

    clients = _install_sync_httpx(monkeypatch, handler)
    with pytest.raises(LLMAPIClientError):
        list(_client("httpx").stream(MODEL, messages=json.loads(json.dumps(MESSAGES)), max_tokens=37))
    assert len(seen) == 1
    assert sources == ["closed"]
    assert clients[0].is_closed
    assert _isolate_network_and_native_policy["attempts"] == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "transport,error,error_class",
    [
        ("requests", requests.Timeout("controlled"), LLMAPITimeoutError),
        ("requests", requests.ConnectionError("controlled"), LLMAPIClientError),
        ("httpx", httpx.ReadTimeout("controlled"), LLMAPITimeoutError),
        ("httpx", httpx.ConnectError("controlled"), LLMAPIClientError),
    ],
)
def test_selected_transport_failures_map_once_without_fallback(transport, error, error_class, monkeypatch, _isolate_network_and_native_policy):
    caller_messages = json.loads(json.dumps(MESSAGES))
    original_messages = json.loads(json.dumps(MESSAGES))
    if transport == "requests":
        requests_http = _isolate_network_and_native_policy["requests"]
        requests_http.post(MESSAGE_URL, exc=error)
        with pytest.raises(error_class):
            _client(transport).chat_completion(MODEL, timeout_s=6.5, messages=caller_messages, max_tokens=37)
        _assert_requests_history(
            requests_http, model=MODEL, messages=original_messages,
            expected_timeout=6.5,
        )
    else:
        seen = []

        def handler(request):
            seen.append(request)
            raise type(error)("controlled", request=request)

        clients = _install_sync_httpx(monkeypatch, handler)
        with pytest.raises(error_class):
            _client(transport).chat_completion(MODEL, timeout_s=6.5, messages=caller_messages, max_tokens=37)
        assert len(seen) == 1
        assert clients[0].is_closed
        request = seen[0]
        _assert_wire(request, model=MODEL, messages=original_messages)
    assert caller_messages == original_messages
    assert _isolate_network_and_native_policy["attempts"] == []


