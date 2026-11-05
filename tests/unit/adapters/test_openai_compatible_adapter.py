"""Basic synchronous request contract for compatible OpenAI endpoints."""

from copy import deepcopy
import json
import socket
import traceback
import asyncio
from pathlib import Path
import subprocess
import sys
import warnings
from unittest.mock import AsyncMock

import httpx
import pytest
import requests
import requests_mock

from src.llm_api_adapter.adapters.openai_compatible_adapter import (
    OpenAICompatibleAdapter,
    _validate_api_root,
)
from src.llm_api_adapter.errors.config_errors import LLMConfigError
from src.llm_api_adapter.errors.llm_api_error import (
    LLMAPIClientError,
    LLMAPIAuthorizationError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
    InvalidToolArgumentsError,
    InvalidToolSchemaError,
    JSONSchemaError,
    ToolChoiceError,
)
from src.llm_api_adapter.adapters.openai.adapter import OpenAIAdapter
from src.llm_api_adapter.llm_registry.llm_registry import Pricing
from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.llms.transports import SSEFrameDecoder
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.models.responses.chat_response import ChatResponse
from src.llm_api_adapter.models.tools import ToolSpec
from src.llm_api_adapter.organization_registry import OrganizationPluginDiscovery
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from tests.fixtures.openai_compatible import (
    FIXTURE_ID,
    FIXTURE_MODEL,
    event_lines,
    premature_eof_sse_lines,
    text_response,
    text_sse_lines,
    tool_call_response,
    tool_call_sse_lines,
    usage_payload,
    zero_usage_payload,
)


_ENDPOINT_CASES = [
    (
        "http://localhost:11434/v1",
        "http://localhost:11434/v1/chat/completions",
        "vendor/unregistered-model",
    ),
    (
        "http://localhost:11434/v1/",
        "http://localhost:11434/v1/chat/completions",
        "local-model:Q4_K_M",
    ),
    (
        "https://example.invalid/gateway/tenant/v1/",
        "https://example.invalid/gateway/tenant/v1/chat/completions",
        "vendor/unregistered-model",
    ),
    (
        "https://example.invalid/api",
        "https://example.invalid/api/chat/completions",
        "local-model:Q4_K_M",
    ),
    (
        "http://127.0.0.1:8080",
        "http://127.0.0.1:8080/chat/completions",
        "vendor/unregistered-model",
    ),
]


def _new_adapter(*, api_key: str, model: str, base_url: str):
    """Import the implementation during execution so this test collects pre-T010."""
    from src.llm_api_adapter.adapters.openai_compatible_adapter import (
        OpenAICompatibleAdapter,
    )

    return OpenAICompatibleAdapter(
        api_key=api_key,
        model=model,
        base_url=base_url,
    )


_INVALID_TOOL_REQUEST_CASES = [
    pytest.param({"tools": [object()]}, InvalidToolSchemaError, id="not-tool-spec"),
    pytest.param(
        {"tools": [ToolSpec(name="", json_schema={"type": "object"})]},
        InvalidToolSchemaError,
        id="empty-name",
    ),
    pytest.param(
        {"tools": [ToolSpec(name="lookup", json_schema=[])]},
        InvalidToolSchemaError,
        id="nondict-schema",
    ),
    pytest.param(
        {"tools": [
            ToolSpec(name="lookup", json_schema={"type": "object"}),
            ToolSpec(name="lookup", json_schema={"type": "object"}),
        ]},
        InvalidToolSchemaError,
        id="duplicate-name",
    ),
    pytest.param(
        {"tools": [ToolSpec(name="lookup", json_schema={"type": "object"})], "tool_choice": 7},
        ToolChoiceError,
        id="choice-type",
    ),
    pytest.param({"tool_choice": "any"}, ToolChoiceError, id="any-without-tools"),
    pytest.param(
        {"tools": [ToolSpec(name="lookup", json_schema={"type": "object"})], "tool_choice": "missing"},
        ToolChoiceError,
        id="unknown-choice",
    ),
]


def _invoke_compatible_sync(adapter, mode, kwargs):
    arguments = {"messages": [UserMessage(content="Use the declared tool")], **kwargs}
    if mode == "chat":
        return adapter.chat(**arguments)
    return list(adapter.stream_chat(**arguments))


async def _invoke_compatible_async(adapter, mode, kwargs):
    arguments = {"messages": [UserMessage(content="Use the declared tool")], **kwargs}
    if mode == "achat":
        return await adapter.achat(**arguments)
    return [text async for text in adapter.astream_chat(**arguments)]


@pytest.mark.unit
@pytest.mark.parametrize("base_url, expected_url, model", _ENDPOINT_CASES)
def test_sync_chat_posts_exact_compatible_request_and_normalizes_text(
    base_url, expected_url, model, monkeypatch
):
    forbidden_attempts = []

    def guard_boundary(name):
        def forbid(*args, **kwargs):
            forbidden_attempts.append(name)
            raise AssertionError(
                f"compatible request crossed forbidden boundary: {name}"
            )

        return forbid

    monkeypatch.setattr(socket, "getaddrinfo", guard_boundary("socket.getaddrinfo"))
    monkeypatch.setattr(httpx.Client, "send", guard_boundary("httpx.Client.send"))
    monkeypatch.setattr(
        httpx.AsyncClient,
        "send",
        guard_boundary("httpx.AsyncClient.send"),
    )
    monkeypatch.setattr(
        OrganizationPluginDiscovery,
        "discover",
        guard_boundary("OrganizationPluginDiscovery.discover"),
    )

    with requests_mock.Mocker(real_http=False) as http:
        http.post(expected_url, json=text_response())
        adapter = _new_adapter(
            api_key="synthetic-test-key-not-a-credential",
            model=model,
            base_url=base_url,
        )
        response = adapter.chat(
            messages=[UserMessage(content="Say hello.")],
            max_tokens=17,
        )

        assert http.call_count == 1
        request = http.request_history[0]
        assert request.method == "POST"
        assert request.url == expected_url
        assert request.headers["Authorization"] == (
            "Bearer synthetic-test-key-not-a-credential"
        )
        assert request.headers["Content-Type"] == "application/json"
        assert request.json() == {
            "model": model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_completion_tokens": 17,
        }
        assert forbidden_attempts == []

    assert isinstance(response, ChatResponse)
    assert response.content == "Fixture reply"
    assert response.model == FIXTURE_MODEL
    assert response.response_id == FIXTURE_ID
    assert response.finish_reason == "stop"


@pytest.mark.unit
@pytest.mark.parametrize(
    "root, expected",
    [
        ("http://localhost:11434/v1", "http://localhost:11434/v1"),
        ("https://[::1]:8443/proxy/tenant/v1///", "https://[::1]:8443/proxy/tenant/v1"),
        ("http://127.0.0.1:0/", "http://127.0.0.1:0"),
    ],
)
def test_api_root_validation_preserves_prefix_and_removes_trailing_slashes(root, expected):
    assert _validate_api_root(root) == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "root",
    [
        None,
        "",
        "   ",
        "ftp://localhost/v1",
        "//localhost/v1",
        "http:///v1",
        "http://user:secret@localhost/v1",
        "http://localhost/v1?",
        "http://localhost/v1#",
        "http://localhost:/v1",
        "http://localhost:99999/v1",
        "http://[not-ipv6]/v1",
        "http://[::1/v1",
        "http://localhost\\@evil/v1",
        "http://localhost/v1/chat/completions",
        "http://localhost/v1/chat/completions/",
        "http://local host/v1",
    ],
)
def test_api_root_validation_rejects_invalid_roots_without_echoing_input(root):
    with pytest.raises(LLMConfigError) as caught:
        _validate_api_root(root)
    if isinstance(root, str) and root:
        assert root not in str(caught.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    "root",
    (
        "https://user:credential-marker@example.invalid/v1",
        "https://example.invalid/v1?credential-marker",
        "https://example.invalid/v1#credential-marker",
    ),
)
def test_compatible_configuration_errors_do_not_echo_credentials_or_roots(root):
    key = "configuration-key-marker"
    with pytest.raises(LLMConfigError) as caught:
        UniversalLLMAPIAdapter(
            organization="openai_compatible",
            model="arbitrary/model",
            api_key=key,
            base_url=root,
        )

    for value in (str(caught.value), repr(caught.value)):
        assert key not in value
        assert "credential-marker" not in value
        assert "example.invalid" not in value


@pytest.mark.unit
def test_constructor_preserves_model_and_disables_official_profile():
    adapter = OpenAICompatibleAdapter(
        api_key="synthetic-key",
        model="gpt-4o",
        base_url="https://example.invalid/proxy/v1/",
    )

    assert adapter.model == "gpt-4o"
    assert adapter.base_url == "https://example.invalid/proxy/v1"
    assert adapter.company == adapter.service_provider == "openai_compatible"
    assert adapter.model_spec is None
    assert adapter.pricing is None
    assert adapter.is_reasoning is False
    assert adapter.is_adaptive_thinking is False
    assert "synthetic-key" not in repr(adapter)
    assert "example.invalid" not in repr(adapter)


@pytest.mark.unit
def test_compatible_diagnostic_repr_redacts_instance_secrets_without_mutating_values():
    key = "repr-secret/+marker-123"
    encoded_key = "repr-secret%2f%2bmarker-123"
    model = f"vendor/{key}"
    root = f"https://example.invalid/proxy/{encoded_key}/v1"
    facade = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model=model,
        api_key=key,
        base_url=root,
    )
    sync_client = facade.adapter._create_sync_client()
    async_client = facade.adapter._create_async_client()
    native = OpenAISyncClient(api_key="native-secret-marker-456")

    for value in (repr(facade), repr(facade.adapter), repr(sync_client), repr(async_client)):
        assert key not in value
        assert encoded_key not in value
    assert facade.model == model
    assert facade.base_url == root
    assert sync_client.endpoint == root
    assert async_client.endpoint == root
    assert "native-secret-marker-456" not in repr(native)
    assert native._diagnostic_formatter is None
    assert native == OpenAISyncClient(api_key="native-secret-marker-456")
    assert repr(native) == (
        "OpenAISyncClient(api_key='native-s...-456', "
        "endpoint='https://api.openai.com/v1')"
    )


@pytest.mark.unit
def test_native_facade_repr_stays_unchanged():
    facade = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-4o-mini",
        api_key="native-secret-marker",
    )
    assert repr(facade) == (
        "UniversalLLMAPIAdapter(organization='openai', "
        "service_provider='openai', model='gpt-4o-mini', "
        "transport='requests', api_key='native-s...rker')"
    )


@pytest.mark.unit
def test_facade_repr_uses_explicit_mode_gate_and_handles_partial_instances():
    expected = (
        "UniversalLLMAPIAdapter(organization='openai', "
        "service_provider='openai', model='gpt-4o-mini', "
        "transport='requests', api_key='native-s...rker')"
    )
    native = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-4o-mini",
        api_key="native-secret-marker",
    )
    native.adapter._format_compatible_diagnostic = lambda _text: "changed"
    assert repr(native) == expected

    partial = object.__new__(UniversalLLMAPIAdapter)
    partial.organization = "openai"
    partial.model = "gpt-4o-mini"
    partial.api_key = "native-secret-marker"
    partial.transport = "requests"
    partial.service_provider = "openai"
    assert repr(partial) == expected

    compatible = object.__new__(UniversalLLMAPIAdapter)
    compatible.organization = "openai_compatible"
    compatible.model = "vendor/partial-secret-marker"
    compatible.api_key = "partial-secret-marker"
    compatible.transport = "requests"
    compatible.service_provider = "openai_compatible"
    result = repr(compatible)
    assert "partial-secret-marker" not in result


_DIAGNOSTIC_KEY = "diagnostic-secret-marker-987"
_DIAGNOSTIC_URL = "https://url-user:url-password'quote@echo.invalid/v1"
_DIAGNOSTIC_QUERY_URL = 'https://echo.invalid/v1?token="query-secret-marker"'
_DIAGNOSTIC_FRAGMENT_URL = "https://echo.invalid/v1?token=query-marker#frag-marker"
_DIAGNOSTIC_TEXT = (
    f"echo {_DIAGNOSTIC_KEY} and {_DIAGNOSTIC_URL} "
    f"{_DIAGNOSTIC_QUERY_URL} {_DIAGNOSTIC_FRAGMENT_URL}"
)


def _assert_diagnostics_redacted(error, caplog):
    rendered = "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )
    messages = "\n".join(record.getMessage() for record in caplog.records)
    for marker in (
        _DIAGNOSTIC_KEY,
        "url-user",
        "url-password",
        "quote",
        "query-secret-marker",
        "query-marker",
        "frag-marker",
    ):
        assert marker not in str(error)
        assert marker not in repr(error)
        assert marker not in rendered
        assert marker not in messages
    assert any(
        reason in str(error).lower()
        for reason in ("authentication", "timed out", "client error", "server error")
    )
    assert "redacted URL" in str(error)


@pytest.mark.unit
def test_credential_url_redaction_precedes_valid_scheme_matching_api_key():
    from src.llm_api_adapter.adapters.openai_compatible_adapter import (
        _redact_compatible_diagnostic,
    )

    diagnostic = "https://url-user:url-pass@echo.invalid/v1"
    redacted = _redact_compatible_diagnostic(diagnostic, "https")

    assert "url-pass" not in redacted
    assert redacted == "[redacted URL]"


@pytest.mark.unit
@pytest.mark.parametrize("mode", ("chat", "stream_chat"))
def test_json_escaped_credential_is_redacted_at_real_error_boundary(mode, caplog):
    key = "wire'\\slash/key"
    url = f"{_COLLISION_ROOT}/chat/completions"
    payload = {
        "error": {
            "message": f"invalid credential {key}",
            "type": "invalid_api_key",
        }
    }
    with requests_mock.Mocker(real_http=False) as http:
        if mode == "chat":
            http.post(url, status_code=401, json=payload)
        else:
            event = f"event: error\ndata: {json.dumps(payload)}\n\n"
            http.post(
                url,
                status_code=200,
                text=event,
                headers={"Content-Type": "text/event-stream"},
            )
        adapter = _new_adapter(
            api_key=key,
            model="arbitrary/model",
            base_url=_COLLISION_ROOT,
        )
        with pytest.raises(LLMAPIAuthorizationError) as caught:
            if mode == "stream_chat":
                list(adapter.stream_chat([UserMessage(content="hello")]))
            else:
                adapter.chat([UserMessage(content="hello")])

    rendered_traceback = "".join(
        traceback.format_exception(
            type(caught.value), caught.value, caught.value.__traceback__
        )
    )
    log_messages = "\n".join(record.getMessage() for record in caplog.records)
    assert key not in str(caught.value)
    assert key not in repr(caught.value)
    assert key not in rendered_traceback
    assert key not in log_messages
    assert http.call_count == 1
    request = http.request_history[0]
    assert request.url == url
    assert request.headers["Authorization"] == f"Bearer {key}"


@pytest.mark.unit
def test_redactor_matches_mixed_case_percent_hex_without_casefolding_key():
    from src.llm_api_adapter.adapters.openai_compatible_adapter import (
        _redact_compatible_diagnostic,
    )

    key = "a/+marker"
    mixed_percent_case = "endpoint https://echo.invalid/a%2f%2Bmarker/v1"
    case_changed_letter = "endpoint https://echo.invalid/A%2f%2Bmarker/v1"

    assert "a%2f%2Bmarker" not in _redact_compatible_diagnostic(
        mixed_percent_case, key
    )
    assert "A%2f%2Bmarker" in _redact_compatible_diagnostic(
        case_changed_letter, key
    )
    different_encoded_bytes = "endpoint https://echo.invalid/a%41%42marker/v1"
    assert different_encoded_bytes in _redact_compatible_diagnostic(
        different_encoded_bytes, key
    )


@pytest.mark.unit
@pytest.mark.parametrize("mode", ("chat", "stream_chat"))
def test_requests_http_error_diagnostics_are_redacted_before_logs_and_traceback(
    mode, caplog
):
    url = f"{_COLLISION_ROOT}/chat/completions"
    body = {"error": {"message": _DIAGNOSTIC_TEXT, "type": "invalid_api_key"}}
    with requests_mock.Mocker(real_http=False) as http:
        http.post(url, status_code=401, json=body)
        facade = UniversalLLMAPIAdapter(
            organization="openai_compatible",
            model="arbitrary/model",
            api_key=_DIAGNOSTIC_KEY,
            base_url=f"{_COLLISION_ROOT}/",
        )
        with pytest.raises(LLMAPIAuthorizationError) as caught:
            if mode == "stream_chat":
                list(facade.stream_chat([UserMessage(content="hello")]))
            else:
                facade.chat([UserMessage(content="hello")])

    _assert_diagnostics_redacted(caught.value, caplog)
    assert http.call_count == 1
    request = http.request_history[0]
    assert request.url == url
    assert request.headers["Authorization"] == f"Bearer {_DIAGNOSTIC_KEY}"
    assert request.json()["model"] == "arbitrary/model"


@pytest.mark.unit
def test_requests_timeout_diagnostics_redact_transport_log(caplog):
    url = f"{_COLLISION_ROOT}/chat/completions"
    with requests_mock.Mocker(real_http=False) as http:
        http.post(url, exc=requests.Timeout(_DIAGNOSTIC_TEXT))
        adapter = _new_adapter(
            api_key=_DIAGNOSTIC_KEY,
            model="arbitrary/model",
            base_url=_COLLISION_ROOT,
        )
        with pytest.raises(LLMAPITimeoutError) as caught:
            adapter.chat([UserMessage(content="hello")])

    _assert_diagnostics_redacted(caught.value, caplog)
    assert http.call_count == 1
    assert http.request_history[0].url == url
    assert http.request_history[0].headers["Authorization"] == f"Bearer {_DIAGNOSTIC_KEY}"


@pytest.mark.unit
def test_compatible_error_redaction_preserves_instance_without_reinitializing():
    adapter = _new_adapter(
        api_key=_DIAGNOSTIC_KEY,
        model="arbitrary/model",
        base_url=_COLLISION_ROOT,
    )
    error = LLMAPIClientError(detail=f"echo {_DIAGNOSTIC_KEY}")
    error.status_code = 418
    error.args = ("status detail", f"Client error: {_DIAGNOSTIC_KEY}")

    def forbidden_reinit():
        raise AssertionError("error lifecycle hook must not be called")

    error.__post_init__ = forbidden_reinit
    with pytest.raises(LLMAPIClientError) as caught:
        adapter.handle_error(error)

    assert caught.value is error
    assert caught.value.status_code == 418
    assert caught.value.detail == "echo [redacted]"
    assert _DIAGNOSTIC_KEY not in str(caught.value)
    assert _DIAGNOSTIC_KEY not in repr(caught.value)
    assert caught.value.args == ("status detail", "Client error: [redacted]")


@pytest.mark.unit
def test_success_content_and_caller_callback_error_are_not_redacted():
    url = f"{_COLLISION_ROOT}/chat/completions"
    with requests_mock.Mocker(real_http=False) as http:
        http.post(
            url,
            response_list=[
                {"json": text_response(content=_DIAGNOSTIC_KEY)},
                {
                    "text": "\n".join(
                        text_sse_lines(_DIAGNOSTIC_KEY, usage=_COLLISION_USAGE)
                    ),
                    "headers": {"Content-Type": "text/event-stream"},
                },
            ],
        )
        adapter = _new_adapter(
            api_key=_DIAGNOSTIC_KEY,
            model="arbitrary/model",
            base_url=_COLLISION_ROOT,
        )
        response = adapter.chat([UserMessage(content="hello")])
        assert response.content == _DIAGNOSTIC_KEY

        callback_error = RuntimeError(_DIAGNOSTIC_KEY, "caller-owned marker")

        def fail_callback(_delta):
            raise callback_error

        with pytest.raises(RuntimeError) as caught:
            list(
                adapter.stream_chat(
                    [UserMessage(content="hello")],
                    on_delta=fail_callback,
                )
            )

    assert caught.value is callback_error
    assert caught.value.args == (_DIAGNOSTIC_KEY, "caller-owned marker")
    assert http.call_count == 2


@pytest.mark.unit
def test_sync_httpx_http_error_diagnostics_are_redacted_before_logs(caplog, monkeypatch):
    real_client = httpx.Client
    requests_seen = []

    def handler(request):
        requests_seen.append(request)
        return httpx.Response(
            401,
            json={"error": {"message": _DIAGNOSTIC_TEXT, "type": "invalid_api_key"}},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "Client", lambda: real_client(transport=transport))
    adapter = OpenAICompatibleAdapter(
        api_key=_DIAGNOSTIC_KEY,
        model="arbitrary/model",
        base_url=_COLLISION_ROOT,
        transport="httpx",
    )
    with pytest.raises(LLMAPIAuthorizationError) as caught:
        adapter.chat([UserMessage(content="hello")])

    _assert_diagnostics_redacted(caught.value, caplog)
    assert len(requests_seen) == 1
    assert str(requests_seen[0].url) == f"{_COLLISION_ROOT}/chat/completions"
    assert requests_seen[0].headers["Authorization"] == f"Bearer {_DIAGNOSTIC_KEY}"


@pytest.mark.unit
@pytest.mark.parametrize("mode", ("chat", "stream_chat"))
@pytest.mark.parametrize(
    "failure,error_type",
    (("timeout", LLMAPITimeoutError), ("connection", LLMAPIClientError)),
)
def test_sync_httpx_transport_failure_diagnostics_are_redacted(
    mode, failure, error_type, caplog, monkeypatch
):
    real_client = httpx.Client
    requests_seen = []

    def handler(request):
        requests_seen.append(request)
        if failure == "timeout":
            raise httpx.TimeoutException(_DIAGNOSTIC_TEXT, request=request)
        raise httpx.ConnectError(_DIAGNOSTIC_TEXT, request=request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(httpx, "Client", lambda: real_client(transport=transport))
    adapter = OpenAICompatibleAdapter(
        api_key=_DIAGNOSTIC_KEY,
        model="arbitrary/model",
        base_url=_COLLISION_ROOT,
        transport="httpx",
    )
    with pytest.raises(error_type) as caught:
        if mode == "stream_chat":
            list(adapter.stream_chat([UserMessage(content="hello")]))
        else:
            adapter.chat([UserMessage(content="hello")])

    _assert_diagnostics_redacted(caught.value, caplog)
    assert len(requests_seen) == 1
    assert str(requests_seen[0].url) == f"{_COLLISION_ROOT}/chat/completions"
    assert requests_seen[0].headers["Authorization"] == f"Bearer {_DIAGNOSTIC_KEY}"


@pytest.mark.unit
def test_stream_error_diagnostics_are_redacted_and_skip_completion(caplog):
    url = f"{_COLLISION_ROOT}/chat/completions"
    event = {
        "error": {"type": "invalid_api_key", "message": _DIAGNOSTIC_TEXT}
    }
    event_text = f"event: error\ndata: {json.dumps(event)}\n\n"
    completed = []
    with requests_mock.Mocker(real_http=False) as http:
        http.post(url, text=event_text, headers={"Content-Type": "text/event-stream"})
        adapter = _new_adapter(
            api_key=_DIAGNOSTIC_KEY,
            model="arbitrary/model",
            base_url=_COLLISION_ROOT,
        )
        with pytest.raises(LLMAPIAuthorizationError) as caught:
            list(adapter.stream_chat([UserMessage(content="hello")], on_done=completed.append))

    _assert_diagnostics_redacted(caught.value, caplog)
    assert http.call_count == 1
    assert http.request_history[0].url == url
    assert completed == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("achat", "astream_chat"))
async def test_async_httpx_http_error_diagnostics_are_redacted_before_logs(
    mode, caplog, monkeypatch
):
    real_client = httpx.AsyncClient
    requests_seen = []

    def handler(request):
        requests_seen.append(request)
        return httpx.Response(
            401,
            json={"error": {"message": _DIAGNOSTIC_TEXT, "type": "invalid_api_key"}},
            request=request,
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda: real_client(transport=transport),
    )
    adapter = OpenAICompatibleAdapter(
        api_key=_DIAGNOSTIC_KEY,
        model="arbitrary/model",
        base_url=_COLLISION_ROOT,
    )
    with pytest.raises(LLMAPIAuthorizationError) as caught:
        if mode == "astream_chat":
            _ = [
                text
                async for text in adapter.astream_chat(
                    [UserMessage(content="hello")]
                )
            ]
        else:
            await adapter.achat([UserMessage(content="hello")])

    _assert_diagnostics_redacted(caught.value, caplog)
    assert len(requests_seen) == 1
    assert str(requests_seen[0].url) == f"{_COLLISION_ROOT}/chat/completions"
    assert requests_seen[0].headers["Authorization"] == f"Bearer {_DIAGNOSTIC_KEY}"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("achat", "astream_chat"))
@pytest.mark.parametrize(
    "failure,error_type",
    (("timeout", LLMAPITimeoutError), ("connection", LLMAPIClientError)),
)
async def test_async_httpx_transport_failure_diagnostics_are_redacted(
    mode, failure, error_type, caplog, monkeypatch
):
    real_client = httpx.AsyncClient
    requests_seen = []

    def handler(request):
        requests_seen.append(request)
        if failure == "timeout":
            raise httpx.TimeoutException(_DIAGNOSTIC_TEXT, request=request)
        raise httpx.ConnectError(_DIAGNOSTIC_TEXT, request=request)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda: real_client(transport=transport),
    )
    adapter = OpenAICompatibleAdapter(
        api_key=_DIAGNOSTIC_KEY,
        model="arbitrary/model",
        base_url=_COLLISION_ROOT,
    )
    with pytest.raises(error_type) as caught:
        if mode == "astream_chat":
            _ = [
                text
                async for text in adapter.astream_chat(
                    [UserMessage(content="hello")]
                )
            ]
        else:
            await adapter.achat([UserMessage(content="hello")])

    _assert_diagnostics_redacted(caught.value, caplog)
    assert len(requests_seen) == 1
    assert str(requests_seen[0].url) == f"{_COLLISION_ROOT}/chat/completions"
    assert requests_seen[0].headers["Authorization"] == f"Bearer {_DIAGNOSTIC_KEY}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "field, value",
    [("api_key", ""), ("api_key", None), ("model", ""), ("model", None)],
)
def test_constructor_rejects_invalid_common_key_and_model_with_value_error(field, value):
    values = {
        "api_key": "synthetic-key",
        "model": "arbitrary/model",
        "base_url": "http://localhost/v1",
    }
    values[field] = value
    with pytest.raises(ValueError):
        OpenAICompatibleAdapter(**values)


@pytest.mark.unit
@pytest.mark.parametrize(
    "selector",
    [
        {"company": "openai"},
        {"service_provider": "openai"},
    ],
)
def test_constructor_rejects_conflicting_native_selectors(selector):
    with pytest.raises(LLMConfigError):
        OpenAICompatibleAdapter(
            api_key="synthetic-key",
            model="gpt-4o",
            base_url="http://localhost/v1",
            **selector,
        )


@pytest.mark.unit
def test_client_hooks_use_selected_root_and_only_compatible_policy():
    adapter = OpenAICompatibleAdapter(
        api_key="synthetic-key",
        model="gpt-4o",
        base_url="https://example.invalid/proxy/v1/",
    )
    sync_client = adapter._create_sync_client()
    async_client = adapter._create_async_client()
    native_sync = OpenAISyncClient(api_key="native-key")
    native_async = OpenAIAsyncClient(api_key="native-key")

    assert isinstance(sync_client, OpenAISyncClient)
    assert isinstance(async_client, OpenAIAsyncClient)
    assert sync_client.endpoint == async_client.endpoint == "https://example.invalid/proxy/v1"
    assert sync_client.transport == "requests"
    assert sync_client._official_behavior_enabled is False
    assert async_client._official_behavior_enabled is False
    assert native_sync.endpoint == native_async.endpoint == "https://api.openai.com/v1"
    assert native_sync._official_behavior_enabled is True
    assert native_async._official_behavior_enabled is True


_COLLIDING_MODELS = ("gpt-5", "gpt-4.1-mini", "gpt-5-2025-08-07")
_COLLISION_ROOT = "https://example.invalid/gateway/tenant/v1"
_COLLISION_URL = f"{_COLLISION_ROOT}/chat/completions"
_COLLISION_KEY = "synthetic-collision-key"
_COLLISION_USAGE = usage_payload(
    2,
    3,
    5,
    prompt_tokens_details={"cached_tokens": 1},
)


def _forbid_official_lookup(name, attempts):
    def forbidden(*_args, **_kwargs):
        attempts.append(name)
        raise AssertionError(f"compatible mode performed official lookup: {name}")

    return forbidden


def _guard_official_model_lookups(monkeypatch):
    """Guard the bound lookup symbols used during construction and dispatch."""
    import src.llm_api_adapter.adapters.base_adapter as base_adapter_module
    import src.llm_api_adapter.llms.openai.sync_client as openai_sync_module

    attempts = []
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        _forbid_official_lookup("socket.getaddrinfo", attempts),
    )
    monkeypatch.setattr(
        OrganizationPluginDiscovery,
        "discover",
        _forbid_official_lookup("OrganizationPluginDiscovery.discover", attempts),
    )
    monkeypatch.setattr(
        base_adapter_module,
        "resolve_model_spec",
        _forbid_official_lookup("base_adapter.resolve_model_spec", attempts),
    )
    for symbol in (
        "model_api_variant",
        "model_spec_for",
        "apply_model_request_rules",
    ):
        monkeypatch.setattr(
            openai_sync_module,
            symbol,
            _forbid_official_lookup(f"openai.sync_client.{symbol}", attempts),
        )
    return attempts


def _assert_collision_response(response, model):
    assert isinstance(response, ChatResponse)
    assert response.content == "Fixture reply"
    assert response.model == model
    assert response.response_id == FIXTURE_ID
    assert response.finish_reason == "stop"
    assert response.usage.input_tokens == 2
    assert response.usage.output_tokens == 3
    assert response.usage.total_tokens == 5
    assert response.usage.cached_tokens == 1
    assert response.cost_input is None
    assert response.cost_output is None
    assert response.cost_total is None
    assert response.currency is None
    assert response.cost_breakdown is None


def _collision_body(model, *, streaming):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_completion_tokens": 17,
        "temperature": 0.4,
        "top_p": 0.8,
    }
    if streaming:
        body.update(
            stream=True,
            stream_options={"include_usage": True},
        )
    return body


def _collision_sse_lines(model):
    lines = text_sse_lines("Fixture reply", usage=_COLLISION_USAGE)
    model_field = f'"model":"{FIXTURE_MODEL}"'
    replacement = f'"model":"{model}"'
    return [line.replace(model_field, replacement) for line in lines]


def _fragmented_text_and_tool_sse_lines():
    """Join existing protocol fixtures into one valid tool-completion stream."""
    text = premature_eof_sse_lines(
        "Hi!", content_fragments=("H", "i!")
    )
    tool = tool_call_sse_lines(
        name="lookup",
        arguments_json='{"city":"Haifa"}',
        argument_fragments=("{\"ci", 'ty\":', '"Ha', 'ifa"}',),
        call_id="call_fragmented_1",
        usage=usage_payload(4, 2, 6),
    )
    return text + tool


def _tool_spec():
    return ToolSpec(
        name="lookup",
        json_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    )


class _CountedRequestsStreamResponse:
    def __init__(self, lines):
        self.lines = lines
        self.close_count = 0

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=False):
        del decode_unicode
        yield from self.lines

    def close(self):
        self.close_count += 1


class _CountedSyncByteStream(httpx.SyncByteStream):
    def __init__(self, content):
        self.content = content
        self.close_count = 0

    def __iter__(self):
        yield self.content

    def close(self):
        self.close_count += 1


class _CountedAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, lines, suspended=None, release=None):
        self.lines = lines
        self.suspended = suspended
        self.release = release
        self.close_count = 0

    async def __aiter__(self):
        for index, line in enumerate(self.lines):
            yield (line + "\n").encode()
            if index == len(self.lines) - 1 and self.suspended is not None:
                self.suspended.set()
                await self.release.wait()

    async def aclose(self):
        self.close_count += 1


def _counted_sync_stream_route(transport, lines, monkeypatch, model):
    requests_seen = []
    client_closes = []
    if transport == "requests":
        response = _CountedRequestsStreamResponse(lines)

        def post(url, **kwargs):
            requests_seen.append((url, kwargs))
            return response

        monkeypatch.setattr(requests, "post", post)
        adapter = UniversalLLMAPIAdapter(
            organization="openai_compatible", model=model,
            api_key=_COLLISION_KEY, base_url=_COLLISION_ROOT,
        )
        return adapter, requests_seen, response, client_closes, None

    byte_stream = _CountedSyncByteStream("\n".join(lines).encode())
    real_client = httpx.Client

    def handler(request):
        requests_seen.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=byte_stream,
        )

    client = real_client(transport=httpx.MockTransport(handler))
    original_close = client.close

    def close_client():
        client_closes.append(True)
        original_close()

    client.close = close_client
    monkeypatch.setattr(httpx, "Client", lambda: client)
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible", model=model,
        api_key=_COLLISION_KEY, base_url=_COLLISION_ROOT, transport="httpx",
    )
    return adapter, requests_seen, byte_stream, client_closes, client


def _counted_async_stream_route(
    lines, monkeypatch, model, *, suspended=None, release=None
):
    requests_seen = []
    client_closes = []
    byte_stream = _CountedAsyncByteStream(lines, suspended, release)
    real_client = httpx.AsyncClient

    def handler(request):
        requests_seen.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=byte_stream,
        )

    client = real_client(transport=httpx.MockTransport(handler))
    original_aclose = client.aclose

    async def close_client():
        client_closes.append(True)
        await original_aclose()

    client.aclose = close_client
    monkeypatch.setattr(httpx, "AsyncClient", lambda: client)
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible", model=model,
        api_key=_COLLISION_KEY, base_url=_COLLISION_ROOT,
    )
    return adapter, requests_seen, byte_stream, client_closes, client


@pytest.mark.unit
@pytest.mark.parametrize("model", _COLLIDING_MODELS)
@pytest.mark.parametrize("mode", ("chat", "stream_chat"))
def test_compatible_sync_collision_models_use_only_chat_completions(
    model, mode, monkeypatch
):
    forbidden = _guard_official_model_lookups(monkeypatch)
    stream = mode == "stream_chat"
    with requests_mock.Mocker(real_http=False) as http:
        if stream:
            http.post(
                _COLLISION_URL,
                text="\n".join(_collision_sse_lines(model)),
                headers={"Content-Type": "text/event-stream"},
            )
        else:
            http.post(
                _COLLISION_URL,
                json=text_response(model=model, usage=_COLLISION_USAGE),
            )
        adapter = UniversalLLMAPIAdapter(
            organization="openai_compatible",
            model=model,
            api_key=_COLLISION_KEY,
            base_url=f"{_COLLISION_ROOT}/",
        )
        assert adapter.adapter.model_spec is None
        assert adapter.adapter.pricing is None
        assert adapter.adapter.is_reasoning is False
        assert adapter.adapter.is_adaptive_thinking is False

        if stream:
            completed = []
            visible = list(
                adapter.stream_chat(
                    messages=[UserMessage(content="Say hello.")],
                    max_tokens=17,
                    temperature=0.4,
                    top_p=0.8,
                    on_done=completed.append,
                )
            )
            assert visible == ["Fixture reply"]
            assert len(completed) == 1
            _assert_collision_response(completed[0], model)
        else:
            _assert_collision_response(
                adapter.chat(
                    messages=[UserMessage(content="Say hello.")],
                    max_tokens=17,
                    temperature=0.4,
                    top_p=0.8,
                ),
                model,
            )

        assert http.call_count == 1
        request = http.request_history[0]
        assert request.method == "POST"
        assert request.url == _COLLISION_URL
        assert request.headers["Authorization"] == f"Bearer {_COLLISION_KEY}"
        assert request.headers["Content-Type"] == "application/json"
        assert request.json() == _collision_body(model, streaming=stream)
        assert forbidden == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("model", _COLLIDING_MODELS)
@pytest.mark.parametrize("mode", ("achat", "astream_chat"))
async def test_compatible_async_collision_models_use_only_chat_completions(
    model, mode, monkeypatch
):
    forbidden = _guard_official_model_lookups(monkeypatch)
    requests_seen = []
    real_async_client = httpx.AsyncClient
    stream = mode == "astream_chat"

    def handler(request):
        requests_seen.append(request)
        if stream:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content="\n".join(_collision_sse_lines(model)),
            )
        return httpx.Response(
            200,
            json=text_response(model=model, usage=_COLLISION_USAGE),
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda: real_async_client(transport=transport),
    )
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model=model,
        api_key=_COLLISION_KEY,
        base_url=f"{_COLLISION_ROOT}/",
    )
    assert adapter.adapter.model_spec is None
    assert adapter.adapter.pricing is None
    assert adapter.adapter.is_reasoning is False
    assert adapter.adapter.is_adaptive_thinking is False

    if stream:
        completed = []
        visible = [
            text
            async for text in adapter.astream_chat(
                messages=[UserMessage(content="Say hello.")],
                max_tokens=17,
                temperature=0.4,
                top_p=0.8,
                on_done=completed.append,
            )
        ]
        assert visible == ["Fixture reply"]
        assert len(completed) == 1
        _assert_collision_response(completed[0], model)
    else:
        _assert_collision_response(
            await adapter.achat(
                messages=[UserMessage(content="Say hello.")],
                max_tokens=17,
                temperature=0.4,
                top_p=0.8,
            ),
            model,
        )

    assert len(requests_seen) == 1
    request = requests_seen[0]
    assert request.method == "POST"
    assert str(request.url) == _COLLISION_URL
    assert request.headers["Authorization"] == f"Bearer {_COLLISION_KEY}"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == _collision_body(model, streaming=stream)
    assert forbidden == []


_FAILURE_CASES = (
    ("authorization", 401, LLMAPIAuthorizationError, None),
    ("unsupported-feature", 400, LLMAPIClientError, None),
    ("rate-limit", 429, LLMAPIRateLimitError, None),
    ("server-error", 503, LLMAPIServerError, None),
    ("timeout", None, LLMAPITimeoutError, "timeout"),
    ("connection", None, LLMAPIClientError, "connection"),
)
_FAILURE_MODEL = "gpt-5"


def _failure_payload(case_name):
    return {
        "error": {
            "message": f"synthetic {case_name} failure",
            "type": "invalid_request_error" if case_name == "unsupported-feature" else "server_error",
            "code": "unsupported_parameter" if case_name == "unsupported-feature" else case_name,
        }
    }


@pytest.mark.unit
@pytest.mark.parametrize("mode", ("chat", "stream_chat"))
@pytest.mark.parametrize(
    "case_name,status,error_type,transport_failure", _FAILURE_CASES
)
def test_sync_failures_never_fall_back_or_retry(
    mode, case_name, status, error_type, transport_failure, monkeypatch
):
    forbidden = _guard_official_model_lookups(monkeypatch)
    streaming = mode == "stream_chat"

    with requests_mock.Mocker(real_http=False) as http:
        http.post(requests_mock.ANY, status_code=599)
        if transport_failure == "timeout":
            http.post(_COLLISION_URL, exc=requests.Timeout("synthetic timeout"))
        elif transport_failure == "connection":
            http.post(
                _COLLISION_URL,
                exc=requests.ConnectionError("synthetic connection failure"),
            )
        else:
            http.post(
                _COLLISION_URL,
                status_code=status,
                json=_failure_payload(case_name),
            )
        adapter = UniversalLLMAPIAdapter(
            organization="openai_compatible",
            model=_FAILURE_MODEL,
            api_key=_COLLISION_KEY,
            base_url=f"{_COLLISION_ROOT}/",
        )
        completed = []
        with pytest.raises(error_type):
            if streaming:
                list(adapter.stream_chat(
                    messages=[UserMessage(content="Say hello.")],
                    max_tokens=17,
                    temperature=0.4,
                    top_p=0.8,
                    on_done=completed.append,
                ))
            else:
                adapter.chat(
                    messages=[UserMessage(content="Say hello.")],
                    max_tokens=17,
                    temperature=0.4,
                    top_p=0.8,
                )

        assert http.call_count == 1
        assert len(http.request_history) == 1
        request = http.request_history[0]
        assert request.method == "POST"
        assert request.url == _COLLISION_URL
        assert request.headers["Authorization"] == f"Bearer {_COLLISION_KEY}"
        assert request.json() == _collision_body(_FAILURE_MODEL, streaming=streaming)
        assert completed == []
        assert forbidden == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("achat", "astream_chat"))
@pytest.mark.parametrize(
    "case_name,status,error_type,transport_failure", _FAILURE_CASES
)
async def test_async_failures_never_fall_back_or_retry(
    mode, case_name, status, error_type, transport_failure, monkeypatch
):
    forbidden = _guard_official_model_lookups(monkeypatch)
    real_async_client = httpx.AsyncClient
    streaming = mode == "astream_chat"
    outgoing = []

    def handler(request):
        outgoing.append(request)
        if transport_failure == "timeout":
            raise httpx.TimeoutException("synthetic timeout", request=request)
        if transport_failure == "connection":
            raise httpx.ConnectError("synthetic connection failure", request=request)
        return httpx.Response(
            status,
            json=_failure_payload(case_name),
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda: real_async_client(transport=transport),
    )
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model=_FAILURE_MODEL,
        api_key=_COLLISION_KEY,
        base_url=f"{_COLLISION_ROOT}/",
    )
    completed = []
    with pytest.raises(error_type):
        if streaming:
            _ = [
                text
                async for text in adapter.astream_chat(
                    messages=[UserMessage(content="Say hello.")],
                    max_tokens=17,
                    temperature=0.4,
                    top_p=0.8,
                    on_done=completed.append,
                )
            ]
        else:
            await adapter.achat(
                messages=[UserMessage(content="Say hello.")],
                max_tokens=17,
                temperature=0.4,
                top_p=0.8,
            )

    assert len(outgoing) == 1
    request = outgoing[0]
    assert request.method == "POST"
    assert str(request.url) == _COLLISION_URL
    assert request.headers["Authorization"] == f"Bearer {_COLLISION_KEY}"
    assert json.loads(request.content) == _collision_body(
        _FAILURE_MODEL, streaming=streaming
    )
    assert completed == []
    assert forbidden == []


def _native_gpt5_response():
    return {
        "id": "resp-native-fixture",
        "model": "gpt-5",
        "status": "completed",
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "Official fixture"}],
        }],
        "usage": {
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 5,
            "input_tokens_details": {"cached_tokens": 0},
        },
    }


def _assert_native_gpt5_request(request):
    assert request.method == "POST"
    assert str(request.url).endswith("/v1/responses")
    assert request.headers["Authorization"] == "Bearer synthetic-official-key"
    body = (
        request.json()
        if hasattr(request, "json")
        else json.loads(request.content)
    )
    assert body["model"] == "gpt-5"
    assert body["max_output_tokens"] == 17
    assert body["reasoning"] == {"effort": "minimal"}
    assert "max_tokens" not in body
    assert "messages" not in body


@pytest.mark.unit
@pytest.mark.parametrize("first_compatible_fails", (False, True))
def test_compatible_and_official_sync_instances_keep_roots_keys_and_profiles_isolated(
    first_compatible_fails, monkeypatch
):
    forbidden = []
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        _forbid_official_lookup("socket.getaddrinfo", forbidden),
    )
    monkeypatch.setattr(
        OrganizationPluginDiscovery,
        "discover",
        _forbid_official_lookup("OrganizationPluginDiscovery.discover", forbidden),
    )

    with requests_mock.Mocker(real_http=False) as http:
        http.post(requests_mock.ANY, status_code=599)
        compatible_responses = [
            {"status_code": 400, "json": _failure_payload("unsupported-feature")}
            if first_compatible_fails
            else {"status_code": 200, "json": text_response(
                model="gpt-5", usage=_COLLISION_USAGE
            )},
            {"status_code": 200, "json": text_response(
                model="gpt-5", usage=_COLLISION_USAGE
            )},
        ]
        http.post(_COLLISION_URL, compatible_responses)
        http.post(
            "https://api.openai.com/v1/responses",
            status_code=200,
            json=_native_gpt5_response(),
        )
        compatible = UniversalLLMAPIAdapter(
            organization="openai_compatible",
            model="gpt-5",
            api_key=_COLLISION_KEY,
            base_url=f"{_COLLISION_ROOT}/",
        )
        assert compatible.adapter.model_spec is None
        assert compatible.adapter.pricing is None
        if first_compatible_fails:
            with pytest.raises(LLMAPIClientError):
                compatible.chat(
                    [UserMessage(content="Say hello.")],
                    max_tokens=17,
                    temperature=0.4,
                    top_p=0.8,
                )
        else:
            _assert_collision_response(
                compatible.chat(
                    [UserMessage(content="Say hello.")],
                    max_tokens=17,
                    temperature=0.4,
                    top_p=0.8,
                ),
                "gpt-5",
            )

        official = UniversalLLMAPIAdapter(
            organization="openai",
            model="gpt-5",
            api_key="synthetic-official-key",
        )
        assert official.adapter.model_spec is not None
        assert official.adapter.pricing is not None
        official_response = official.chat(
            [UserMessage(content="Say hello.")], max_tokens=17
        )
        assert official_response.content == "Official fixture"
        assert official_response.cost_total is not None

        _assert_collision_response(
            compatible.chat(
                [UserMessage(content="Say hello.")],
                max_tokens=17,
                temperature=0.4,
                top_p=0.8,
            ),
            "gpt-5",
        )

    attempts = http.request_history
    assert len(attempts) == 3
    assert [request.url for request in attempts] == [
        _COLLISION_URL,
        "https://api.openai.com/v1/responses",
        _COLLISION_URL,
    ]
    for index in (0, 2):
        request = attempts[index]
        assert request.headers["Authorization"] == f"Bearer {_COLLISION_KEY}"
        assert request.json() == _collision_body("gpt-5", streaming=False)
    _assert_native_gpt5_request(attempts[1])
    assert forbidden == []


@pytest.mark.unit
@pytest.mark.parametrize("transport", ("requests", "httpx"))
def test_compatible_fragmented_text_tool_and_usage_complete_in_sync_transports(
    transport, monkeypatch
):
    model = "vendor/arbitrary-stream-model"
    lines = _fragmented_text_and_tool_sse_lines()
    completed = []
    tools = []
    chunks = []
    final_order = []

    def record_chunk(chunk):
        chunks.append(chunk)
        final_order.append(f"chunk:{chunk.text}")

    def record_delta(text):
        final_order.append(f"delta:{text}")

    def record_tool(tool):
        tools.append(tool)
        final_order.append("tool")

    def record_done(response):
        completed.append(response)
        final_order.append("done")

    adapter, requests_seen, stream_resource, client_closes, client = (
        _counted_sync_stream_route(transport, lines, monkeypatch, model)
    )
    try:
        visible = []
        for text in adapter.stream_chat(
                [UserMessage(content="Find a city.")],
                tools=[_tool_spec()], buffer_chars=2,
                on_chunk=record_chunk, on_delta=record_delta,
                on_tool_call=record_tool, on_done=record_done,
            ):
            visible.append(text)
            final_order.append(f"yield:{text}")
        assert len(requests_seen) == 1
        if transport == "requests":
            assert requests_seen[0][0] == _COLLISION_URL
            assert requests_seen[0][1]["json"]["model"] == model
        else:
            assert str(requests_seen[0].url) == _COLLISION_URL
            assert json.loads(requests_seen[0].content)["model"] == model
            assert len(client_closes) == 1
        assert stream_resource.close_count == 1
        assert visible == ["Hi", "!"]
        assert len(tools) == len(completed) == 1
        assert final_order == [
            "chunk:Hi", "delta:Hi", "yield:Hi", "chunk:!", "delta:!",
            "yield:!", "tool", "done",
        ]
        assert chunks[0].usage is None
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.input_tokens == 4
        assert chunks[-1].usage.output_tokens == 2
        assert chunks[-1].usage.total_tokens == 6
        assert tools[0].name == "lookup"
        assert tools[0].arguments == {"city": "Haifa"}
        assert tools[0].call_id == "call_fragmented_1"
        response = completed[0]
        assert response.content == "Hi!"
        assert response.tool_calls == [tools[0]]
        assert response.usage.input_tokens == 4
        assert response.usage.output_tokens == 2
        assert response.usage.total_tokens == 6
        assert response.cost_input is response.cost_output is response.cost_total is None
        assert response.currency is None
        assert response.cost_breakdown is None
    finally:
        if stream_resource.close_count == 0:
            stream_resource.close()
        if transport == "httpx" and not client_closes:
            client.close()


@pytest.mark.unit
@pytest.mark.parametrize("transport", ("requests", "httpx"))
@pytest.mark.parametrize("outcome", ("eof", "error"))
def test_compatible_sync_stream_without_terminal_outcome_is_not_success(
    transport, outcome, monkeypatch
):
    model = "vendor/unregistered-model"
    lines = premature_eof_sse_lines("pending")
    if outcome == "error":
        lines += event_lines({"type": "error", "error": {"message": "stream rejected"}})
    completed = []
    visible = []
    adapter, requests_seen, stream_resource, client_closes, client = (
        _counted_sync_stream_route(transport, lines, monkeypatch, model)
    )
    caught = None
    try:
        try:
            for text in adapter.stream_chat(
                [UserMessage(content="Hello")], buffer_chars=64,
                on_done=completed.append,
            ):
                visible.append(text)
        except LLMAPIClientError as error:
            caught = error

        assert len(requests_seen) == 1
        if transport == "requests":
            assert requests_seen[0][0] == _COLLISION_URL
            source_close_count = stream_resource.close_count
            assert source_close_count == 1
        else:
            assert str(requests_seen[0].url) == _COLLISION_URL
            source_close_count = stream_resource.close_count
            client_close_count = len(client_closes)
            assert source_close_count == 1
            assert client_close_count == 1
        assert isinstance(caught, LLMAPIClientError) and not visible and not completed, (
            f"terminal outcome={caught!r}; visible={visible!r}; done={completed!r}"
        )
        if outcome == "error":
            assert "stream rejected" in str(caught)
    finally:
        if stream_resource.close_count == 0:
            stream_resource.close()
        if transport == "httpx" and not client_closes:
            client.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compatible_async_fragmented_text_tool_and_usage_complete(monkeypatch):
    model = "vendor/arbitrary-stream-model"
    lines = _fragmented_text_and_tool_sse_lines()
    completed = []
    tools = []
    chunks = []
    final_order = []

    def record_chunk(chunk):
        chunks.append(chunk)
        final_order.append(f"chunk:{chunk.text}")

    def record_delta(text):
        final_order.append(f"delta:{text}")

    def record_tool(tool):
        tools.append(tool)
        final_order.append("tool")

    def record_done(response):
        completed.append(response)
        final_order.append("done")
    adapter, requests_seen, stream_resource, client_closes, client = (
        _counted_async_stream_route(lines, monkeypatch, model)
    )
    try:
        visible = []
        async for text in adapter.astream_chat(
                [UserMessage(content="Find a city.")],
                tools=[_tool_spec()], buffer_chars=2,
                on_chunk=record_chunk, on_delta=record_delta,
                on_tool_call=record_tool, on_done=record_done,
            ):
            visible.append(text)
            final_order.append(f"yield:{text}")

        assert len(requests_seen) == 1
        assert str(requests_seen[0].url) == _COLLISION_URL
        assert json.loads(requests_seen[0].content)["model"] == model
        assert stream_resource.close_count == 1
        assert len(client_closes) == 1
        assert visible == ["Hi", "!"]
        assert len(tools) == len(completed) == 1
        assert final_order == [
            "chunk:Hi", "delta:Hi", "yield:Hi", "chunk:!", "delta:!",
            "yield:!", "tool", "done",
        ]
        assert len(chunks) == 2
        assert chunks[0].usage is None
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.input_tokens == 4
        assert chunks[-1].usage.output_tokens == 2
        assert chunks[-1].usage.total_tokens == 6
        assert tools[0].name == "lookup"
        assert tools[0].arguments == {"city": "Haifa"}
        assert tools[0].call_id == "call_fragmented_1"
        assert completed[0].tool_calls == [tools[0]]
        assert completed[0].usage.input_tokens == 4
        assert completed[0].usage.output_tokens == 2
        assert completed[0].usage.total_tokens == 6
        assert completed[0].cost_total is None
        assert completed[0].cost_input is None
        assert completed[0].cost_output is None
        assert completed[0].currency is None
        assert completed[0].cost_breakdown is None
    finally:
        if stream_resource.close_count == 0:
            await stream_resource.aclose()
        if not client_closes:
            await client.aclose()


_METADATA_ABSENT = object()
_METADATA_ALL_UNKNOWN = (None, None, None, None, None)


def _one_invalid_base_counter(field, value):
    counters = {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}
    counters[field] = value
    return usage_payload(**counters)


_METADATA_CASES = [
    pytest.param(_METADATA_ABSENT, None, id="usage-absent"),
    pytest.param(None, None, id="usage-null"),
    pytest.param("malformed", None, id="usage-string"),
    pytest.param([1, 2], None, id="usage-list"),
    pytest.param(False, None, id="usage-bool"),
    pytest.param({}, _METADATA_ALL_UNKNOWN, id="empty-mapping"),
    pytest.param(usage_payload(None, None, None), _METADATA_ALL_UNKNOWN, id="all-null-counters"),
    pytest.param(zero_usage_payload(), (0, 0, 0, None, None), id="reported-zero"),
    pytest.param(
        usage_payload(prompt_tokens=False, completion_tokens=0, total_tokens=5),
        (None, 0, 5, None, None),
        id="bool-false-is-not-zero",
    ),
    pytest.param(usage_payload(prompt_tokens=7), (7, None, None, None, None), id="input-only"),
    pytest.param(usage_payload(completion_tokens=4), (None, 4, None, None, None), id="output-only"),
    pytest.param(usage_payload(total_tokens=11), (None, None, 11, None, None), id="total-only"),
    pytest.param(
        usage_payload(prompt_tokens=7, completion_tokens=4),
        (7, 4, None, None, None),
        id="input-and-output-no-total",
    ),
    *[
        pytest.param(
            _one_invalid_base_counter(field, invalid),
            tuple(None if index == field_index else (5, 6, 11)[index] for index in range(3))
            + (None, None),
            id=f"{field}-{label}",
        )
        for field_index, field in enumerate(("prompt_tokens", "completion_tokens", "total_tokens"))
        for label, invalid in (
            ("negative", -1), ("bool", True), ("string", "5"),
            ("float", 5.0), ("null", None),
        )
    ],
    pytest.param(
        usage_payload(10, 3, 13), (10, 3, 13, None, None),
        id="cache-details-absent",
    ),
    pytest.param(
        usage_payload(10, 3, 13, prompt_tokens_details=None),
        (10, 3, 13, None, None), id="cache-details-null",
    ),
    pytest.param(
        usage_payload(10, 3, 13, prompt_tokens_details="malformed"),
        (10, 3, 13, None, None), id="cache-details-nonmapping",
    ),
    pytest.param(
        usage_payload(10, 3, 13, prompt_tokens_details={}),
        (10, 3, 13, None, None), id="cache-details-empty",
    ),
    pytest.param(
        usage_payload(10, 3, 13, prompt_tokens_details={"cached_tokens": 0}),
        (10, 3, 13, 0, None), id="cache-read-zero-only",
    ),
    pytest.param(
        usage_payload(10, 3, 13, prompt_tokens_details={"cache_write_tokens": 2}),
        (10, 3, 13, None, 2), id="cache-write-only",
    ),
    pytest.param(
        usage_payload(
            10, 3, 13,
            prompt_tokens_details={"cached_tokens": 4, "cache_write_tokens": 2},
        ),
        (10, 3, 13, 4, 2), id="cache-split-valid",
    ),
    pytest.param(
        usage_payload(
            10, 3, 13,
            prompt_tokens_details={"cached_tokens": True, "cache_write_tokens": 2},
        ),
        (10, 3, 13, None, 2), id="cache-read-bool-keeps-write",
    ),
    pytest.param(
        usage_payload(
            10, 3, 13,
            prompt_tokens_details={"cached_tokens": 4, "cache_write_tokens": -1},
        ),
        (10, 3, 13, 4, None), id="cache-write-negative-keeps-read",
    ),
    pytest.param(
        usage_payload(
            10, 3, 13,
            prompt_tokens_details={"cached_tokens": 8, "cache_write_tokens": 5},
        ),
        (10, 3, 13, None, None), id="cache-sum-contradicts-input",
    ),
    pytest.param(
        usage_payload(
            10, 3, 13,
            prompt_tokens_details={"cached_tokens": 2, "cache_write_tokens": 11},
        ),
        (10, 3, 13, None, None), id="cache-component-exceeds-input",
    ),
    pytest.param(
        usage_payload(completion_tokens=3, prompt_tokens_details={"cached_tokens": 2}),
        (None, 3, None, 2, None), id="cache-retained-without-base-input",
    ),
]


def _metadata_response_body(usage):
    if usage is _METADATA_ABSENT:
        return text_response(model="gpt-5")
    return text_response(model="gpt-5", usage=usage)


def _metadata_stream_lines(usage):
    if usage is _METADATA_ABSENT:
        lines = text_sse_lines("Fixture reply")
    else:
        lines = text_sse_lines("Fixture reply", usage=usage)
    return [line.replace(f'"model":"{FIXTURE_MODEL}"', '"model":"gpt-5"') for line in lines]


def _assert_compatible_metadata_result(response, expected_usage):
    assert isinstance(response, ChatResponse)
    assert response.content == "Fixture reply"
    assert response.model == "gpt-5"
    assert response.response_id == FIXTURE_ID
    assert response.finish_reason == "stop"
    if expected_usage is None:
        assert response.usage is None
    else:
        assert response.usage is not None
        assert (
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens,
            response.usage.cached_tokens,
            response.usage.cache_write_tokens,
        ) == expected_usage
        for actual, expected in zip(
            (
                response.usage.input_tokens,
                response.usage.output_tokens,
                response.usage.total_tokens,
                response.usage.cached_tokens,
                response.usage.cache_write_tokens,
            ),
            expected_usage,
        ):
            if expected == 0:
                assert type(actual) is int
    assert response.cost_input is None
    assert response.cost_output is None
    assert response.cost_total is None
    assert response.currency is None
    assert response.cost_breakdown is None


def _assert_official_collision_has_no_profile(adapter):
    assert adapter.adapter.model_spec is None
    assert adapter.adapter.pricing is None


def _metadata_input_snapshot(usage):
    return usage if usage is _METADATA_ABSENT else deepcopy(usage)


def _assert_metadata_input_unchanged(usage, snapshot):
    if snapshot is _METADATA_ABSENT:
        assert usage is _METADATA_ABSENT
    else:
        assert usage == snapshot


@pytest.mark.unit
@pytest.mark.parametrize("mode", ("chat", "stream_chat"))
@pytest.mark.parametrize("usage, expected_usage", _METADATA_CASES)
def test_compatible_sync_metadata_contract_matrix(mode, usage, expected_usage, monkeypatch):
    streaming = mode == "stream_chat"
    usage_snapshot = _metadata_input_snapshot(usage)
    forbidden = _guard_official_model_lookups(monkeypatch)
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible", model="gpt-5",
        api_key=_COLLISION_KEY, base_url=_COLLISION_ROOT,
    )
    completed = []
    with requests_mock.Mocker(real_http=False) as http:
        if streaming:
            http.post(
                _COLLISION_URL,
                text="\n".join(_metadata_stream_lines(usage)),
                headers={"Content-Type": "text/event-stream"},
            )
            visible = list(adapter.stream_chat(
                [UserMessage(content="Say hello.")], on_done=completed.append
            ))
            assert visible == ["Fixture reply"]
            assert len(completed) == 1
            _assert_compatible_metadata_result(completed[0], expected_usage)
        else:
            http.post(_COLLISION_URL, json=_metadata_response_body(usage))
            response = adapter.chat([UserMessage(content="Say hello.")])
            _assert_compatible_metadata_result(response, expected_usage)
        assert http.call_count == 1
        request = http.request_history[0]
        assert request.method == "POST"
        assert request.url == _COLLISION_URL
        assert request.json()["model"] == "gpt-5"
    _assert_official_collision_has_no_profile(adapter)
    _assert_metadata_input_unchanged(usage, usage_snapshot)
    assert forbidden == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("achat", "astream_chat"))
@pytest.mark.parametrize("usage, expected_usage", _METADATA_CASES)
async def test_compatible_async_metadata_contract_matrix(mode, usage, expected_usage, monkeypatch):
    streaming = mode == "astream_chat"
    usage_snapshot = _metadata_input_snapshot(usage)
    forbidden = _guard_official_model_lookups(monkeypatch)
    outgoing = []
    real_client = httpx.AsyncClient

    def handler(request):
        outgoing.append(request)
        if streaming:
            return httpx.Response(
                200, headers={"Content-Type": "text/event-stream"},
                content="\n".join(_metadata_stream_lines(usage)),
            )
        return httpx.Response(200, json=_metadata_response_body(usage))

    client = real_client(transport=httpx.MockTransport(handler))
    constructions = _install_counted_httpx_client(monkeypatch, client, asynchronous=True)
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible", model="gpt-5",
        api_key=_COLLISION_KEY, base_url=_COLLISION_ROOT,
    )
    completed = []
    try:
        if streaming:
            visible = [text async for text in adapter.astream_chat(
                [UserMessage(content="Say hello.")], on_done=completed.append
            )]
            assert visible == ["Fixture reply"]
            assert len(completed) == 1
            _assert_compatible_metadata_result(completed[0], expected_usage)
        else:
            response = await adapter.achat([UserMessage(content="Say hello.")])
            _assert_compatible_metadata_result(response, expected_usage)
        assert len(outgoing) == 1
        assert len(constructions) == 1
        request = outgoing[0]
        assert request.method == "POST"
        assert str(request.url) == _COLLISION_URL
        assert json.loads(request.content)["model"] == "gpt-5"
        _assert_official_collision_has_no_profile(adapter)
        _assert_metadata_input_unchanged(usage, usage_snapshot)
        assert forbidden == []
    finally:
        await client.aclose()


def _caller_pricing_fixture():
    return Pricing.from_dict(
        [
            {
                "up_to_prompt_tokens": 2,
                "input_per_1m": 100,
                "output_per_1m": 200,
                "cache_read_input_per_1m": 10,
            },
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 300,
                "output_per_1m": 400,
                "cache_read_input_per_1m": 30,
            },
        ],
        currency="EUR",
    )


def _assert_caller_pricing_response(response):
    # Select by inclusive input=2; cached input is a disjoint priced component.
    assert response.usage.input_tokens == 2
    assert response.usage.cached_tokens == 1
    assert response.cost_input == pytest.approx(110 / 1_000_000)
    assert response.cost_output == pytest.approx(600 / 1_000_000)
    assert response.cost_total == pytest.approx(710 / 1_000_000)
    assert response.currency == "EUR"
    assert response.cost_breakdown is None


def _caller_boundary_pricing():
    return Pricing.from_dict(
        [
            {
                "up_to_prompt_tokens": 100,
                "input_per_1m": 100,
                "output_per_1m": 200,
                "cache_read_input_per_1m": 1,
                "cache_write_input_per_1m": 2,
            },
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 300,
                "output_per_1m": 400,
                "cache_read_input_per_1m": 30,
                "cache_write_input_per_1m": 40,
            },
        ],
        currency="CAD",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("input_tokens", "input_rate", "output_rate"),
    [(99, 100, 200), (100, 100, 200), (101, 300, 400)],
    ids=["below-boundary", "at-boundary", "above-boundary"],
)
def test_caller_pricing_compatible_tiers_use_full_inclusive_input(
    input_tokens, input_rate, output_rate, monkeypatch
):
    forbidden = _guard_official_model_lookups(monkeypatch)
    model = "gpt-5"
    adapter = OpenAICompatibleAdapter(
        api_key=_COLLISION_KEY,
        model=model,
        base_url=_COLLISION_ROOT,
        pricing=_caller_boundary_pricing(),
    )
    usage = usage_payload(
        input_tokens,
        5,
        input_tokens + 5,
        prompt_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
    )
    with requests_mock.Mocker(real_http=False) as http:
        http.post(_COLLISION_URL, json=text_response(model=model, usage=usage))
        response = adapter.chat([UserMessage("hello")])

    assert adapter.model_spec is None
    assert adapter.pricing is not None
    assert forbidden == []
    assert http.call_count == 1
    assert http.request_history[0].json()["model"] == model
    assert response.cost_input == pytest.approx(input_tokens * input_rate / 1_000_000)
    assert response.cost_output == pytest.approx(5 * output_rate / 1_000_000)
    assert response.cost_total == pytest.approx(
        (input_tokens * input_rate + 5 * output_rate) / 1_000_000
    )
    assert response.currency == "CAD"
    assert response.cost_breakdown is None


@pytest.mark.unit
def test_caller_pricing_cache_tier_uses_full_input_and_disjoint_read_write_counts(
    monkeypatch,
):
    forbidden = _guard_official_model_lookups(monkeypatch)
    model = "gpt-5"
    adapter = OpenAICompatibleAdapter(
        api_key=_COLLISION_KEY,
        model=model,
        base_url=_COLLISION_ROOT,
        pricing=_caller_boundary_pricing(),
    )
    # Inclusive input crosses 100; uncached input is only 11 and stays below it.
    usage = usage_payload(
        101,
        5,
        106,
        prompt_tokens_details={"cached_tokens": 60, "cache_write_tokens": 30},
    )
    with requests_mock.Mocker(real_http=False) as http:
        http.post(_COLLISION_URL, json=text_response(model=model, usage=usage))
        response = adapter.chat([UserMessage("hello")])

    assert adapter.model_spec is None
    assert adapter.pricing is not None
    assert forbidden == []
    assert http.request_history[0].json()["model"] == model
    assert response.usage.input_tokens == 101
    assert response.usage.cached_tokens == 60
    assert response.usage.cache_write_tokens == 30
    assert response.cost_input == pytest.approx((11 * 300 + 60 * 30 + 30 * 40) / 1_000_000)
    assert response.cost_output == pytest.approx(5 * 400 / 1_000_000)
    assert response.cost_total == pytest.approx(8_300 / 1_000_000)
    assert response.currency == "CAD"
    assert response.cost_breakdown is None


@pytest.mark.unit
@pytest.mark.parametrize("model", ["vendor/caller-tariff-model", "gpt-5"])
@pytest.mark.parametrize("transport", ["requests", "httpx"])
@pytest.mark.parametrize("mode", ["chat", "stream_chat"])
def test_caller_pricing_is_used_for_compatible_sync_requests(
    model, transport, mode, monkeypatch
):
    forbidden = _guard_official_model_lookups(monkeypatch)
    pricing = _caller_pricing_fixture()
    response_body = text_response(model=model, usage=_COLLISION_USAGE)
    adapter = OpenAICompatibleAdapter(
        api_key=_COLLISION_KEY,
        model=model,
        base_url=_COLLISION_ROOT,
        transport=transport,
        pricing=pricing,
    )
    outgoing = []
    streaming = mode == "stream_chat"
    stream_body = "\n".join(text_sse_lines("Fixture reply", usage=_COLLISION_USAGE))

    if transport == "requests":
        with requests_mock.Mocker(real_http=False) as http:
            if streaming:
                http.post(
                    _COLLISION_URL,
                    text=stream_body,
                    headers={"Content-Type": "text/event-stream"},
                )
                completed = []
                assert list(adapter.stream_chat(
                    [UserMessage("hello")],
                    buffer_chars=len("Fixture reply") + 20,
                    on_done=completed.append,
                )) == ["Fixture reply"]
                response = completed[0]
            else:
                http.post(_COLLISION_URL, json=response_body)
                response = adapter.chat([UserMessage("hello")])
            assert http.call_count == 1
            assert http.request_history[0].url == _COLLISION_URL
            assert http.request_history[0].json()["model"] == model
    else:
        def handler(request):
            outgoing.append(request)
            if streaming:
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/event-stream"},
                    content=stream_body,
                )
            return httpx.Response(200, json=response_body)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        _install_counted_httpx_client(monkeypatch, client)
        if streaming:
            completed = []
            assert list(adapter.stream_chat(
                [UserMessage("hello")],
                buffer_chars=len("Fixture reply") + 20,
                on_done=completed.append,
            )) == ["Fixture reply"]
            response = completed[0]
        else:
            response = adapter.chat([UserMessage("hello")])
        assert len(outgoing) == 1
        assert str(outgoing[0].url) == _COLLISION_URL
        assert json.loads(outgoing[0].content)["model"] == model
        client.close()

    assert adapter.model_spec is None
    assert forbidden == []
    _assert_caller_pricing_response(response)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["vendor/caller-tariff-model", "gpt-5"])
@pytest.mark.parametrize("mode", ["achat", "astream_chat"])
async def test_caller_pricing_is_used_for_compatible_async_requests(
    model, mode, monkeypatch
):
    forbidden = _guard_official_model_lookups(monkeypatch)
    pricing = _caller_pricing_fixture()
    outgoing = []
    streaming = mode == "astream_chat"
    response_body = text_response(model=model, usage=_COLLISION_USAGE)

    def handler(request):
        outgoing.append(request)
        if streaming:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content="\n".join(
                    text_sse_lines("Fixture reply", usage=_COLLISION_USAGE)
                ),
            )
        return httpx.Response(200, json=response_body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    _install_counted_httpx_client(monkeypatch, client, asynchronous=True)
    adapter = OpenAICompatibleAdapter(
        api_key=_COLLISION_KEY,
        model=model,
        base_url=_COLLISION_ROOT,
        transport="httpx",
        pricing=pricing,
    )
    try:
        if streaming:
            completed = []
            visible = [
                item async for item in adapter.astream_chat(
                    [UserMessage("hello")],
                    buffer_chars=len("Fixture reply") + 20,
                    on_done=completed.append,
                )
            ]
            assert visible == ["Fixture reply"]
            response = completed[0]
        else:
            response = await adapter.achat([UserMessage("hello")])
        assert len(outgoing) == 1
        assert str(outgoing[0].url) == _COLLISION_URL
        assert json.loads(outgoing[0].content)["model"] == model
        assert adapter.model_spec is None
        assert forbidden == []
        _assert_caller_pricing_response(response)
    finally:
        await client.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_caller_pricing_concurrent_async_responses_keep_cache_evidence_isolated(
    monkeypatch,
):
    pricing = Pricing.from_dict(
        [{"up_to_prompt_tokens": None, "input_per_1m": 10, "output_per_1m": 20}],
        currency="USD",
    )
    adapter = OpenAICompatibleAdapter(
        api_key=_COLLISION_KEY,
        model="vendor/concurrent-pricing",
        base_url=_COLLISION_ROOT,
        transport="httpx",
        pricing=pricing,
    )
    valid_started = asyncio.Event()
    invalid_finished = asyncio.Event()

    async def complete(*args, **kwargs):
        messages = kwargs["messages"]
        prompt = messages[-1]["content"]
        if prompt == "invalid-cache":
            await valid_started.wait()
            invalid_finished.set()
            usage = usage_payload(
                10,
                5,
                15,
                prompt_tokens_details={"cached_tokens": 11},
            )
        else:
            valid_started.set()
            await invalid_finished.wait()
            usage = usage_payload(10, 5, 15)
        return text_response("reply", model=adapter.model, usage=usage)

    monkeypatch.setattr(OpenAIAsyncClient, "complete", complete)
    assert adapter.pricing is not None
    valid_task = asyncio.create_task(adapter.achat([UserMessage("valid")]))
    invalid_task = asyncio.create_task(
        adapter.achat([UserMessage("invalid-cache")])
    )
    try:
        valid_response, invalid_response = await asyncio.wait_for(
            asyncio.gather(valid_task, invalid_task),
            timeout=2,
        )
    finally:
        for task in (valid_task, invalid_task):
            if not task.done():
                task.cancel()
        await asyncio.wait_for(
            asyncio.gather(valid_task, invalid_task, return_exceptions=True),
            timeout=2,
        )

    assert valid_response.usage.input_tokens == 10
    assert valid_response.cost_input == pytest.approx(100 / 1_000_000)
    assert valid_response.cost_output == pytest.approx(100 / 1_000_000)
    assert valid_response.cost_total == pytest.approx(200 / 1_000_000)
    assert invalid_response.usage.input_tokens == 10
    assert invalid_response.usage.cached_tokens is None
    assert invalid_response.cost_input is None
    assert invalid_response.cost_output == pytest.approx(100 / 1_000_000)
    assert invalid_response.cost_total is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_caller_pricing_interleaved_async_streams_keep_cache_evidence_local(
    monkeypatch,
):
    pricing = Pricing.from_dict(
        [{"up_to_prompt_tokens": None, "input_per_1m": 10, "output_per_1m": 20}],
        currency="USD",
    )
    adapter = OpenAICompatibleAdapter(
        api_key=_COLLISION_KEY,
        model="vendor/interleaved-pricing",
        base_url=_COLLISION_ROOT,
        transport="httpx",
        pricing=pricing,
    )
    valid_usage_seen = asyncio.Event()
    invalid_usage_seen = asyncio.Event()
    allow_valid_finish = asyncio.Event()
    allow_invalid_finish = asyncio.Event()
    valid_done = asyncio.Event()

    async def stream(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        is_valid = prompt == "valid-stream"
        usage = (
            usage_payload(10, 5, 15)
            if is_valid
            else usage_payload(
                10,
                5,
                15,
                prompt_tokens_details={"cached_tokens": 11},
            )
        )
        seen = valid_usage_seen if is_valid else invalid_usage_seen
        gate = allow_valid_finish if is_valid else allow_invalid_finish
        if not is_valid:
            await valid_usage_seen.wait()
        for event in _caller_pricing_sse_events(usage):
            yield event
            if isinstance(event.data, dict) and "usage" in event.data:
                seen.set()
                await gate.wait()

    monkeypatch.setattr(OpenAIAsyncClient, "stream", stream)
    assert adapter.pricing is not None
    valid_completed = []
    invalid_completed = []

    def mark_valid_done(response):
        valid_completed.append(response)
        valid_done.set()

    async def consume(prompt, completed, *, on_done=None):
        return [
            chunk async for chunk in adapter.astream_chat(
                [UserMessage(prompt)],
                buffer_chars=64,
                on_done=on_done or completed.append,
            )
        ]

    valid_task = asyncio.create_task(
        consume("valid-stream", valid_completed, on_done=mark_valid_done)
    )
    invalid_task = asyncio.create_task(
        consume("invalid-cache-stream", invalid_completed)
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(valid_usage_seen.wait(), invalid_usage_seen.wait()),
            timeout=2,
        )
        allow_valid_finish.set()
        await asyncio.wait_for(valid_done.wait(), timeout=2)
        allow_invalid_finish.set()
        valid_visible, invalid_visible = await asyncio.wait_for(
            asyncio.gather(valid_task, invalid_task),
            timeout=2,
        )
    finally:
        allow_valid_finish.set()
        allow_invalid_finish.set()
        for task in (valid_task, invalid_task):
            if not task.done():
                task.cancel()
        await asyncio.wait_for(
            asyncio.gather(valid_task, invalid_task, return_exceptions=True),
            timeout=2,
        )

    assert valid_visible == ["Fixture reply"]
    assert invalid_visible == ["Fixture reply"]
    assert len(valid_completed) == len(invalid_completed) == 1
    valid_response = valid_completed[0]
    invalid_response = invalid_completed[0]
    assert valid_response.cost_input == pytest.approx(100 / 1_000_000)
    assert valid_response.cost_output == pytest.approx(100 / 1_000_000)
    assert valid_response.cost_total == pytest.approx(200 / 1_000_000)
    assert invalid_response.usage.input_tokens == 10
    assert invalid_response.usage.cached_tokens is None
    assert invalid_response.cost_input is None
    assert invalid_response.cost_output == pytest.approx(100 / 1_000_000)
    assert invalid_response.cost_total is None


@pytest.mark.unit
@pytest.mark.parametrize("cache_rate", [None, 5])
def test_caller_pricing_contradictory_cache_usage_never_invents_input_cost(
    cache_rate,
):
    tier = {
        "up_to_prompt_tokens": None,
        "input_per_1m": 10,
        "output_per_1m": 20,
    }
    if cache_rate is not None:
        tier["cache_read_input_per_1m"] = cache_rate
    pricing = Pricing.from_dict([tier], currency="EUR")
    adapter = OpenAICompatibleAdapter(
        api_key=_COLLISION_KEY,
        model="vendor/contradictory-cache",
        base_url=_COLLISION_ROOT,
        pricing=pricing,
    )
    assert adapter.pricing is not None
    invalid_usage = usage_payload(
        10,
        5,
        15,
        prompt_tokens_details={"cached_tokens": 11},
    )
    with requests_mock.Mocker(real_http=False) as http:
        http.post(
            _COLLISION_URL,
            json=text_response(model=adapter.model, usage=invalid_usage),
        )
        response = adapter.chat([UserMessage("hello")])

    assert http.call_count == 1
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5
    assert response.usage.cached_tokens is None
    assert response.cost_input is None
    assert response.cost_output == pytest.approx(100 / 1_000_000)
    assert response.cost_total is None


_CALLER_PRICING_USAGE_CASES = (
    pytest.param(
        _METADATA_ABSENT, False,
        None, None, None, None, None,
        id="no-usage",
    ),
    pytest.param(
        usage_payload(completion_tokens=5, total_tokens=5), False,
        (None, 5, 5, None, None), None, None, None, None,
        id="missing-input",
    ),
    pytest.param(
        usage_payload(100, total_tokens=100), False,
        (100, None, 100, None, None), 1000 / 1_000_000, None, None, "EUR",
        id="missing-output",
    ),
    pytest.param(
        usage_payload(100, 5), False,
        (100, 5, None, None, None), 1000 / 1_000_000,
        100 / 1_000_000, 1100 / 1_000_000, "EUR",
        id="missing-total-with-components",
    ),
    pytest.param(
        usage_payload(0, 0, 0, prompt_tokens_details={"cached_tokens": 0}), False,
        (0, 0, 0, 0, None), 0, 0, 0, "EUR",
        id="reported-zero-counts",
    ),
    pytest.param(
        usage_payload(
            10, 5, 15, prompt_tokens_details={"cached_tokens": 11}
        ), False,
        (10, 5, 15, None, None), None, 100 / 1_000_000, None, "EUR",
        id="contradictory-cache-without-cache-tariff",
    ),
    pytest.param(
        usage_payload(
            10, 5, 15, prompt_tokens_details={"cached_tokens": 11}
        ), True,
        (10, 5, 15, None, None), None, 100 / 1_000_000, None, "EUR",
        id="contradictory-cache-with-cache-tariff",
    ),
    pytest.param(
        usage_payload(
            10, 5, 15, prompt_tokens_details={"cached_tokens": "bad"}
        ), False,
        (10, 5, 15, None, None), None, 100 / 1_000_000, None, "EUR",
        id="malformed-cache-without-cache-tariff",
    ),
    pytest.param(
        usage_payload(
            10, 5, 15, prompt_tokens_details={"cached_tokens": "bad"}
        ), True,
        (10, 5, 15, None, None), None, 100 / 1_000_000, None, "EUR",
        id="malformed-cache-with-cache-tariff",
    ),
    pytest.param(
        usage_payload(100, 5, 105, prompt_tokens_details={"cached_tokens": 20}),
        False,
        (100, 5, 105, 20, None), None, 100 / 1_000_000, None, "EUR",
        id="positive-cache-without-rate",
    ),
    pytest.param(
        usage_payload(100, 5, 105), True,
        (100, 5, 105, None, None), None, 100 / 1_000_000, None, "EUR",
        id="configured-cache-rate-without-count",
    ),
)


def _caller_pricing_sse_events(usage):
    decoder = SSEFrameDecoder()
    events = [
        event
        for line in text_sse_lines("Fixture reply", **(
            {} if usage is _METADATA_ABSENT else {"usage": usage}
        ))
        if (event := decoder.feed(line)) is not None
    ]
    final = decoder.finish()
    if final is not None:
        events.append(final)
    return events


async def _caller_pricing_async_events(events):
    for event in events:
        yield event


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("chat", "achat", "stream_chat", "astream_chat"))
@pytest.mark.parametrize(
    "usage,cache_rate,expected_usage,expected_input,expected_output,expected_total,currency",
    _CALLER_PRICING_USAGE_CASES,
)
async def test_caller_pricing_public_modes_preserve_partial_and_cache_usage(
    mode,
    usage,
    cache_rate,
    expected_usage,
    expected_input,
    expected_output,
    expected_total,
    currency,
    monkeypatch,
):
    tier = {
        "up_to_prompt_tokens": None,
        "input_per_1m": 10,
        "output_per_1m": 20,
    }
    if cache_rate:
        tier["cache_read_input_per_1m"] = 5
    pricing = Pricing.from_dict([tier], currency="EUR")
    adapter = OpenAICompatibleAdapter(
        api_key=_COLLISION_KEY,
        model="vendor/public-pricing-metadata",
        base_url=_COLLISION_ROOT,
        transport="httpx" if mode in ("achat", "astream_chat") else "requests",
        pricing=pricing,
    )
    assert adapter.pricing is not None
    response_body = text_response(
        model=adapter.model,
        **({} if usage is _METADATA_ABSENT else {"usage": usage}),
    )
    events = _caller_pricing_sse_events(usage)
    chunks = []
    completed = []
    streaming = mode in ("stream_chat", "astream_chat")
    if mode in ("chat", "stream_chat"):
        if streaming:
            monkeypatch.setattr(OpenAISyncClient, "stream", lambda self, **kwargs: iter(events))
            visible = list(adapter.stream_chat(
                [UserMessage("hello")],
                buffer_chars=len("Fixture reply") + 20,
                on_chunk=chunks.append,
                on_done=completed.append,
            ))
            assert visible == ["Fixture reply"]
            response = completed[0]
        else:
            monkeypatch.setattr(OpenAISyncClient, "complete", lambda self, **kwargs: response_body)
            response = adapter.chat([UserMessage("hello")])
    else:
        if streaming:
            monkeypatch.setattr(
                OpenAIAsyncClient,
                "stream",
                lambda self, **kwargs: _caller_pricing_async_events(events),
            )
            visible = [
                item async for item in adapter.astream_chat(
                    [UserMessage("hello")],
                    buffer_chars=len("Fixture reply") + 20,
                    on_chunk=chunks.append,
                    on_done=completed.append,
                )
            ]
            assert visible == ["Fixture reply"]
            response = completed[0]
        else:
            monkeypatch.setattr(
                OpenAIAsyncClient,
                "complete",
                AsyncMock(return_value=response_body),
            )
            response = await adapter.achat([UserMessage("hello")])

    if expected_usage is None:
        assert response.usage is None
    else:
        assert (
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.total_tokens,
            response.usage.cached_tokens,
            response.usage.cache_write_tokens,
        ) == expected_usage
    for actual, expected in (
        (response.cost_input, expected_input),
        (response.cost_output, expected_output),
        (response.cost_total, expected_total),
    ):
        if expected is None:
            assert actual is None
        elif expected == 0:
            assert actual == 0
        else:
            assert actual == pytest.approx(expected)
    assert response.currency == currency
    assert response.cost_breakdown is None
    if streaming:
        assert len(completed) == 1
        assert len(chunks) == 1
        if expected_usage is None:
            assert chunks[0].usage is None
        else:
            assert (
                chunks[0].usage.input_tokens,
                chunks[0].usage.output_tokens,
                chunks[0].usage.total_tokens,
                chunks[0].usage.cached_tokens,
                chunks[0].usage.cache_write_tokens,
            ) == expected_usage


@pytest.mark.unit
@pytest.mark.parametrize("transport", ("requests", "httpx"))
@pytest.mark.parametrize("mode", ("chat", "stream_chat"))
@pytest.mark.parametrize("kwargs,error_type", _INVALID_TOOL_REQUEST_CASES)
def test_compatible_sync_invalid_tool_input_fails_before_transport(
    transport, mode, kwargs, error_type, monkeypatch
):
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model="vendor/tool-validation",
        api_key=_COLLISION_KEY,
        base_url=_COLLISION_ROOT,
        **({"transport": "httpx"} if transport == "httpx" else {}),
    )
    if transport == "requests":
        with requests_mock.Mocker(real_http=False) as http:
            with pytest.raises(error_type):
                _invoke_compatible_sync(adapter, mode, kwargs)
            assert http.call_count == 0
    else:
        outgoing = []
        client = httpx.Client(transport=httpx.MockTransport(
            lambda request: _capture_httpx_request(request, outgoing)
        ))
        constructions = _install_counted_httpx_client(monkeypatch, client)
        with pytest.raises(error_type):
            _invoke_compatible_sync(adapter, mode, kwargs)
        assert outgoing == []
        assert constructions == []
        client.close()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("achat", "astream_chat"))
@pytest.mark.parametrize("kwargs,error_type", _INVALID_TOOL_REQUEST_CASES)
async def test_compatible_async_invalid_tool_input_fails_before_httpx_send(
    mode, kwargs, error_type, monkeypatch
):
    outgoing = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: _capture_httpx_request(request, outgoing)
    ))
    constructions = _install_counted_httpx_client(
        monkeypatch, client, asynchronous=True
    )
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model="vendor/tool-validation",
        api_key=_COLLISION_KEY,
        base_url=_COLLISION_ROOT,
    )
    with pytest.raises(error_type):
        await _invoke_compatible_async(adapter, mode, kwargs)
    assert outgoing == []
    assert constructions == []
    await client.aclose()


def _capture_httpx_request(request, outgoing, body=None, status=200):
    outgoing.append(request)
    return httpx.Response(status, json=text_response() if body is None else body)


def _install_counted_httpx_client(monkeypatch, client, *, asynchronous=False):
    constructions = []
    client_name = "AsyncClient" if asynchronous else "Client"

    def client_factory():
        constructions.append(True)
        return client

    monkeypatch.setattr(
        httpx, client_name, client_factory,
    )
    return constructions


def _assert_compatible_tool_request(body, model, choice="any"):
    expected_tool = {
        "type": "function",
        "function": {"name": "lookup", "parameters": _tool_spec().json_schema},
    }
    assert body["model"] == model
    assert body["tools"] == [expected_tool]
    expected_choice = {
        "auto": "auto",
        "none": "none",
        "any": "required",
        "lookup": {"type": "function", "function": {"name": "lookup"}},
    }[choice]
    assert body["tool_choice"] == expected_choice


def _malformed_tool_sse_lines():
    return premature_eof_sse_lines("pending") + tool_call_sse_lines(
        name="lookup",
        arguments_json='{"city":',
        call_id="call_bad_arguments",
    )


@pytest.mark.unit
@pytest.mark.parametrize("transport", ("requests", "httpx"))
def test_compatible_malformed_returned_tool_arguments_raise_after_receipt(
    transport, monkeypatch
):
    body = tool_call_response(
        name="lookup", arguments_json='{"city":', call_id="call_bad_arguments"
    )
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model="vendor/tool-validation",
        api_key=_COLLISION_KEY,
        base_url=_COLLISION_ROOT,
        **({"transport": "httpx"} if transport == "httpx" else {}),
    )
    if transport == "requests":
        with requests_mock.Mocker(real_http=False) as http:
            http.post(_COLLISION_URL, json=body)
            with pytest.raises(InvalidToolArgumentsError):
                adapter.chat(
                    [UserMessage(content="Find a city")],
                    tools=[_tool_spec()], tool_choice="any",
                )
            assert http.call_count == 1
            assert http.request_history[0].url == _COLLISION_URL
            _assert_compatible_tool_request(
                http.request_history[0].json(), "vendor/tool-validation"
            )
    else:
        outgoing = []
        client = httpx.Client(transport=httpx.MockTransport(
            lambda request: _capture_httpx_request(request, outgoing, body)
        ))
        constructions = _install_counted_httpx_client(monkeypatch, client)
        with pytest.raises(InvalidToolArgumentsError):
            adapter.chat(
                [UserMessage(content="Find a city")],
                tools=[_tool_spec()], tool_choice="any",
            )
        assert len(outgoing) == 1
        assert constructions == [True]
        assert str(outgoing[0].url) == _COLLISION_URL
        _assert_compatible_tool_request(
            json.loads(outgoing[0].content), "vendor/tool-validation"
        )
        client.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compatible_async_malformed_returned_tool_arguments_raise_after_receipt(
    monkeypatch,
):
    outgoing = []
    body = tool_call_response(
        name="lookup", arguments_json='{"city":', call_id="call_bad_arguments"
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: _capture_httpx_request(request, outgoing, body)
    ))
    constructions = _install_counted_httpx_client(
        monkeypatch, client, asynchronous=True
    )
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model="vendor/tool-validation",
        api_key=_COLLISION_KEY,
        base_url=_COLLISION_ROOT,
    )
    with pytest.raises(InvalidToolArgumentsError):
        await adapter.achat(
            [UserMessage(content="Find a city")],
            tools=[_tool_spec()], tool_choice="any",
        )
    assert len(outgoing) == 1
    assert constructions == [True]
    assert str(outgoing[0].url) == _COLLISION_URL
    _assert_compatible_tool_request(
        json.loads(outgoing[0].content), "vendor/tool-validation"
    )
    await client.aclose()


@pytest.mark.unit
@pytest.mark.parametrize("transport", ("requests", "httpx"))
def test_compatible_sync_stream_malformed_tool_arguments_fail_without_success(
    transport, monkeypatch
):
    adapter, outgoing, resource, client_closes, client = _counted_sync_stream_route(
        transport, _malformed_tool_sse_lines(), monkeypatch, "vendor/tool-validation"
    )
    constructions = (
        _install_counted_httpx_client(monkeypatch, client)
        if transport == "httpx" else []
    )
    visible, tools, completed = [], [], []
    try:
        with pytest.raises(InvalidToolArgumentsError):
            visible.extend(adapter.stream_chat(
                [UserMessage(content="Find a city")],
                tools=[_tool_spec()], tool_choice="any", buffer_chars=64,
                on_tool_call=tools.append, on_done=completed.append,
            ))
        assert visible == []
        assert tools == []
        assert completed == []
        assert len(outgoing) == 1
        if transport == "requests":
            assert outgoing[0][0] == _COLLISION_URL
        else:
            assert str(outgoing[0].url) == _COLLISION_URL
        request_body = (
            outgoing[0][1]["json"]
            if transport == "requests"
            else json.loads(outgoing[0].content)
        )
        _assert_compatible_tool_request(request_body, "vendor/tool-validation")
        assert resource.close_count == 1
        if transport == "httpx":
            assert constructions == [True]
            assert client_closes == [True]
    finally:
        if resource.close_count == 0:
            resource.close()
        if transport == "httpx" and not client_closes:
            client.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compatible_async_stream_malformed_tool_arguments_fail_without_success(
    monkeypatch,
):
    adapter, outgoing, resource, client_closes, client = _counted_async_stream_route(
        _malformed_tool_sse_lines(), monkeypatch, "vendor/tool-validation"
    )
    constructions = _install_counted_httpx_client(
        monkeypatch, client, asynchronous=True
    )
    visible, tools, completed = [], [], []
    try:
        with pytest.raises(InvalidToolArgumentsError):
            async for text in adapter.astream_chat(
                [UserMessage(content="Find a city")],
                tools=[_tool_spec()], tool_choice="any", buffer_chars=64,
                on_tool_call=tools.append, on_done=completed.append,
            ):
                visible.append(text)
        assert visible == []
        assert tools == []
        assert completed == []
        assert len(outgoing) == 1
        assert str(outgoing[0].url) == _COLLISION_URL
        _assert_compatible_tool_request(
            json.loads(outgoing[0].content), "vendor/tool-validation"
        )
        assert resource.close_count == 1
        assert constructions == [True]
        assert client_closes == [True]
    finally:
        if resource.close_count == 0:
            await resource.aclose()
        if not client_closes:
            await client.aclose()


@pytest.mark.unit
@pytest.mark.parametrize("transport", ("requests", "httpx"))
@pytest.mark.parametrize("mode", ("chat", "stream_chat"))
def test_compatible_tool_server_rejection_is_single_request_without_fallback(
    transport, mode, monkeypatch
):
    outgoing = []
    model = "vendor/tool-validation"
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model=model,
        api_key=_COLLISION_KEY,
        base_url=_COLLISION_ROOT,
        **({"transport": "httpx"} if transport == "httpx" else {}),
    )
    if transport == "requests":
        with requests_mock.Mocker(real_http=False) as http:
            http.post(
                _COLLISION_URL, status_code=400,
                json=_failure_payload("unsupported-feature"),
            )
            completed = []
            with pytest.raises(LLMAPIClientError):
                if mode == "chat":
                    adapter.chat(
                        [UserMessage(content="Find a city")],
                        tools=[_tool_spec()], tool_choice="any",
                    )
                else:
                    list(adapter.stream_chat(
                        [UserMessage(content="Find a city")],
                        tools=[_tool_spec()], tool_choice="any",
                        on_done=completed.append,
                    ))
            assert http.call_count == 1
            body = http.request_history[0].json()
            assert http.request_history[0].url == _COLLISION_URL
    else:
        client = httpx.Client(transport=httpx.MockTransport(
            lambda request: _capture_httpx_request(
                request, outgoing, _failure_payload("unsupported-feature"), 400
            )
        ))
        constructions = _install_counted_httpx_client(monkeypatch, client)
        completed = []
        with pytest.raises(LLMAPIClientError):
            if mode == "chat":
                adapter.chat(
                    [UserMessage(content="Find a city")],
                    tools=[_tool_spec()], tool_choice="any",
                )
            else:
                list(adapter.stream_chat(
                    [UserMessage(content="Find a city")],
                    tools=[_tool_spec()], tool_choice="any",
                    on_done=completed.append,
                ))
        assert len(outgoing) == 1
        assert constructions == [True]
        assert str(outgoing[0].url) == _COLLISION_URL
        body = json.loads(outgoing[0].content)
        client.close()
    _assert_compatible_tool_request(body, model)
    assert completed == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("achat", "astream_chat"))
async def test_compatible_async_tool_server_rejection_is_single_request_without_fallback(
    mode, monkeypatch,
):
    outgoing = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: _capture_httpx_request(
            request, outgoing, _failure_payload("unsupported-feature"), 400
        )
    ))
    constructions = _install_counted_httpx_client(
        monkeypatch, client, asynchronous=True
    )
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model="vendor/tool-validation",
        api_key=_COLLISION_KEY,
        base_url=_COLLISION_ROOT,
    )
    completed = []
    with pytest.raises(LLMAPIClientError):
        if mode == "achat":
            await adapter.achat(
                [UserMessage(content="Find a city")],
                tools=[_tool_spec()], tool_choice="any",
            )
        else:
            async for _text in adapter.astream_chat(
                [UserMessage(content="Find a city")],
                tools=[_tool_spec()], tool_choice="any",
                on_done=completed.append,
            ):
                pass
    assert len(outgoing) == 1
    assert constructions == [True]
    assert str(outgoing[0].url) == _COLLISION_URL
    body = json.loads(outgoing[0].content)
    _assert_compatible_tool_request(body, "vendor/tool-validation")
    assert completed == []
    await client.aclose()


@pytest.mark.unit
@pytest.mark.parametrize("transport", ("requests", "httpx"))
@pytest.mark.parametrize("choice", ("auto", "none", "any", "lookup"))
def test_compatible_sync_stream_tool_choice_payload_and_completion(
    transport, choice, monkeypatch
):
    model = "vendor/tool-choice-stream"
    lines = (
        text_sse_lines("No tool", fragments=("No ", "tool"), usage=usage_payload(2, 2, 4))
        if choice == "none"
        else _fragmented_text_and_tool_sse_lines()
    )
    adapter, outgoing, resource, client_closes, client = _counted_sync_stream_route(
        transport, lines, monkeypatch, model
    )
    completed = []
    try:
        visible = list(adapter.stream_chat(
            [UserMessage(content="Find a city")],
            tools=[_tool_spec()],
            tool_choice=choice,
            on_done=completed.append,
        ))
        assert len(outgoing) == 1
        if transport == "requests":
            assert outgoing[0][0] == _COLLISION_URL
        else:
            assert str(outgoing[0].url) == _COLLISION_URL
        body = (
            outgoing[0][1]["json"]
            if transport == "requests"
            else json.loads(outgoing[0].content)
        )
        _assert_compatible_tool_request(body, model, choice)
        assert resource.close_count == 1
        if transport == "httpx":
            assert client_closes == [True]
        assert len(completed) == 1
        if choice == "none":
            assert visible == ["No ", "tool"]
            assert not completed[0].tool_calls
        else:
            assert len(completed[0].tool_calls) == 1
            assert completed[0].tool_calls[0].name == "lookup"
            assert completed[0].tool_calls[0].arguments == {"city": "Haifa"}
            assert completed[0].tool_calls[0].call_id == "call_fragmented_1"
    finally:
        if resource.close_count == 0:
            resource.close()
        if transport == "httpx" and not client_closes:
            client.close()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ("auto", "none", "any", "lookup"))
async def test_compatible_async_stream_tool_choice_payload_and_completion(
    choice, monkeypatch
):
    model = "vendor/tool-choice-stream"
    lines = (
        text_sse_lines("No tool", fragments=("No ", "tool"), usage=usage_payload(2, 2, 4))
        if choice == "none"
        else _fragmented_text_and_tool_sse_lines()
    )
    adapter, outgoing, resource, client_closes, client = (
        _counted_async_stream_route(lines, monkeypatch, model)
    )
    completed = []
    try:
        visible = [text async for text in adapter.astream_chat(
            [UserMessage(content="Find a city")],
            tools=[_tool_spec()],
            tool_choice=choice,
            on_done=completed.append,
        )]
        assert len(outgoing) == 1
        assert str(outgoing[0].url) == _COLLISION_URL
        body = json.loads(outgoing[0].content)
        _assert_compatible_tool_request(body, model, choice)
        assert resource.close_count == 1
        assert client_closes == [True]
        assert len(completed) == 1
        if choice == "none":
            assert visible == ["No ", "tool"]
            assert not completed[0].tool_calls
        else:
            assert len(completed[0].tool_calls) == 1
            assert completed[0].tool_calls[0].name == "lookup"
            assert completed[0].tool_calls[0].arguments == {"city": "Haifa"}
            assert completed[0].tool_calls[0].call_id == "call_fragmented_1"
    finally:
        if resource.close_count == 0:
            await resource.aclose()
        if not client_closes:
            await client.aclose()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("eof", "error"))
async def test_compatible_async_stream_without_terminal_outcome_is_not_success(
    outcome, monkeypatch
):
    model = "vendor/unregistered-model"
    lines = premature_eof_sse_lines("pending")
    if outcome == "error":
        lines += event_lines({"type": "error", "error": {"message": "stream rejected"}})
    completed = []
    visible = []
    adapter, requests_seen, stream_resource, client_closes, client = (
        _counted_async_stream_route(lines, monkeypatch, model)
    )
    caught = None
    try:
        try:
            async for text in adapter.astream_chat(
                [UserMessage(content="Hello")], buffer_chars=64,
                on_done=completed.append,
            ):
                visible.append(text)
        except LLMAPIClientError as error:
            caught = error

        assert len(requests_seen) == 1
        assert str(requests_seen[0].url) == _COLLISION_URL
        stream_close_count = stream_resource.close_count
        client_close_count = len(client_closes)
        assert stream_close_count == 1
        assert client_close_count == 1
        assert isinstance(caught, LLMAPIClientError) and not visible and not completed, (
            f"terminal outcome={caught!r}; visible={visible!r}; done={completed!r}"
        )
        if outcome == "error":
            assert "stream rejected" in str(caught)
    finally:
        if stream_resource.close_count == 0:
            await stream_resource.aclose()
        if not client_closes:
            await client.aclose()


@pytest.mark.unit
@pytest.mark.parametrize("transport", ("requests", "httpx"))
def test_compatible_sync_transport_early_close_releases_owned_resources(
    transport, monkeypatch
):
    model = "vendor/unregistered-model"
    lines = text_sse_lines(
        "Hello!", fragments=("He", "llo!"), usage=usage_payload(2, 3, 5)
    )
    completed = []
    adapter, requests_seen, stream_resource, client_closes, client = (
        _counted_sync_stream_route(transport, lines, monkeypatch, model)
    )

    stream = adapter.stream_chat(
        [UserMessage(content="Hello")], buffer_chars=4, on_done=completed.append
    )
    try:
        assert next(stream) == "Hell"
        stream.close()
        if transport == "requests":
            assert stream_resource.close_count == 1
            assert requests_seen[0][0] == _COLLISION_URL
        else:
            assert stream_resource.close_count == 1
            assert client_closes == [True]
            assert str(requests_seen[0].url) == _COLLISION_URL
        assert len(requests_seen) == 1
        assert completed == []
    finally:
        stream.close()
        if stream_resource.close_count == 0:
            stream_resource.close()
        if transport == "httpx" and not client_closes:
            client.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compatible_async_cancellation_releases_httpx_stream_once(monkeypatch):
    model = "vendor/unregistered-model"
    lines = premature_eof_sse_lines(
        "Hello!", content_fragments=("He", "llo!")
    )
    suspended = asyncio.Event()
    release = asyncio.Event()
    first_chunk = asyncio.Event()
    completed = []
    adapter, requests_seen, stream_bytes, client_closes, client = (
        _counted_async_stream_route(
            lines, monkeypatch, model, suspended=suspended, release=release
        )
    )

    async def consume():
        async for _text in adapter.astream_chat(
            [UserMessage(content="Hello")], buffer_chars=4,
            on_done=completed.append,
        ):
            first_chunk.set()

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(first_chunk.wait(), timeout=2)
        await asyncio.wait_for(suspended.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)

        assert len(requests_seen) == 1
        assert str(requests_seen[0].url) == _COLLISION_URL
        assert stream_bytes.close_count == 1
        assert client_closes == [True]
        assert completed == []
    finally:
        release.set()
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if stream_bytes.close_count == 0:
            await stream_bytes.aclose()
        if not client_closes:
            await client.aclose()


def _assert_t018_public_result(adapter, mode):
    if mode == "stream_chat":
        completed = []
        visible = list(
            adapter.stream_chat(
                messages=[UserMessage(content="Say hello.")],
                max_tokens=17,
                temperature=0.4,
                top_p=0.8,
                on_done=completed.append,
            )
        )
        return visible, completed
    response = adapter.chat(
        messages=[UserMessage(content="Say hello.")],
        max_tokens=17,
        temperature=0.4,
        top_p=0.8,
    )
    return response, []


_T018_SYNC_ROUTES = (
    ("chat", "requests"),
    ("stream_chat", "requests"),
    ("chat", "httpx"),
    ("stream_chat", "httpx"),
)


@pytest.mark.unit
@pytest.mark.parametrize("mode,transport", _T018_SYNC_ROUTES)
def test_compatible_sync_modes_preserve_transport_and_normalized_result(
    mode, transport, monkeypatch
):
    model = "vendor/unregistered-model"
    streaming = mode == "stream_chat"
    forbidden = _guard_official_model_lookups(monkeypatch)
    requests_seen = []

    if transport == "requests":
        def forbidden_httpx_client(*_args, **_kwargs):
            raise AssertionError("default requests route constructed HTTPX Client")

        monkeypatch.setattr(httpx, "Client", forbidden_httpx_client)
        with requests_mock.Mocker(real_http=False) as http:
            if streaming:
                http.post(
                    _COLLISION_URL,
                    text="\n".join(_collision_sse_lines(model)),
                    headers={"Content-Type": "text/event-stream"},
                )
            else:
                http.post(
                    _COLLISION_URL,
                    json=text_response(model=model, usage=_COLLISION_USAGE),
                )
            adapter = UniversalLLMAPIAdapter(
                organization="openai_compatible",
                model=model,
                api_key=_COLLISION_KEY,
                base_url=f"{_COLLISION_ROOT}/",
            )
            result, completed = _assert_t018_public_result(adapter, mode)
            if streaming:
                visible = result
                assert visible == ["Fixture reply"]
                assert len(completed) == 1
                _assert_collision_response(completed[0], model)
            else:
                _assert_collision_response(result, model)
            assert http.call_count == 1
            request = http.request_history[0]
            assert request.method == "POST"
            assert request.url == _COLLISION_URL
            assert request.headers["Authorization"] == f"Bearer {_COLLISION_KEY}"
            assert request.headers["Content-Type"] == "application/json"
            assert request.json() == _collision_body(model, streaming=streaming)
            assert forbidden == []
        return

    real_client = httpx.Client

    def handler(request):
        requests_seen.append(request)
        if streaming:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content="\n".join(_collision_sse_lines(model)),
            )
        return httpx.Response(
            200,
            json=text_response(model=model, usage=_COLLISION_USAGE),
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda: real_client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("explicit HTTPX route used requests")
        ),
    )
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model=model,
        api_key=_COLLISION_KEY,
        base_url=f"{_COLLISION_ROOT}/",
        transport="httpx",
    )
    result, completed = _assert_t018_public_result(adapter, mode)
    if streaming:
        visible = result
        assert visible == ["Fixture reply"]
        assert len(completed) == 1
        _assert_collision_response(completed[0], model)
    else:
        _assert_collision_response(result, model)
    assert len(requests_seen) == 1
    request = requests_seen[0]
    assert request.method == "POST"
    assert str(request.url) == _COLLISION_URL
    assert request.headers["Authorization"] == f"Bearer {_COLLISION_KEY}"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == _collision_body(model, streaming=streaming)
    assert forbidden == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("achat", "astream_chat"))
async def test_compatible_async_modes_preserve_httpx_route_and_normalized_result(
    mode, monkeypatch
):
    model = "vendor/unregistered-model"
    streaming = mode == "astream_chat"
    forbidden = _guard_official_model_lookups(monkeypatch)
    requests_seen = []
    real_async_client = httpx.AsyncClient

    def handler(request):
        requests_seen.append(request)
        if streaming:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content="\n".join(_collision_sse_lines(model)),
            )
        return httpx.Response(
            200,
            json=text_response(model=model, usage=_COLLISION_USAGE),
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda: real_async_client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(
        requests,
        "post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("async HTTPX route used requests")
        ),
    )
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model=model,
        api_key=_COLLISION_KEY,
        base_url=f"{_COLLISION_ROOT}/",
    )
    if streaming:
        completed = []
        visible = [
            text
            async for text in adapter.astream_chat(
                messages=[UserMessage(content="Say hello.")],
                max_tokens=17,
                temperature=0.4,
                top_p=0.8,
                on_done=completed.append,
            )
        ]
        assert visible == ["Fixture reply"]
        assert len(completed) == 1
        _assert_collision_response(completed[0], model)
    else:
        _assert_collision_response(
            await adapter.achat(
                messages=[UserMessage(content="Say hello.")],
                max_tokens=17,
                temperature=0.4,
                top_p=0.8,
            ),
            model,
        )
    assert len(requests_seen) == 1
    request = requests_seen[0]
    assert request.method == "POST"
    assert str(request.url) == _COLLISION_URL
    assert request.headers["Authorization"] == f"Bearer {_COLLISION_KEY}"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == _collision_body(model, streaming=streaming)
    assert forbidden == []


def _block_httpx_import(monkeypatch):
    import builtins

    real_import = builtins.__import__
    attempts = []

    def missing_httpx(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("httpx is missing")
        return real_import(name, *args, **kwargs)

    def forbidden(name):
        def fail(*_args, **_kwargs):
            attempts.append(name)
            raise AssertionError(f"unavailable HTTPX route attempted {name}")

        return fail

    monkeypatch.setattr(builtins, "__import__", missing_httpx)
    monkeypatch.setattr(httpx, "Client", forbidden("httpx.Client"))
    monkeypatch.setattr(httpx, "AsyncClient", forbidden("httpx.AsyncClient"))
    monkeypatch.setattr(socket, "getaddrinfo", forbidden("socket.getaddrinfo"))
    return attempts


@pytest.mark.unit
@pytest.mark.parametrize("mode", ("chat", "stream_chat"))
def test_compatible_default_requests_modes_work_without_httpx(mode, monkeypatch):
    model = "vendor/unregistered-model"
    with requests_mock.Mocker(real_http=False) as http:
        http.post(
            _COLLISION_URL,
            text="\n".join(_collision_sse_lines(model))
            if mode == "stream_chat"
            else None,
            json=None
            if mode == "stream_chat"
            else text_response(model=model, usage=_COLLISION_USAGE),
            headers={"Content-Type": "text/event-stream"}
            if mode == "stream_chat"
            else {"Content-Type": "application/json"},
        )
        default = UniversalLLMAPIAdapter(
            organization="openai_compatible",
            model=model,
            api_key=_COLLISION_KEY,
            base_url=_COLLISION_ROOT,
        )
        attempts = []
        with pytest.MonkeyPatch.context() as scoped:
            attempts = _block_httpx_import(scoped)
            result, completed = _assert_t018_public_result(default, mode)
        if mode == "chat":
            _assert_collision_response(result, model)
        else:
            assert result == ["Fixture reply"]
            assert len(completed) == 1
            _assert_collision_response(completed[0], model)
        assert attempts == []
        assert http.call_count == 1
        request = http.request_history[0]
        assert request.method == "POST"
        assert request.url == _COLLISION_URL
        assert request.headers["Authorization"] == f"Bearer {_COLLISION_KEY}"
        assert request.headers["Content-Type"] == "application/json"
        assert request.json() == _collision_body(model, streaming=mode == "stream_chat")


@pytest.mark.unit
@pytest.mark.parametrize("mode", ("chat", "stream_chat"))
@pytest.mark.parametrize("capture_reasoning", (False, True))
@pytest.mark.parametrize(
    "model,reasoning_level,warning_expected",
    (
        ("vendor/unregistered-model", None, False),
        ("vendor/unregistered-model", "high", True),
        ("gpt-5", 1024, True),
    ),
)
def test_compatible_unknown_reasoning_controls_are_omitted_and_never_invented(
    mode, capture_reasoning, model, reasoning_level, warning_expected, monkeypatch
):
    forbidden = _guard_official_model_lookups(monkeypatch)
    url = f"{_COLLISION_ROOT}/chat/completions"
    streaming = mode == "stream_chat"
    done = []
    reasoning = []

    response_list = []
    if streaming:
        response_list.append({
            "text": "\n".join(_collision_sse_lines(model)),
            "headers": {"Content-Type": "text/event-stream"},
        })
    else:
        response_list.append({
            "json": text_response(model=model, usage=_COLLISION_USAGE),
        })
    with requests_mock.Mocker(real_http=False) as http:
        http.post(url, response_list=response_list)
        adapter = UniversalLLMAPIAdapter(
            organization="openai_compatible", model=model,
            api_key=_COLLISION_KEY, base_url=_COLLISION_ROOT,
        )
        kwargs = dict(
            messages=[UserMessage(content="Say hello.")],
            capture_reasoning=capture_reasoning,
        )
        if streaming:
            kwargs["on_reasoning"] = reasoning.append
        if reasoning_level is not None:
            kwargs["reasoning_level"] = reasoning_level
        if streaming:
            kwargs["on_done"] = done.append
        with warnings.catch_warnings(record=True) as emitted_warnings:
            warnings.simplefilter("always")
            result = list(adapter.stream_chat(**kwargs)) if streaming else adapter.chat(**kwargs)
        if warning_expected:
            assert len(emitted_warnings) == 1
            assert emitted_warnings[0].category is UserWarning
            assert "not verified" in str(emitted_warnings[0].message)
            assert "reasoning is disabled" in str(emitted_warnings[0].message)
        else:
            assert emitted_warnings == []
        assert http.call_count == 1
        request = http.request_history[0]
        body = request.json()

    assert request.url == url
    assert body["model"] == model
    assert not {"reasoning_effort", "reasoning", "thinking", "capture_reasoning"} & body.keys()
    assert forbidden == []
    if streaming:
        assert result == ["Fixture reply"]
        assert len(done) == 1
        response = done[0]
    else:
        response = result
    assert response.content == "Fixture reply"
    assert response.reasoning_events == []
    assert reasoning == []
    assert adapter.adapter.model_spec is None
    assert adapter.adapter.pricing is None


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("achat", "astream_chat"))
@pytest.mark.parametrize("capture_reasoning", (False, True))
@pytest.mark.parametrize(
    "model,reasoning_level,warning_expected",
    (
        ("vendor/unregistered-model", None, False),
        ("vendor/unregistered-model", "high", True),
        ("gpt-5", 1024, True),
    ),
)
async def test_compatible_async_unknown_reasoning_controls_are_omitted_and_never_invented(
    mode, capture_reasoning, model, reasoning_level, warning_expected, monkeypatch
):
    forbidden = _guard_official_model_lookups(monkeypatch)
    url = f"{_COLLISION_ROOT}/chat/completions"
    streaming = mode == "astream_chat"
    outgoing = []
    done = []
    reasoning = []

    def handler(request):
        outgoing.append(request)
        if streaming:
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content="\n".join(_collision_sse_lines(model)),
            )
        return httpx.Response(
            200, json=text_response(model=model, usage=_COLLISION_USAGE)
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    _install_counted_httpx_client(monkeypatch, client, asynchronous=True)
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible", model=model,
        api_key=_COLLISION_KEY, base_url=_COLLISION_ROOT,
    )
    kwargs = dict(
        messages=[UserMessage(content="Say hello.")],
        capture_reasoning=capture_reasoning,
    )
    if streaming:
        kwargs["on_reasoning"] = reasoning.append
    if reasoning_level is not None:
        kwargs["reasoning_level"] = reasoning_level
    if streaming:
        kwargs["on_done"] = done.append
    try:
        with warnings.catch_warnings(record=True) as emitted_warnings:
            warnings.simplefilter("always")
            result = (
                [text async for text in adapter.astream_chat(**kwargs)]
                if streaming else await adapter.achat(**kwargs)
            )
    finally:
        await client.aclose()
    if warning_expected:
        assert len(emitted_warnings) == 1
        assert emitted_warnings[0].category is UserWarning
        assert "not verified" in str(emitted_warnings[0].message)
        assert "reasoning is disabled" in str(emitted_warnings[0].message)
    else:
        assert emitted_warnings == []
    assert len(outgoing) == 1
    request = outgoing[0]
    body = json.loads(request.content)
    assert str(request.url) == url
    assert body["model"] == model
    assert not {"reasoning_effort", "reasoning", "thinking", "capture_reasoning"} & body.keys()
    assert forbidden == []
    if streaming:
        assert result == ["Fixture reply"]
        assert len(done) == 1
        response = done[0]
    else:
        response = result
    assert response.content == "Fixture reply"
    assert response.reasoning_events == []
    assert reasoning == []
    assert adapter.adapter.model_spec is None
    assert adapter.adapter.pricing is None


@pytest.mark.unit
def test_compatible_base_install_sync_requests_chat_and_stream_subprocess_smoke():
    repo_root = Path(__file__).resolve().parents[3]
    src_path = repo_root / "src"
    assert src_path.is_dir()
    child = r'''import importlib.abc, json, socket, sys
from pathlib import Path
src = Path(sys.argv[1])
assert src.is_dir()
sys.path.insert(0, str(src))
blocked = (
    "httpx", "llm_api_adapter_mistral", "llm_api_adapter_xai",
    "llm_api_adapter_qwen", "llm_api_adapter_kimi",
    "llm_api_adapter_deepseek", "llm_api_adapter_zai",
    "openai", "anthropic", "google", "mistralai", "xai", "qwen", "pydantic",
    "kimi", "deepseek", "zai",
)
attempts = []
class BlockOptionalModules(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == root or fullname.startswith(root + ".") for root in blocked):
            attempts.append(fullname)
            raise ModuleNotFoundError("optional dependency blocked: " + fullname)
sys.meta_path.insert(0, BlockOptionalModules())
def blocked_network(*args, **kwargs):
    raise AssertionError("real network access attempted")
socket.getaddrinfo = blocked_network
socket.socket.connect = blocked_network
socket.create_connection = blocked_network
import requests
from unittest.mock import patch
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.organization_registry import OrganizationPluginDiscovery
import llm_api_adapter.adapters.base_adapter as base_adapter
import llm_api_adapter.llms.openai.sync_client as openai_sync
input_data = json.load(sys.stdin)
assert Path(universal_module.__file__).resolve().is_relative_to(src.resolve())
calls = []
boundary_attempts = []
def forbidden_boundary(name):
    def fail(*args, **kwargs):
        boundary_attempts.append(name)
        raise AssertionError("compatible mode crossed boundary: " + name)
    return fail
OrganizationPluginDiscovery.discover = forbidden_boundary("plugin discovery")
base_adapter.resolve_model_spec = forbidden_boundary("profile lookup")
for symbol in ("model_api_variant", "model_spec_for", "apply_model_request_rules"):
    setattr(openai_sync, symbol, forbidden_boundary(symbol))
class Response:
    status_code = 200
    headers = {"Content-Type": "application/json"}
    def __init__(self, content, stream=False):
        self.content = content.encode()
        self.text = content
        self.stream = stream
    def raise_for_status(self): pass
    def json(self): return json.loads(self.text)
    def iter_lines(self, **kwargs): return iter(self.content.splitlines())
    def close(self): pass
def fake_post(url, **kwargs):
    calls.append((url, kwargs))
    if kwargs.get("stream"):
        return Response(input_data["sse"], True)
    return Response(json.dumps(input_data["json"]))
with patch.object(requests, "post", fake_post):
    adapter = UniversalLLMAPIAdapter(organization="openai_compatible", model="vendor/subprocess-smoke", api_key="synthetic", base_url="https://example.invalid/v1")
    response = adapter.chat([{"role": "user", "content": "hello"}])
    done = []
    visible = list(adapter.stream_chat([{"role": "user", "content": "hello"}], on_done=done.append))
assert response.content == "Subprocess reply"
assert visible == ["Subprocess reply"] and len(done) == 1 and done[0].content == response.content
assert len(calls) == 2 and all(call[0] == "https://example.invalid/v1/chat/completions" for call in calls)
assert all(call[1]["json"]["model"] == "vendor/subprocess-smoke" for call in calls)
assert adapter.adapter.model_spec is None and adapter.adapter.pricing is None
assert not any(
    name == root or name.startswith(root + ".")
    for name in sys.modules
    for root in blocked
)
assert attempts == []
assert boundary_attempts == []
print("compatible requests isolation smoke passed")
    '''
    payload = {
        "json": text_response("Subprocess reply", model="vendor/subprocess-smoke"),
        "sse": "\n".join(text_sse_lines("Subprocess reply")),
    }
    completed = subprocess.run(
        [sys.executable, "-I", "-c", child, str(src_path)],
        input=json.dumps(payload), text=True, capture_output=True, timeout=15,
        cwd=repo_root,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "compatible requests isolation smoke passed"


@pytest.mark.unit
@pytest.mark.parametrize("mode", ("chat", "stream_chat"))
def test_compatible_explicit_sync_httpx_requires_extra(mode, monkeypatch):
    model = "vendor/unregistered-model"
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model=model,
        api_key=_COLLISION_KEY,
        base_url=_COLLISION_ROOT,
        transport="httpx",
    )
    with requests_mock.Mocker(real_http=False) as http:
        with pytest.MonkeyPatch.context() as scoped:
            attempts = _block_httpx_import(scoped)
            with pytest.raises(ImportError, match=r"llm-api-adapter\[httpx\]"):
                if mode == "chat":
                    adapter.chat([UserMessage(content="Say hello.")])
                else:
                    list(adapter.stream_chat([UserMessage(content="Say hello.")]))
        assert attempts == []
        assert http.call_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("achat", "astream_chat"))
async def test_compatible_async_httpx_requires_extra(mode, monkeypatch):
    model = "vendor/unregistered-model"
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model=model,
        api_key=_COLLISION_KEY,
        base_url=_COLLISION_ROOT,
    )
    with requests_mock.Mocker(real_http=False) as http:
        with pytest.MonkeyPatch.context() as scoped:
            attempts = _block_httpx_import(scoped)
            with pytest.raises(ImportError, match=r"llm-api-adapter\[async\]"):
                if mode == "achat":
                    await adapter.achat([UserMessage(content="Say hello.")])
                else:
                    async for _text in adapter.astream_chat(
                        [UserMessage(content="Say hello.")]
                    ):
                        pass
        assert attempts == []
        assert http.call_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("first_compatible_fails", (False, True))
async def test_compatible_and_official_async_instances_keep_roots_keys_and_profiles_isolated(
    first_compatible_fails, monkeypatch
):
    forbidden = []
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        _forbid_official_lookup("socket.getaddrinfo", forbidden),
    )
    monkeypatch.setattr(
        OrganizationPluginDiscovery,
        "discover",
        _forbid_official_lookup("OrganizationPluginDiscovery.discover", forbidden),
    )
    real_async_client = httpx.AsyncClient
    attempts = []

    def handler(request):
        attempts.append(request)
        if str(request.url).endswith("/v1/responses"):
            return httpx.Response(200, json=_native_gpt5_response())
        is_first_compatible = sum(
            str(item.url) == _COLLISION_URL for item in attempts
        ) == 1
        if first_compatible_fails and is_first_compatible:
            return httpx.Response(400, json=_failure_payload("unsupported-feature"))
        return httpx.Response(
            200,
            json=text_response(model="gpt-5", usage=_COLLISION_USAGE),
        )

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda: real_async_client(transport=transport),
    )
    compatible = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model="gpt-5",
        api_key=_COLLISION_KEY,
        base_url=f"{_COLLISION_ROOT}/",
    )
    assert compatible.adapter.model_spec is None
    assert compatible.adapter.pricing is None
    if first_compatible_fails:
        with pytest.raises(LLMAPIClientError):
            await compatible.achat(
                [UserMessage(content="Say hello.")],
                max_tokens=17,
                temperature=0.4,
                top_p=0.8,
            )
    else:
        _assert_collision_response(
            await compatible.achat(
                [UserMessage(content="Say hello.")],
                max_tokens=17,
                temperature=0.4,
                top_p=0.8,
            ),
            "gpt-5",
        )

    official = UniversalLLMAPIAdapter(
        organization="openai",
        model="gpt-5",
        api_key="synthetic-official-key",
    )
    assert official.adapter.model_spec is not None
    assert official.adapter.pricing is not None
    official_response = await official.achat(
        [UserMessage(content="Say hello.")], max_tokens=17
    )
    assert official_response.content == "Official fixture"
    assert official_response.cost_total is not None

    _assert_collision_response(
        await compatible.achat(
            [UserMessage(content="Say hello.")],
            max_tokens=17,
            temperature=0.4,
            top_p=0.8,
        ),
        "gpt-5",
    )

    assert len(attempts) == 3
    assert [str(request.url) for request in attempts] == [
        _COLLISION_URL,
        "https://api.openai.com/v1/responses",
        _COLLISION_URL,
    ]
    for index in (0, 2):
        request = attempts[index]
        assert request.headers["Authorization"] == f"Bearer {_COLLISION_KEY}"
        assert json.loads(request.content) == _collision_body(
            "gpt-5", streaming=False
        )
    _assert_native_gpt5_request(attempts[1])
    assert forbidden == []


_MALFORMED_COMPATIBLE_COMPLETIONS = [
    None,
    {},
    {"choices": []},
    {"choices": "bad"},
    {"choices": [None]},
    {"choices": [{"message": None, "finish_reason": "stop"}]},
    {"choices": [{"message": {}, "finish_reason": None}]},
    {"choices": [{"message": {}, "finish_reason": "  "}]},
    {"choices": [{"message": {}, "finish_reason": 1}]},
    {"choices": [{"message": {"content": 1}, "finish_reason": "stop"}]},
    {"choices": [{"message": {"refusal": 1}, "finish_reason": "stop"}]},
    {"choices": [{"message": {"tool_calls": {}}, "finish_reason": "stop"}]},
    {"choices": [{"message": {"function_call": []}, "finish_reason": "stop"}]},
]


@pytest.mark.unit
@pytest.mark.parametrize("body", _MALFORMED_COMPATIBLE_COMPLETIONS)
def test_compatible_chat_rejects_malformed_completion_envelopes(body):
    """Malformed response envelopes fail through the public sanitized error."""
    model = "vendor/unregistered-model"
    with requests_mock.Mocker(real_http=False) as http:
        http.post(
            _COLLISION_URL,
            text="null" if body is None else json.dumps(body),
            headers={"Content-Type": "application/json"},
        )
        adapter = UniversalLLMAPIAdapter(
            organization="openai_compatible",
            model=model,
            api_key=_COLLISION_KEY,
            base_url=_COLLISION_ROOT,
        )
        with pytest.raises(LLMAPIClientError):
            adapter.chat([UserMessage(content="Hello")])
        assert http.call_count == 1
        assert http.request_history[0].url == _COLLISION_URL


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("body", _MALFORMED_COMPATIBLE_COMPLETIONS)
async def test_compatible_async_chat_rejects_malformed_completion_envelopes(
    body, monkeypatch
):
    real_client = httpx.AsyncClient
    requests_seen = []

    def handler(request):
        requests_seen.append(request)
        return httpx.Response(
            200,
            content=b"null" if body is None else json.dumps(body).encode(),
        )

    client = real_client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "AsyncClient", lambda: client)
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model="vendor/unregistered-model",
        api_key=_COLLISION_KEY,
        base_url=_COLLISION_ROOT,
    )
    with pytest.raises(LLMAPIClientError):
        await adapter.achat([UserMessage(content="Hello")])
    assert len(requests_seen) == 1
    assert str(requests_seen[0].url) == _COLLISION_URL
    await client.aclose()


@pytest.mark.unit
@pytest.mark.parametrize(
    "transport",
    ["requests", "httpx"],
)
@pytest.mark.parametrize(
    "malformed",
    [
        "not-an-object",
        {"choices": "bad"},
        {"choices": [{"index": 0, "delta": "bad", "finish_reason": None}]},
    ],
)
def test_compatible_sync_stream_rejects_malformed_envelopes_before_completion(
    transport, malformed, monkeypatch
):
    model = "vendor/unregistered-model"
    lines = event_lines(
        {"choices": [{"index": 0, "delta": {"content": "pending"}, "finish_reason": None}]}
    ) + event_lines(malformed) + event_lines(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    )
    completed = []
    visible = []
    adapter, requests_seen, resource, client_closes, client = _counted_sync_stream_route(
        transport, lines, monkeypatch, model
    )
    try:
        with pytest.raises(LLMAPIClientError):
            visible.extend(
                adapter.stream_chat(
                    [UserMessage(content="Hello")],
                    on_done=completed.append,
                    buffer_chars=64,
                )
            )
        assert len(requests_seen) == 1
        assert resource.close_count == 1
        assert visible == []
        assert completed == []
        if transport == "httpx":
            assert client_closes == [True]
    finally:
        if resource.close_count == 0:
            resource.close()
        if transport == "httpx" and not client_closes:
            client.close()


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "malformed",
    [
        "not-an-object",
        {"choices": "bad"},
        {"choices": [{"index": 0, "delta": "bad", "finish_reason": None}]},
    ],
)
async def test_compatible_async_stream_rejects_malformed_before_completion(
    malformed, monkeypatch
):
    model = "vendor/unregistered-model"
    lines = event_lines(
        {"choices": [{"index": 0, "delta": {"content": "pending"}, "finish_reason": None}]}
    ) + event_lines(malformed) + event_lines(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    )
    completed = []
    visible = []
    adapter, requests_seen, resource, client_closes, client = (
        _counted_async_stream_route(lines, monkeypatch, model)
    )
    try:
        with pytest.raises(LLMAPIClientError):
            async for text in adapter.astream_chat(
                [UserMessage(content="Hello")],
                on_done=completed.append,
                buffer_chars=64,
            ):
                visible.append(text)
        assert len(requests_seen) == 1
        assert str(requests_seen[0].url) == _COLLISION_URL
        assert resource.close_count == 1
        assert client_closes == [True]
        assert visible == []
        assert completed == []
    finally:
        if resource.close_count == 0:
            await resource.aclose()
        if not client_closes:
            await client.aclose()


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_compatible_empty_completion_without_optional_metadata_normalizes(
    transport, monkeypatch
):
    body = {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}
    model = "vendor/unregistered-model"
    if transport == "requests":
        with requests_mock.Mocker(real_http=False) as http:
            http.post(
                _COLLISION_URL,
                json=body,
                headers={"Content-Type": "application/json"},
            )
            adapter = UniversalLLMAPIAdapter(
                organization="openai_compatible",
                model=model,
                api_key=_COLLISION_KEY,
                base_url=_COLLISION_ROOT,
            )
            with pytest.warns(UserWarning, match="empty content"):
                response = adapter.chat([UserMessage(content="Hello")])
            assert http.call_count == 1
            assert http.request_history[0].url == _COLLISION_URL
    else:
        requests_seen = []
        real_client = httpx.Client

        def handler(request):
            requests_seen.append(request)
            return httpx.Response(200, json=body)

        client = real_client(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(httpx, "Client", lambda: client)
        adapter = UniversalLLMAPIAdapter(
            organization="openai_compatible",
            model=model,
            api_key=_COLLISION_KEY,
            base_url=_COLLISION_ROOT,
            transport="httpx",
        )
        try:
            with pytest.warns(UserWarning, match="empty content"):
                response = adapter.chat([UserMessage(content="Hello")])
            assert len(requests_seen) == 1
            assert str(requests_seen[0].url) == _COLLISION_URL
        finally:
            client.close()
    assert response.content == ""
    assert response.finish_reason == "stop"
    assert response.response_id is None
    assert response.usage is None


@pytest.mark.unit
def test_compatible_portable_schema_diagnostic_uses_compatible_provider_label():
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string", "pattern": "x+"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    original = json.loads(json.dumps(schema))
    adapter = _new_adapter(
        api_key="test-key", model="vendor/schema", base_url="https://example.invalid/v1"
    )

    with pytest.raises(JSONSchemaError) as caught:
        adapter._to_openai_structured_output_schema(schema)

    assert "openai_compatible" in caught.value.detail
    assert "#/properties/answer" in caught.value.detail
    assert "pattern" in caught.value.detail
    assert schema == original


@pytest.mark.unit
def test_compatible_malformed_tool_arguments_have_mode_correct_safe_diagnostic():
    malformed = tool_call_response(
        name="lookup", arguments_json='{"city":', call_id="call_bad_arguments"
    )
    compatible = _new_adapter(
        api_key="test-key", model="vendor/tool-validation",
        base_url="https://example.invalid/v1",
    )
    native = OpenAIAdapter(api_key="test-key", model="gpt-4o-mini")

    with pytest.raises(InvalidToolArgumentsError) as compatible_error:
        compatible._normalize_chat_completions_response(malformed)
    with pytest.raises(InvalidToolArgumentsError) as native_error:
        native._normalize_chat_completions_response(malformed)

    assert "openai_compatible" in compatible_error.value.detail
    assert "OpenAI" not in compatible_error.value.detail
    assert "OpenAI" in native_error.value.detail
    assert "{\"city\":" not in compatible_error.value.detail


@pytest.mark.unit
def test_compatible_tool_diagnostic_preserves_sdk_error_identity_and_attributes(
    monkeypatch,
):
    compatible = _new_adapter(
        api_key="test-key", model="vendor/tool-validation",
        base_url="https://example.invalid/v1",
    )
    original_error = InvalidToolArgumentsError(
        detail="OpenAI tool arguments JSON parse failed: private argument text"
    )
    original_error.custom_marker = object()
    original_marker = original_error.custom_marker

    def raise_original(cls, response):
        raise original_error

    monkeypatch.setattr(ChatResponse, "from_openai_response", classmethod(raise_original))
    with pytest.raises(InvalidToolArgumentsError) as caught:
        compatible._normalize_chat_completions_response({
            "choices": [{
                "message": {"content": ""},
                "finish_reason": "stop",
            }],
        })

    assert caught.value is original_error
    assert caught.value.custom_marker is original_marker
    assert "openai_compatible" in caught.value.detail
    assert "private argument text" not in caught.value.detail
    assert caught.value.args == (
        f"{caught.value.message} Detail: {caught.value.detail}",
    )


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_compatible_terminal_only_stream_without_done_succeeds(transport, monkeypatch):
    lines = event_lines(
        {"choices": [{"index": 0, "finish_reason": "stop"}]}
    )
    completed = []
    adapter, requests_seen, resource, client_closes, client = (
        _counted_sync_stream_route(
            transport, lines, monkeypatch, "vendor/unregistered-model"
        )
    )
    try:
        with pytest.warns(UserWarning, match="empty content"):
            visible = list(
                adapter.stream_chat(
                    [UserMessage(content="Hello")], on_done=completed.append
                )
            )
        assert visible == []
        assert len(requests_seen) == 1
        assert resource.close_count == 1
        assert completed[0].finish_reason == "stop"
        if transport == "httpx":
            assert client_closes == [True]
    finally:
        if resource.close_count == 0:
            resource.close()
        if transport == "httpx" and not client_closes:
            client.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compatible_async_terminal_only_stream_without_done_succeeds(monkeypatch):
    lines = event_lines(
        {"choices": [{"index": 0, "finish_reason": "stop"}]}
    )
    completed = []
    adapter, requests_seen, resource, client_closes, client = (
        _counted_async_stream_route(
            lines, monkeypatch, "vendor/unregistered-model"
        )
    )
    try:
        with pytest.warns(UserWarning, match="empty content"):
            visible = [
                text async for text in adapter.astream_chat(
                    [UserMessage(content="Hello")], on_done=completed.append
                )
            ]
        assert visible == []
        assert len(requests_seen) == 1
        assert resource.close_count == 1
        assert client_closes == [True]
        assert completed[0].finish_reason == "stop"
    finally:
        if resource.close_count == 0:
            await resource.aclose()
        if not client_closes:
            await client.aclose()


@pytest.mark.unit
@pytest.mark.parametrize("transport", ["requests", "httpx"])
def test_compatible_done_sentinel_without_finish_reason_is_failure(transport, monkeypatch):
    lines = event_lines(
        {"choices": [{"index": 0, "delta": {"content": "pending"}}]}
    ) + ["data: [DONE]", ""]
    completed = []
    visible = []
    adapter, requests_seen, resource, client_closes, client = (
        _counted_sync_stream_route(
            transport, lines, monkeypatch, "vendor/unregistered-model"
        )
    )
    try:
        with pytest.raises(LLMAPIClientError):
            visible.extend(
                adapter.stream_chat(
                    [UserMessage(content="Hello")],
                    buffer_chars=64,
                    on_done=completed.append,
                )
            )
        assert visible == []
        assert completed == []
        assert len(requests_seen) == 1
        assert resource.close_count == 1
        if transport == "httpx":
            assert client_closes == [True]
    finally:
        if resource.close_count == 0:
            resource.close()
        if transport == "httpx" and not client_closes:
            client.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_compatible_async_done_sentinel_without_finish_reason_is_failure(
    monkeypatch,
):
    lines = event_lines(
        {"choices": [{"index": 0, "delta": {"content": "pending"}}]}
    ) + ["data: [DONE]", ""]
    completed = []
    visible = []
    adapter, requests_seen, resource, client_closes, client = (
        _counted_async_stream_route(
            lines, monkeypatch, "vendor/unregistered-model"
        )
    )
    try:
        with pytest.raises(LLMAPIClientError):
            async for text in adapter.astream_chat(
                [UserMessage(content="Hello")],
                buffer_chars=64,
                on_done=completed.append,
            ):
                visible.append(text)
        assert visible == []
        assert completed == []
        assert len(requests_seen) == 1
        assert resource.close_count == 1
        assert client_closes == [True]
    finally:
        if resource.close_count == 0:
            await resource.aclose()
        if not client_closes:
            await client.aclose()


_BUFFERED_METADATA_SNAPSHOT_CASES = [
    pytest.param(
        usage_payload(
            10,
            3,
            13,
            prompt_tokens_details={"cached_tokens": 8, "cache_write_tokens": 5},
        ),
        (10, 3, 13, None, None),
        id="contradictory-cache-split",
    ),
    pytest.param(
        usage_payload(
            0,
            0,
            0,
            prompt_tokens_details={"cached_tokens": 0, "cache_write_tokens": 0},
        ),
        (0, 0, 0, 0, 0),
        id="reported-zero-split",
    ),
    pytest.param({}, _METADATA_ALL_UNKNOWN, id="present-empty-usage"),
]


@pytest.mark.unit
@pytest.mark.parametrize("usage, expected_usage", _BUFFERED_METADATA_SNAPSHOT_CASES)
def test_compatible_sync_buffered_metadata_snapshot(usage, expected_usage, monkeypatch):
    snapshot = deepcopy(usage)
    forbidden = _guard_official_model_lookups(monkeypatch)
    chunks = []
    completed = []
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model="gpt-5",
        api_key=_COLLISION_KEY,
        base_url=_COLLISION_ROOT,
    )
    lines = _metadata_stream_lines(usage)
    with requests_mock.Mocker(real_http=False) as http:
        http.post(
            _COLLISION_URL,
            text="\n".join(lines),
            headers={"Content-Type": "text/event-stream"},
        )
        visible = list(
            adapter.stream_chat(
                [UserMessage(content="Say hello.")],
                buffer_chars=len("Fixture reply") + 20,
                on_chunk=chunks.append,
                on_done=completed.append,
            )
        )

        assert visible == ["Fixture reply"]
        assert len(completed) == 1
        _assert_compatible_metadata_result(completed[0], expected_usage)
        assert len(chunks) == 1
        assert chunks[0].text == "Fixture reply"
        if expected_usage == _METADATA_ALL_UNKNOWN:
            assert chunks[0].usage is None
        else:
            assert chunks[0].usage is not None
            assert (
                chunks[0].usage.input_tokens,
                chunks[0].usage.output_tokens,
                chunks[0].usage.total_tokens,
                chunks[0].usage.cached_tokens,
                chunks[0].usage.cache_write_tokens,
            ) == expected_usage
            for actual, expected in zip(
                (
                    chunks[0].usage.input_tokens,
                    chunks[0].usage.output_tokens,
                    chunks[0].usage.total_tokens,
                    chunks[0].usage.cached_tokens,
                    chunks[0].usage.cache_write_tokens,
                ),
                expected_usage,
            ):
                if expected == 0:
                    assert type(actual) is int
        assert completed[0].response_id == FIXTURE_ID
        assert http.call_count == 1
        assert http.request_history[0].url == _COLLISION_URL
        assert http.request_history[0].json()["model"] == "gpt-5"

    _assert_official_collision_has_no_profile(adapter)
    _assert_metadata_input_unchanged(usage, snapshot)
    assert forbidden == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("usage, expected_usage", _BUFFERED_METADATA_SNAPSHOT_CASES)
async def test_compatible_async_buffered_metadata_snapshot(usage, expected_usage, monkeypatch):
    snapshot = deepcopy(usage)
    forbidden = _guard_official_model_lookups(monkeypatch)
    chunks = []
    completed = []
    outgoing = []
    real_client = httpx.AsyncClient

    def handler(request):
        outgoing.append(request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content="\n".join(_metadata_stream_lines(usage)),
        )

    client = real_client(transport=httpx.MockTransport(handler))
    constructions = _install_counted_httpx_client(
        monkeypatch, client, asynchronous=True
    )
    adapter = UniversalLLMAPIAdapter(
        organization="openai_compatible",
        model="gpt-5",
        api_key=_COLLISION_KEY,
        base_url=_COLLISION_ROOT,
    )
    try:
        visible = [
            text
            async for text in adapter.astream_chat(
                [UserMessage(content="Say hello.")],
                buffer_chars=len("Fixture reply") + 20,
                on_chunk=chunks.append,
                on_done=completed.append,
            )
        ]
        assert visible == ["Fixture reply"]
        assert len(completed) == 1
        _assert_compatible_metadata_result(completed[0], expected_usage)
        assert len(chunks) == 1
        assert chunks[0].text == "Fixture reply"
        if expected_usage == _METADATA_ALL_UNKNOWN:
            assert chunks[0].usage is None
        else:
            assert chunks[0].usage is not None
            assert (
                chunks[0].usage.input_tokens,
                chunks[0].usage.output_tokens,
                chunks[0].usage.total_tokens,
                chunks[0].usage.cached_tokens,
                chunks[0].usage.cache_write_tokens,
            ) == expected_usage
            for actual, expected in zip(
                (
                    chunks[0].usage.input_tokens,
                    chunks[0].usage.output_tokens,
                    chunks[0].usage.total_tokens,
                    chunks[0].usage.cached_tokens,
                    chunks[0].usage.cache_write_tokens,
                ),
                expected_usage,
            ):
                if expected == 0:
                    assert type(actual) is int
        assert completed[0].response_id == FIXTURE_ID
        assert len(outgoing) == len(constructions) == 1
        assert outgoing[0].method == "POST"
        assert str(outgoing[0].url) == _COLLISION_URL
        assert json.loads(outgoing[0].content)["model"] == "gpt-5"
        _assert_official_collision_has_no_profile(adapter)
        _assert_metadata_input_unchanged(usage, snapshot)
        assert forbidden == []
    finally:
        await client.aclose()
