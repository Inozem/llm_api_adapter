"""Basic synchronous request contract for compatible OpenAI endpoints."""

import json
import socket
import traceback

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
)
from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.models.responses.chat_response import ChatResponse
from src.llm_api_adapter.organization_registry import OrganizationPluginDiscovery
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from tests.fixtures.openai_compatible import (
    FIXTURE_ID,
    FIXTURE_MODEL,
    text_response,
    text_sse_lines,
    usage_payload,
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
