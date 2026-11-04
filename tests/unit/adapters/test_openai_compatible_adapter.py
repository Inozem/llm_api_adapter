"""Basic synchronous request contract for compatible OpenAI endpoints."""

import json
import socket

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
