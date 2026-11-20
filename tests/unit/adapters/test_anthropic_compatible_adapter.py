"""Direct public contract for the Anthropic-compatible Messages adapter."""

import asyncio
import inspect
import json
import logging
import socket
import traceback

import httpx
import pytest
import requests
import requests_mock

from src.llm_api_adapter.errors.config_errors import LLMConfigError
from src.llm_api_adapter.errors.llm_api_error import LLMAPIAuthorizationError
from src.llm_api_adapter.models.messages.chat_message import UserMessage


def _new_anthropic_compatible_adapter(**kwargs):
    """Import the planned public adapter at runtime so pre-feature collection works."""
    from src.llm_api_adapter.adapters.anthropic_compatible.adapter import (
        AnthropicCompatibleAdapter,
    )

    return AnthropicCompatibleAdapter(**kwargs)


def _forbid_contract_boundary(boundary, calls):
    def forbidden(*_args, **_kwargs):
        calls.append(boundary)
        raise AssertionError(f"unexpected discovery or I/O: {boundary}")

    return forbidden


def _guard_model_and_discovery(monkeypatch):
    import src.llm_api_adapter.adapters.base_adapter as base_adapter_module
    import src.llm_api_adapter.llm_registry.llm_registry as registry_module
    import src.llm_api_adapter.universal_adapter as universal_module

    calls = []
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        _forbid_contract_boundary("socket.getaddrinfo", calls),
    )
    monkeypatch.setattr(
        base_adapter_module,
        "resolve_model_spec",
        _forbid_contract_boundary("base_adapter.resolve_model_spec", calls),
    )
    monkeypatch.setattr(
        registry_module,
        "resolve_model_spec",
        _forbid_contract_boundary("llm_registry.resolve_model_spec", calls),
    )
    monkeypatch.setattr(
        universal_module.ORGANIZATION_PLUGIN_DISCOVERY,
        "discover",
        _forbid_contract_boundary("organization_plugin_discovery.discover", calls),
    )
    return calls


def _guard_configuration_no_io(monkeypatch):
    calls = _guard_model_and_discovery(monkeypatch)
    monkeypatch.setattr(
        requests.sessions.Session,
        "send",
        _forbid_contract_boundary("requests.Session.send", calls),
    )
    monkeypatch.setattr(
        httpx.Client,
        "send",
        _forbid_contract_boundary("httpx.Client.send", calls),
    )
    monkeypatch.setattr(
        httpx.AsyncClient,
        "send",
        _forbid_contract_boundary("httpx.AsyncClient.send", calls),
    )
    return calls


def _sensitive_model_and_markers():
    raw_key = "t007/Slash+Plus-Key"
    encoded_lower = "t007%2fSlash%2bPlus-Key"
    encoded_mixed = "t007%2FSlash%2bPlus-Key"
    credential_url = (
        "https://synthetic-user:synthetic-password@private.invalid/path"
        "?token=synthetic-query-value#synthetic-fragment-value"
    )
    model = f"vendor/{raw_key}/{encoded_lower}/{encoded_mixed}/{credential_url}"
    sensitive_markers = (
        raw_key,
        encoded_lower,
        encoded_mixed,
        "synthetic-user",
        "synthetic-password",
        "synthetic-query-value",
        "synthetic-fragment-value",
    )
    return raw_key, model, credential_url, sensitive_markers


@pytest.mark.unit
@pytest.mark.parametrize(
    "model",
    [
        "vendor/unregistered-model",
        "claude-sonnet-4-5-20250929",
        "qwen3.8-max",
        "claude-sonnet-4-5-20250929-2026-11-18",
    ],
)
def test_anthropic_compatible_direct_adapter_keeps_exact_unprofiled_identity(
    model, monkeypatch
):
    attempts = _guard_configuration_no_io(monkeypatch)
    key = "t007-direct-synthetic-key-raw"
    root = "https://example.invalid/proxy/T007%2fSecret/v1///"
    adapter = _new_anthropic_compatible_adapter(
        api_key=key,
        model=model,
        base_url=root,
    )

    assert adapter.model == model
    assert adapter.api_key == key
    assert adapter.company == "anthropic_compatible"
    assert adapter.service_provider == "anthropic_compatible"
    assert adapter.base_url == "https://example.invalid/proxy/T007%2fSecret/v1"
    assert adapter.model_spec is None
    assert adapter.pricing is None
    assert adapter.is_reasoning is False
    assert adapter.is_adaptive_thinking is False
    assert "t007-direct-synthetic-key-raw" not in repr(adapter)
    assert attempts == []


@pytest.mark.unit
def test_anthropic_compatible_direct_constructor_has_only_base_url_extension(
    monkeypatch,
):
    attempts = _guard_configuration_no_io(monkeypatch)
    from src.llm_api_adapter.adapters.anthropic_compatible.adapter import (
        AnthropicCompatibleAdapter,
    )
    from src.llm_api_adapter.adapters.base_adapter import LLMAdapterBase

    base_fields = {
        "api_key", "model", "company", "transport", "pricing",
        "is_reasoning", "is_adaptive_thinking", "service_provider",
    }
    parameters = inspect.signature(AnthropicCompatibleAdapter).parameters
    assert "api_variant" not in parameters
    assert "base_url" in parameters
    assert parameters["base_url"].default is None
    assert base_fields <= set(parameters)
    assert set(parameters) <= base_fields | {"base_url"}

    with pytest.raises(TypeError, match="api_variant"):
        _new_anthropic_compatible_adapter(
            api_key="synthetic-direct-key",
            model="vendor/model",
            base_url="https://example.invalid/v1",
            api_variant=None,
        )

    assert LLMAdapterBase.__dataclass_fields__["model_spec"].init is False
    assert attempts == []


@pytest.mark.unit
def test_anthropic_compatible_direct_adapter_rejects_provider_company_mismatch(
    monkeypatch,
):
    attempts = _guard_configuration_no_io(monkeypatch)
    with pytest.raises(LLMConfigError):
        _new_anthropic_compatible_adapter(
            api_key="synthetic-direct-key",
            model="vendor/model",
            company="anthropic",
            service_provider="anthropic_compatible",
            base_url="https://example.invalid/v1",
        )

    with pytest.raises(LLMConfigError):
        _new_anthropic_compatible_adapter(
            api_key="synthetic-direct-key",
            model="vendor/model",
            company="anthropic_compatible",
            service_provider="anthropic",
            base_url="https://example.invalid/v1",
        )
    assert attempts == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        ({"api_key": "synthetic-key", "model": "", "base_url": "https://example.invalid/v1"}, ValueError),
        ({"api_key": "synthetic-key", "model": None, "base_url": "https://example.invalid/v1"}, ValueError),
        ({"api_key": "synthetic-key", "model": 17, "base_url": "https://example.invalid/v1"}, ValueError),
        ({"api_key": "synthetic-key", "model": False, "base_url": "https://example.invalid/v1"}, ValueError),
        ({"api_key": "", "model": "vendor/model", "base_url": "https://example.invalid/v1"}, ValueError),
        ({"api_key": None, "model": "vendor/model", "base_url": "https://example.invalid/v1"}, ValueError),
        ({"api_key": 23, "model": "vendor/model", "base_url": "https://example.invalid/v1"}, ValueError),
        ({"api_key": False, "model": "vendor/model", "base_url": "https://example.invalid/v1"}, ValueError),
        ({"api_key": "synthetic-key", "model": "vendor/model", "base_url": None}, LLMConfigError),
        ({"api_key": "synthetic-key", "model": "vendor/model", "base_url": ""}, LLMConfigError),
        ({"api_key": "synthetic-key", "model": "vendor/model", "base_url": "ftp://example.invalid/v1"}, LLMConfigError),
        ({"api_key": "synthetic-key", "model": "vendor/model", "base_url": "https://user:synthetic-url-secret@example.invalid/v1"}, LLMConfigError),
        ({"api_key": "synthetic-key", "model": "vendor/model", "base_url": "https://example.invalid/v1?synthetic-query-secret"}, LLMConfigError),
        ({"api_key": "synthetic-key", "model": "vendor/model", "base_url": "https://example.invalid/v1#synthetic-fragment-secret"}, LLMConfigError),
        ({"api_key": "synthetic-key", "model": "vendor/model", "base_url": "https://example.invalid/v1/messages/"}, LLMConfigError),
    ],
)
def test_anthropic_compatible_direct_adapter_rejects_invalid_inputs_and_redacts_diagnostics(
    kwargs, expected_error, monkeypatch
):
    attempts = _guard_configuration_no_io(monkeypatch)
    with pytest.raises(expected_error) as error:
        _new_anthropic_compatible_adapter(**kwargs)

    diagnostic = f"{error.value!s} {error.value!r}"
    for value in (
        "synthetic-key",
        "synthetic-url-secret",
        "synthetic-query-secret",
        "synthetic-fragment-secret",
    ):
        assert value not in diagnostic
    assert attempts == []


@pytest.mark.unit
@pytest.mark.parametrize("surface", ["direct", "facade"])
def test_anthropic_compatible_repr_redacts_key_and_model_url_secrets(
    surface, monkeypatch, caplog
):
    attempts = _guard_configuration_no_io(monkeypatch)
    raw_key, model, credential_url, sensitive_markers = _sensitive_model_and_markers()
    root = "https://example.invalid/proxy/v1"

    with caplog.at_level(logging.DEBUG):
        if surface == "direct":
            target = _new_anthropic_compatible_adapter(
                api_key=raw_key,
                model=model,
                base_url=root,
            )
        else:
            from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

            facade = UniversalLLMAPIAdapter(
                organization="anthropic_compatible",
                model=model,
                api_key=raw_key,
                base_url=root,
            )
            target = (facade, facade.adapter)

    targets = target if isinstance(target, tuple) else (target,)
    if surface == "direct":
        assert target.api_key == raw_key
        assert target.model == model
    else:
        assert target[0].api_key == raw_key
        assert target[0].model == model
    diagnostics = "\n".join([*(repr(item) for item in targets), caplog.text])
    for value in sensitive_markers:
        assert value not in diagnostics
    assert credential_url not in diagnostics
    assert attempts == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mode", "transport"),
    [
        ("chat", "requests"),
        ("stream_chat", "requests"),
        ("chat", "httpx"),
        ("stream_chat", "httpx"),
        ("achat", "requests"),
        ("astream_chat", "requests"),
    ],
)
def test_anthropic_compatible_failure_dispatches_once_to_messages_without_fallback(
    mode, transport, monkeypatch, caplog
):
    """A controlled auth failure must use only the selected Messages root."""
    from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

    attempts = _guard_model_and_discovery(monkeypatch)
    raw_key, model, credential_url, sensitive_markers = _sensitive_model_and_markers()
    root = "https://example.invalid/gateway/tenant/v1"
    expected_url = f"{root}/messages"
    error_body = {
        "error": {
            "type": "authentication_error",
            "message": (
                f"authentication rejected {raw_key}; {credential_url}; "
                "encoded variants t007%2fSlash%2bPlus-Key and "
                "t007%2FSlash%2bPlus-Key"
            ),
        }
    }
    sync_httpx_requests = []
    async_httpx_requests = []
    sync_clients = []
    async_clients = []

    real_sync_client = httpx.Client

    def sync_client_factory(*args, **kwargs):
        kwargs.pop("transport", None)

        def handler(request):
            sync_httpx_requests.append(request)
            return httpx.Response(401, json=error_body, request=request)

        client = real_sync_client(
            *args, transport=httpx.MockTransport(handler), **kwargs
        )
        sync_clients.append(client)
        return client

    real_async_client = httpx.AsyncClient

    def async_client_factory(*args, **kwargs):
        kwargs.pop("transport", None)

        async def handler(request):
            async_httpx_requests.append(request)
            return httpx.Response(401, json=error_body, request=request)

        client = real_async_client(
            *args, transport=httpx.MockTransport(handler), **kwargs
        )
        async_clients.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", sync_client_factory)
    monkeypatch.setattr(httpx, "AsyncClient", async_client_factory)
    completed = []
    options = {"json": error_body, "status_code": 401}

    try:
        with requests_mock.Mocker(real_http=False) as requests_http:
            requests_http.register_uri(requests_mock.ANY, requests_mock.ANY, **options)
            with caplog.at_level(logging.DEBUG):
                facade = UniversalLLMAPIAdapter(
                    organization="anthropic_compatible",
                    model=model,
                    api_key=raw_key,
                    base_url=f"{root}/",
                    transport=transport,
                )
                assert facade.model == model
                assert facade.api_key == raw_key
                with pytest.raises(LLMAPIAuthorizationError) as error:
                    arguments = {
                        "messages": [UserMessage(content="hello")],
                        "max_tokens": 17,
                    }
                    if mode == "chat":
                        facade.chat(**arguments)
                    elif mode == "stream_chat":
                        list(facade.stream_chat(**arguments, on_done=completed.append))
                    elif mode == "achat":
                        asyncio.run(facade.achat(**arguments))
                    else:
                        async def collect():
                            return [
                                chunk
                                async for chunk in facade.astream_chat(
                                    **arguments, on_done=completed.append
                                )
                            ]

                        asyncio.run(collect())

                dispatches = [
                    *requests_http.request_history,
                    *sync_httpx_requests,
                    *async_httpx_requests,
                ]
                request_count = len(requests_http.request_history)
                sync_httpx_count = len(sync_httpx_requests)
                async_httpx_count = len(async_httpx_requests)
                if mode in ("achat", "astream_chat"):
                    assert (request_count, sync_httpx_count, async_httpx_count) == (
                        0, 0, 1
                    )
                elif transport == "requests":
                    assert (request_count, sync_httpx_count, async_httpx_count) == (
                        1, 0, 0
                    )
                else:
                    assert (request_count, sync_httpx_count, async_httpx_count) == (
                        0, 1, 0
                    )
                assert len(dispatches) == 1
                request = dispatches[0]
                assert str(request.url) == expected_url
                assert request.method == "POST"
                payload = request.json() if hasattr(request, "json") else None
                if payload is None:
                    body = request.content
                    if isinstance(body, str):
                        body = body.encode("utf-8")
                    payload = json.loads(body.decode("utf-8"))
                assert payload["model"] == model
                assert completed == []
                assert attempts == []

                diagnostic = "".join(
                    traceback.format_exception(
                        type(error.value), error.value, error.value.__traceback__
                    )
                )
                diagnostic += f"\n{error.value!s}\n{error.value!r}\n{caplog.text}"
                for value in sensitive_markers:
                    assert value not in diagnostic
                assert credential_url not in diagnostic
    finally:
        for client in sync_clients:
            client.close()
        for client in async_clients:
            asyncio.run(client.aclose())
