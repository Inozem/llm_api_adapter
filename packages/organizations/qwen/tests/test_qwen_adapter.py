"""Deterministic contract tests for Qwen's Frankfurt Messages adapter."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
CORE_SOURCE = REPOSITORY_ROOT / "src"
PACKAGE_SOURCE = PACKAGE_ROOT / "src"
for source in (str(PACKAGE_SOURCE), str(CORE_SOURCE), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.errors.llm_api_error import (
    LLMAPIAuthorizationError,
    LLMAPIClientError,
    LLMAPIRateLimitError,
    LLMAPIServerError,
    LLMAPITimeoutError,
)
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec, resolve_model_spec
from llm_api_adapter.llms.transports import JSONResponse
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


class FakeSyncTransport:
    def __init__(self, response, *, error=None) -> None:
        self.response = response
        self.error = error
        self.requests = []

    def post_json(self, request, *, http_error_handler=None):
        self.requests.append(request)
        if self.error is not None:
            assert http_error_handler is not None
            http_error_handler(self.error)
        return JSONResponse(self.response)


class FakeHTTPResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class FakeHTTPError(Exception):
    def __init__(self, status_code: int, payload: dict) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = FakeHTTPResponse(status_code, payload)


@pytest.fixture
def qwen_runtime(monkeypatch):
    from llm_api_adapter_qwen.plugin import PLUGIN

    model_registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert model_registry.register_organization_metadata(PLUGIN.model_metadata) is True

    service_provider_registry = ServiceProviderRegistry()
    PLUGIN.register(service_provider_registry)
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", model_registry)
    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        service_provider_registry,
    )
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", model_registry)
    return model_registry


@pytest.mark.unit
@pytest.mark.parametrize(
    "model",
    ["qwen3.8-max", "qwen3.8-flash", "qwen3.7-plus", "qwen3.7-flash"],
)
def test_qwen_plugin_registers_each_declared_model(qwen_runtime, model):
    from llm_api_adapter_qwen.adapter import QwenAdapter

    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )

    assert isinstance(adapter.adapter, QwenAdapter)
    assert adapter.adapter.service_provider == "qwen"
    assert resolve_model_spec(qwen_runtime, "qwen", model) is adapter.adapter.model_spec
    assert adapter.adapter.model_spec is not None
    assert adapter.adapter.model_spec.limits.context_window_tokens == 1_000_000
    assert adapter.adapter.model_spec.limits.max_output_tokens == 131_072


@pytest.mark.integration
@pytest.mark.parametrize(
    "model",
    ["qwen3.8-max", "qwen3.8-flash", "qwen3.7-plus", "qwen3.7-flash"],
)
def test_universal_chat_uses_the_frankfurt_messages_endpoint(qwen_runtime, model):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model=model,
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport(
        {
            "id": "msg-qwen-1",
            "model": model,
            "content": [{"type": "text", "text": "Hallo"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
    )
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [
            {"role": "system", "content": "Reply in German."},
            {"role": "user", "content": "Hello"},
        ],
        max_tokens=64,
        temperature=0.7,
        top_p=0.8,
        timeout_s=12.5,
        workspace_id="frankfurt-workspace",
    )

    assert response.content == "Hallo"
    assert response.response_id == "msg-qwen-1"
    assert response.usage is not None
    assert response.usage.total_tokens == 30
    assert response.currency == "CNY"

    request = transport.requests[0]
    assert request.url == (
        "https://frankfurt-workspace.eu-central-1.maas.aliyuncs.com/"
        "apps/anthropic/v1/messages"
    )
    assert request.headers_dict() == {
        "x-api-key": "qwen-test-key",
        "Content-Type": "application/json",
    }
    assert request.timeout == 12.5
    assert request.payload == {
        "model": model,
        "system": "Reply in German.",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 64,
        "temperature": 0.7,
        "top_p": 0.8,
    }


@pytest.mark.unit
@pytest.mark.parametrize("workspace_id", [None, "", "wrong/workspace", "wrong.workspace"])
def test_qwen_rejects_missing_or_malformed_workspace_before_transport(
    qwen_runtime,
    workspace_id,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    transport = FakeSyncTransport({})
    adapter.adapter._sync_transport = transport

    with pytest.raises(ValueError, match="workspace_id"):
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            workspace_id=workspace_id,
        )

    assert transport.requests == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status_code", "error_type", "expected_error"),
    [
        (401, "authentication_error", LLMAPIAuthorizationError),
        (400, "invalid_request_error", LLMAPIClientError),
        (429, "rate_limit_error", LLMAPIRateLimitError),
        (504, "timeout_error", LLMAPITimeoutError),
        (500, "api_error", LLMAPIServerError),
    ],
)
def test_qwen_normalizes_messages_http_failures(
    qwen_runtime,
    status_code,
    error_type,
    expected_error,
):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        {},
        error=FakeHTTPError(
            status_code,
            {"error": {"type": error_type, "message": "Qwen test failure"}},
        ),
    )

    with pytest.raises(expected_error, match="Qwen test failure"):
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            workspace_id="frankfurt-workspace",
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"id": "msg-qwen-1", "content": "not an array"}, "response.content"),
        (
            {"content": [{"type": "text", "text": 42}]},
            "text content blocks",
        ),
        (
            {
                "content": [],
                "usage": {"input_tokens": "10", "output_tokens": 20},
            },
            "usage token counts",
        ),
    ],
)
def test_qwen_rejects_malformed_messages_response(qwen_runtime, payload, detail):
    adapter = UniversalLLMAPIAdapter(
        organization="qwen",
        model="qwen3.8-max",
        api_key="qwen-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        payload,
    )

    with pytest.raises(LLMAPIClientError, match=detail):
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            max_tokens=64,
            workspace_id="frankfurt-workspace",
        )
