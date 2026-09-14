"""Deterministic contracts for Kimi's synchronous Chat Completions slice."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

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
    LLMAPITokenLimitError,
    LLMAPIUsageLimitError,
)
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec, resolve_model_spec
from llm_api_adapter.llms.transports import JSONResponse
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


KIMI_MODELS = ("kimi-k3", "kimi-k2.7-code", "kimi-k2.6")


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


def kimi_response(model: str, *, cached_tokens: int | None = 12) -> dict:
    usage = {
        "prompt_tokens": 19,
        "completion_tokens": 13,
        "total_tokens": 32,
    }
    if cached_tokens is not None:
        usage["cached_tokens"] = cached_tokens
    return {
        "id": "cmpl-kimi-test",
        "created": 1_789_721_600,
        "model": model,
        "choices": [
            {
                "message": {"role": "assistant", "content": "Kimi test."},
                "finish_reason": "stop",
            },
        ],
        "usage": usage,
    }


@pytest.fixture
def kimi_runtime(monkeypatch):
    from llm_api_adapter_kimi.plugin import PLUGIN

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
@pytest.mark.parametrize("model", KIMI_MODELS)
def test_kimi_plugin_registers_each_declared_model(kimi_runtime, model):
    from llm_api_adapter_kimi.adapter import KimiAdapter

    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )

    assert isinstance(adapter.adapter, KimiAdapter)
    assert adapter.adapter.service_provider == "kimi"
    assert resolve_model_spec(kimi_runtime, "kimi", model) is adapter.adapter.model_spec
    assert adapter.adapter.model_spec is not None
    assert adapter.adapter.model_spec.limits.context_window_tokens in {
        262_144,
        1_048_576,
    }


@pytest.mark.unit
def test_kimi_metadata_records_the_pricing_snapshot_and_cache_split():
    from llm_api_adapter_kimi.registry import CACHE_PRICING, ORGANIZATION_DATA

    assert tuple(ORGANIZATION_DATA["models"]) == KIMI_MODELS
    for model in KIMI_MODELS:
        assert ORGANIZATION_DATA["models"][model]["pricing_as_of"] == "2026-09-14"
        assert CACHE_PRICING[model].cache_hit_input_per_token > 0
        assert CACHE_PRICING[model].cache_miss_input_per_token > 0


@pytest.mark.integration
@pytest.mark.parametrize("model", KIMI_MODELS)
def test_universal_chat_uses_chat_completions_for_every_declared_model(
    kimi_runtime,
    model,
):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport(kimi_response(model))
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [
            {"role": "system", "content": "Reply briefly."},
            {"role": "user", "content": "Hello"},
        ],
        max_tokens=64,
        timeout_s=12.5,
    )

    assert response.content == "Kimi test."
    assert response.response_id == "cmpl-kimi-test"
    assert response.usage is not None
    assert response.usage.total_tokens == 32

    request = transport.requests[0]
    assert request.url == "https://api.moonshot.ai/v1/chat/completions"
    assert request.headers_dict() == {
        "Authorization": "Bearer kimi-test-key",
        "Content-Type": "application/json",
    }
    assert request.timeout == 12.5
    expected_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Reply briefly."},
            {"role": "user", "content": "Hello"},
        ],
    }
    expected_payload["max_completion_tokens" if model == "kimi-k3" else "max_tokens"] = 64
    assert request.payload == expected_payload


@pytest.mark.integration
@pytest.mark.parametrize(
    ("model", "cache_hit_rate", "cache_miss_rate", "output_rate"),
    [
        ("kimi-k3", 0.30, 3.00, 15.00),
        ("kimi-k2.7-code", 0.19, 0.95, 4.00),
        ("kimi-k2.6", 0.16, 0.95, 4.00),
    ],
)
def test_kimi_prices_reported_cache_hits_exactly(
    kimi_runtime,
    model,
    cache_hit_rate,
    cache_miss_rate,
    output_rate,
):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(kimi_response(model))

    response = adapter.chat([{"role": "user", "content": "Hello"}])

    assert response.usage is not None
    assert response.usage.cached_tokens == 12
    assert response.currency == "USD"
    assert response.cost_input == pytest.approx(
        (12 * cache_hit_rate + 7 * cache_miss_rate) / 1_000_000,
    )
    assert response.cost_output == pytest.approx(13 * output_rate / 1_000_000)
    assert response.cost_total == pytest.approx(
        response.cost_input + response.cost_output,
    )


@pytest.mark.unit
def test_kimi_does_not_publish_a_total_when_the_cache_split_is_missing(kimi_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        kimi_response("kimi-k3", cached_tokens=None),
    )

    response = adapter.chat([{"role": "user", "content": "Hello"}])

    assert response.cost_input is None
    assert response.cost_total is None


@pytest.mark.unit
@pytest.mark.parametrize("model", KIMI_MODELS)
def test_kimi_drops_fixed_sampling_parameters_from_registry_rules(kimi_runtime, model):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model=model,
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport(kimi_response(model))
    adapter.adapter._sync_transport = transport

    with pytest.warns(UserWarning, match="will be ignored"):
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            temperature=0.25,
            top_p=0.5,
        )

    assert "temperature" not in transport.requests[0].payload
    assert "top_p" not in transport.requests[0].payload


@pytest.mark.unit
def test_kimi_unknown_model_has_no_inferred_metadata_or_request_rules(kimi_runtime):
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        adapter = UniversalLLMAPIAdapter(
            organization="kimi",
            model="kimi-unverified-model",
            api_key="kimi-test-key",
        )
    transport = FakeSyncTransport(kimi_response("kimi-unverified-model"))
    adapter.adapter._sync_transport = transport

    response = adapter.chat(
        [{"role": "user", "content": "Hello"}],
        max_tokens=10,
        temperature=0.25,
        top_p=0.5,
    )

    assert any("not verified" in str(warning.message) for warning in captured)
    assert adapter.adapter.model_spec is None
    assert adapter.adapter.pricing is None
    assert transport.requests[0].payload["max_tokens"] == 10
    assert transport.requests[0].payload["temperature"] == 0.25
    assert transport.requests[0].payload["top_p"] == 0.5
    assert response.currency is None
    assert response.cost_total is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status_code", "error_type", "expected_error"),
    [
        (401, "authentication_error", LLMAPIAuthorizationError),
        (400, "invalid_request_error", LLMAPIClientError),
        (429, "rate_limit_error", LLMAPIRateLimitError),
        (504, "timeout_error", LLMAPITimeoutError),
        (500, "api_error", LLMAPIServerError),
        (400, "max_tokens_exceeded", LLMAPITokenLimitError),
        (400, "quota_exceeded", LLMAPIUsageLimitError),
    ],
)
def test_kimi_normalizes_chat_completions_http_failures(
    kimi_runtime,
    status_code,
    error_type,
    expected_error,
):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(
        {},
        error=FakeHTTPError(
            status_code,
            {"error": {"type": error_type, "message": "Kimi test failure"}},
        ),
    )

    with pytest.raises(expected_error, match="Kimi test failure"):
        adapter.chat([{"role": "user", "content": "Hello"}])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"choices": []}, "response.choices"),
        ({"choices": [{"message": {"content": 42}}]}, "content must be"),
        (
            {
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {"prompt_tokens": "19", "completion_tokens": 13, "total_tokens": 32},
            },
            "usage token counts",
        ),
        (
            {
                "choices": [{"message": {"content": "Hello"}}],
                "usage": {"prompt_tokens": 19, "completion_tokens": 13, "total_tokens": 32, "cached_tokens": 20},
            },
            "must not exceed",
        ),
    ],
)
def test_kimi_rejects_malformed_chat_completions_responses(kimi_runtime, payload, detail):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    adapter.adapter._sync_transport = FakeSyncTransport(payload)

    with pytest.raises(LLMAPIClientError, match=detail):
        adapter.chat([{"role": "user", "content": "Hello"}])


@pytest.mark.unit
def test_kimi_rejects_unimplemented_features_before_transport(kimi_runtime):
    adapter = UniversalLLMAPIAdapter(
        organization="kimi",
        model="kimi-k3",
        api_key="kimi-test-key",
    )
    transport = FakeSyncTransport(kimi_response("kimi-k3"))
    adapter.adapter._sync_transport = transport

    with pytest.raises(NotImplementedError, match="reasoning controls"):
        adapter.chat(
            [{"role": "user", "content": "Hello"}],
            reasoning_level="high",
        )

    assert transport.requests == []
