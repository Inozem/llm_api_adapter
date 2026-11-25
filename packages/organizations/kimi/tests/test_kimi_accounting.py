"""Native Kimi usage and pricing regressions through public execution paths."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping
from pathlib import Path
import sys
from typing import Any

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (PACKAGE_ROOT / "src", REPOSITORY_ROOT / "src", REPOSITORY_ROOT):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llm_registry.llm_registry import (
    Pricing,
    PricingTier,
    RegistrySpec,
    resolve_model_spec,
)
from llm_api_adapter.llms.transports import (
    HTTPErrorHandler,
    JSONResponse,
    SSEEvent,
    StreamErrorHandler,
    TransportRequest,
)
from llm_api_adapter.organization_registry import (
    ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP,
    OrganizationPlugin,
    OrganizationPluginDiscovery,
)
from llm_api_adapter.models.responses.chat_response import ChatResponse
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_kimi.adapter import KimiUsage as AdapterKimiUsage
from llm_api_adapter_kimi.plugin import PLUGIN
from llm_api_adapter_kimi.usage import KimiUsage


KIMI_MODELS = (
    "kimi-k3",
    "kimi-k3.1-pro",
    "kimi-k3.1-pro-highspeed",
    "kimi-k2.6",
)


class _EntryPoint:
    name = "kimi"
    value = "llm_api_adapter_kimi.plugin:PLUGIN"

    def load(self) -> OrganizationPlugin:
        return PLUGIN


class _Transport:
    def __init__(
        self,
        response: Mapping[str, Any],
        events: list[SSEEvent] | None = None,
    ) -> None:
        self.response = dict(response)
        self.events = list(events or [])
        self.json_calls = 0
        self.stream_calls = 0

    def post_json(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
    ) -> JSONResponse:
        _ = request, http_error_handler
        self.json_calls += 1
        return JSONResponse(self.response)

    def post_sse(
        self,
        request: TransportRequest,
        *,
        http_error_handler: HTTPErrorHandler | None = None,
        stream_error_handler: StreamErrorHandler | None = None,
    ) -> Iterator[SSEEvent]:
        _ = request, http_error_handler, stream_error_handler
        self.stream_calls += 1
        return iter(self.events)


@pytest.fixture
def kimi_runtime(monkeypatch: pytest.MonkeyPatch) -> RegistrySpec:
    registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    registry.register_organization_metadata(PLUGIN.model_metadata)
    service_registry = ServiceProviderRegistry()
    PLUGIN.register(service_registry)

    def entry_points(*, group: str) -> list[_EntryPoint]:
        assert group == ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP
        return [_EntryPoint()]

    monkeypatch.setattr("llm_api_adapter.organization_registry.entry_points", entry_points)
    monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", service_registry)
    monkeypatch.setattr(
        universal_module,
        "ORGANIZATION_PLUGIN_DISCOVERY",
        OrganizationPluginDiscovery(),
    )
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)
    return registry


def _adapter(model: str) -> UniversalLLMAPIAdapter:
    return UniversalLLMAPIAdapter(organization="kimi", model=model, api_key="kimi-test-key")


def _registry_tier(registry: RegistrySpec, model: str, prompt_tokens: int) -> PricingTier:
    model_spec = resolve_model_spec(registry, "kimi", model)
    assert model_spec is not None and model_spec.pricing_tiers is not None
    return model_spec.pricing_tiers.tier_for_prompt_tokens(prompt_tokens)


def _assert_cost(actual: float | None, expected: float | None) -> None:
    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)


def _response(model: str, usage: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl-accounting",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "accounted"},
                "finish_reason": "stop",
            },
        ],
        "usage": usage,
    }


def _usage_event(
    model: str,
    usage: Any,
    *,
    finish_reason: str | None = None,
) -> SSEEvent:
    return SSEEvent(
        event=None,
        data={
            "id": "chatcmpl-accounting-stream",
            "model": model,
            "usage": usage,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "done"} if finish_reason is None else {},
                    "finish_reason": finish_reason,
                },
            ],
        },
    )


@pytest.mark.unit
def test_kimi_usage_compatibility_import_and_native_completed_result(
    kimi_runtime: RegistrySpec,
) -> None:
    assert AdapterKimiUsage is KimiUsage
    tier = _registry_tier(kimi_runtime, "kimi-k3", 10)
    assert tier.cache_read_in_per_token is not None
    assert tier.cache_write_in_per_token is not None
    transport = _Transport(
        _response(
            "kimi-k3",
            {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 777,
                "cached_tokens": 2,
                "prompt_tokens_details": {"cache_write_tokens": 1},
            },
        ),
    )
    adapter = _adapter("kimi-k3")
    adapter.adapter._sync_transport = transport

    response = adapter.chat([{"role": "user", "content": "Hello"}])

    assert isinstance(response.usage, KimiUsage)
    # Reported totals remain authoritative, even when nonadditive.
    assert response.usage.total_tokens == 777
    assert response.cost_input == pytest.approx(
        7 * tier.in_per_token
        + 2 * tier.cache_read_in_per_token
        + tier.cache_write_in_per_token,
    )
    assert response.cost_output == pytest.approx(4 * tier.out_per_token)
    assert response.cost_total == pytest.approx(response.cost_input + response.cost_output)
    assert transport.json_calls == 1


@pytest.mark.unit
@pytest.mark.parametrize("model", KIMI_MODELS)
def test_native_exact_model_tariff_prices_disjoint_cache_components_once(
    kimi_runtime: RegistrySpec,
    model: str,
) -> None:
    tier = _registry_tier(kimi_runtime, model, 10)
    input_rate = tier.in_per_token
    output_rate = tier.out_per_token
    read_rate = tier.cache_read_in_per_token
    write_rate = tier.cache_write_in_per_token
    write_count = 1 if write_rate is not None else None
    details: dict[str, int] = {"cached_tokens": 2}
    if write_count is not None:
        details["cache_write_tokens"] = write_count
    payload = _response(
        model,
        {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "cached_tokens": 8,
            "prompt_tokens_details": details,
        },
    )
    adapter = _adapter(model)
    adapter.adapter._sync_transport = _Transport(payload)

    response = adapter.chat([{"role": "user", "content": "Hello"}])

    assert response.usage is not None
    assert response.usage.total_tokens == 14  # Missing total derives only from both base counts.
    assert response.usage.cached_tokens == 2  # Nested count wins over legacy top-level count.
    assert response.usage.cache_write_tokens == write_count
    ordinary = 10 - 2 - (write_count or 0)
    expected_input = (
        ordinary * input_rate
        + 2 * (read_rate or 0.0)
        + (write_count or 0) * (write_rate or 0.0)
    )
    assert response.cost_input == pytest.approx(expected_input)
    assert response.cost_output == pytest.approx(4 * output_rate)
    assert response.cost_total == pytest.approx(response.cost_input + response.cost_output)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("usage", "expected_usage", "input_cost_tokens", "output_cost_tokens"),
    [
        (
            {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            (10, 4, 14, None, None),
            None,
            4,
        ),
        (
            {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "cached_tokens": 2,
            },
            (10, 4, 14, 2, None),
            None,
            4,
        ),
        (
            {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
                "cached_tokens": 0,
                "prompt_tokens_details": {"cache_write_tokens": 0},
            },
            (10, 4, 14, 0, 0),
            10,
            4,
        ),
        (
            {
                "prompt_tokens": 10,
                "total_tokens": 10,
                "cached_tokens": 0,
                "prompt_tokens_details": {"cache_write_tokens": 0},
            },
            (10, None, 10, 0, 0),
            10,
            None,
        ),
        (
            {"completion_tokens": 4, "total_tokens": 4},
            (None, 4, 4, None, None),
            None,
            None,
        ),
        (
            {"completion_tokens": 4},
            (None, 4, None, None, None),
            None,
            None,
        ),
        (
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "prompt_tokens_details": {"cache_write_tokens": 0},
            },
            (0, 0, 0, 0, 0),
            0,
            0,
        ),
    ],
    ids=(
        "omitted-cache-splits",
        "legacy-cache-read",
        "reported-zero-cache",
        "input-only",
        "output-only",
        "output-without-total",
        "all-zero",
    ),
)
def test_completed_usage_presence_total_and_cost_matrix(
    kimi_runtime: RegistrySpec,
    usage: dict[str, Any],
    expected_usage: tuple[int | None, int | None, int | None, int | None, int | None],
    input_cost_tokens: int | None,
    output_cost_tokens: int | None,
) -> None:
    adapter = _adapter("kimi-k3")
    adapter.adapter._sync_transport = _Transport(_response("kimi-k3", usage))

    response = adapter.chat([{"role": "user", "content": "Hello"}])

    assert isinstance(response.usage, KimiUsage)
    assert (
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.usage.total_tokens,
        response.usage.cached_tokens,
        response.usage.cache_write_tokens,
    ) == expected_usage
    tier = _registry_tier(kimi_runtime, "kimi-k3", 10)
    expected_input_cost = (
        input_cost_tokens * tier.in_per_token
        if input_cost_tokens is not None
        else None
    )
    expected_output_cost = (
        output_cost_tokens * tier.out_per_token
        if output_cost_tokens is not None
        else None
    )
    expected_total_cost = (
        expected_input_cost + expected_output_cost
        if expected_input_cost is not None and expected_output_cost is not None
        else None
    )
    _assert_cost(response.cost_input, expected_input_cost)
    _assert_cost(response.cost_output, expected_output_cost)
    _assert_cost(response.cost_total, expected_total_cost)


@pytest.mark.unit
def test_positive_unpriced_k2_cache_write_keeps_input_cost_unknown(
    kimi_runtime: RegistrySpec,
) -> None:
    tier = _registry_tier(kimi_runtime, "kimi-k2.6", 10)
    adapter = _adapter("kimi-k2.6")
    adapter.adapter._sync_transport = _Transport(
        _response(
            "kimi-k2.6",
            {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "cached_tokens": 0,
                "prompt_tokens_details": {"cache_write_tokens": 1},
            },
        ),
    )

    response = adapter.chat([{"role": "user", "content": "Hello"}])

    assert isinstance(response.usage, KimiUsage)
    assert response.usage.cache_write_tokens == 1
    assert response.cost_input is None
    assert response.cost_output == pytest.approx(4 * tier.out_per_token)
    assert response.cost_total is None


@pytest.mark.unit
def test_caller_rate_override_stays_local_to_kimi_adapter(kimi_runtime: RegistrySpec) -> None:
    kimi = _adapter("kimi-k3")
    kimi.adapter.pricing = Pricing(
        (PricingTier(None, 8e-6, 9e-6, 1e-6, 2e-6),),
        currency="EUR",
    )
    payload = _response(
        "kimi-k3",
        {
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "cached_tokens": 2,
            "prompt_tokens_details": {"cache_write_tokens": 1},
        },
    )
    kimi.adapter._sync_transport = _Transport(payload)

    response = kimi.chat([{"role": "user", "content": "Hello"}])

    assert response.currency == "EUR"
    assert response.cost_input == pytest.approx(7 * 8e-6 + 2 * 1e-6 + 1 * 2e-6)
    assert response.cost_output == pytest.approx(4 * 9e-6)
    assert response.cost_total == pytest.approx(response.cost_input + response.cost_output)

    native_tier = _registry_tier(kimi_runtime, "kimi-k3", 10)
    default_kimi = _adapter("kimi-k3")
    default_kimi.adapter._sync_transport = _Transport(payload)
    native_response = default_kimi.chat([{"role": "user", "content": "Hello"}])
    expected_native_input = 7 * native_tier.in_per_token
    expected_native_input += 2 * (native_tier.cache_read_in_per_token or 0.0)
    expected_native_input += native_tier.cache_write_in_per_token or 0.0
    expected_native_output = 4 * native_tier.out_per_token
    assert native_response.currency == "USD"
    assert native_response.cost_input == pytest.approx(expected_native_input)
    assert native_response.cost_output == pytest.approx(expected_native_output)
    assert native_response.cost_total == pytest.approx(
        expected_native_input + expected_native_output,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "usage",
    [
        [],
        {},
        {"prompt_tokens": True},
        {"prompt_tokens": -1},
        {"prompt_tokens": 1.5},
        {"prompt_tokens": 4, "completion_tokens": 2, "cached_tokens": 2,
         "prompt_tokens_details": {"cache_write_tokens": 3}},
    ],
)
def test_completed_path_rejects_malformed_or_contradictory_usage(
    kimi_runtime: RegistrySpec,
    usage: Any,
) -> None:
    adapter = _adapter("kimi-k3")
    transport = _Transport(_response("kimi-k3", usage))
    adapter.adapter._sync_transport = transport

    with pytest.raises(LLMAPIClientError):
        adapter.chat([{"role": "user", "content": "Hello"}])
    assert transport.json_calls == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_completed_async_path_uses_native_usage_and_authoritative_total(
    kimi_runtime: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    tier = _registry_tier(kimi_runtime, "kimi-k3", 10)
    calls = 0

    async def fake_request(url: str, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _response(
            "kimi-k3",
            {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 700},
        )

    monkeypatch.setattr(async_client_module, "async_request", fake_request)

    response = await _adapter("kimi-k3").achat([{"role": "user", "content": "Hello"}])
    assert isinstance(response.usage, KimiUsage)
    assert response.usage.total_tokens == 700
    assert response.usage.cached_tokens is None
    assert response.usage.cache_write_tokens is None
    assert response.cost_input is None
    assert response.cost_output == pytest.approx(4 * tier.out_per_token)
    assert response.cost_total is None
    assert calls == 1


@pytest.mark.unit
def test_streaming_shallow_usage_merge_keeps_counts_but_replaces_nested_details(
    kimi_runtime: RegistrySpec,
) -> None:
    tier = _registry_tier(kimi_runtime, "kimi-k3", 10)
    adapter = _adapter("kimi-k3")
    completed: list[ChatResponse] = []
    transport = _Transport(
        {},
        [
            _usage_event(
                "kimi-k3",
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 999,
                    "cached_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 2},
                },
            ),
            _usage_event(
                "kimi-k3",
                {"completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 0}},
                finish_reason="stop",
            ),
        ],
    )
    adapter.adapter._sync_transport = transport

    output = list(
        adapter.stream_chat(
            [{"role": "user", "content": "Hello"}],
            on_done=completed.append,
        ),
    )

    assert output == ["done"]
    assert transport.stream_calls == 1
    assert len(completed) == 1
    usage = completed[0].usage
    assert isinstance(usage, KimiUsage)
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (10, 5, 999)
    assert usage.cached_tokens == 0  # Nested explicit zero has priority over legacy count 3.
    assert usage.cache_write_tokens is None  # Later nested details replace, rather than deep merge.
    assert completed[0].cost_input is None
    assert completed[0].cost_output == pytest.approx(5 * tier.out_per_token)
    assert completed[0].cost_total is None


@pytest.mark.unit
def test_stream_interim_bad_usage_is_tolerated_but_bad_final_data_fails_without_on_done(
    kimi_runtime: RegistrySpec,
) -> None:
    interim_adapter = _adapter("kimi-k3")
    interim_done: list[ChatResponse] = []
    interim_adapter.adapter._sync_transport = _Transport(
        {},
        [
            _usage_event("kimi-k3", {"prompt_tokens": -4}),
            _usage_event(
                "kimi-k3",
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 1,
                    "prompt_tokens_details": {"cache_write_tokens": 1},
                },
                finish_reason="stop",
            ),
        ],
    )
    assert list(
        interim_adapter.stream_chat(
            [{"role": "user", "content": "Hello"}],
            on_done=interim_done.append,
        ),
    ) == ["done"]
    assert len(interim_done) == 1
    assert isinstance(interim_done[0].usage, KimiUsage)
    assert interim_done[0].usage.input_tokens == 10

    invalid_adapter = _adapter("kimi-k3")
    invalid_done: list[ChatResponse] = []
    invalid_adapter.adapter._sync_transport = _Transport(
        {},
        [
            _usage_event("kimi-k3", {}),
            _usage_event(
                "kimi-k3",
                {"prompt_tokens": -4, "completion_tokens": 2},
                finish_reason="stop",
            ),
        ],
    )
    with pytest.raises(LLMAPIClientError, match="prompt_tokens must be a non-negative integer"):
        list(
            invalid_adapter.stream_chat(
                [{"role": "user", "content": "Hello"}],
                on_done=invalid_done.append,
            ),
        )
    assert invalid_done == []


@pytest.mark.unit
def test_stream_merged_cache_write_contradiction_fails_after_visible_output(
    kimi_runtime: RegistrySpec,
) -> None:
    adapter = _adapter("kimi-k3")
    completed: list[ChatResponse] = []
    adapter.adapter._sync_transport = _Transport(
        {},
        [
            _usage_event("kimi-k3", {"prompt_tokens": 2}),
            _usage_event(
                "kimi-k3",
                {"prompt_tokens_details": {"cache_write_tokens": 3}},
                finish_reason="stop",
            ),
        ],
    )
    stream = adapter.stream_chat(
        [{"role": "user", "content": "Hello"}],
        on_done=completed.append,
    )

    assert next(stream) == "done"
    with pytest.raises(
        LLMAPIClientError,
        match="cached and cache-write tokens must not exceed prompt_tokens",
    ):
        next(stream)
    assert completed == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_async_stream_uses_native_accumulated_usage(
    kimi_runtime: RegistrySpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_api_adapter_kimi.clients import async_client as async_client_module

    tier = _registry_tier(kimi_runtime, "kimi-k3", 10)
    calls = 0

    def fake_stream_request(url: str, **kwargs: Any) -> AsyncIterator[SSEEvent]:
        nonlocal calls
        calls += 1

        async def events() -> AsyncIterator[SSEEvent]:
            yield _usage_event(
                "kimi-k3",
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 500,
                    "cached_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 1},
                },
            )
            yield _usage_event(
                "kimi-k3",
                {"completion_tokens": 4},
                finish_reason="stop",
            )

        return events()

    monkeypatch.setattr(async_client_module, "async_stream_request", fake_stream_request)
    completed: list[ChatResponse] = []
    output = [
        text
        async for text in _adapter("kimi-k3").astream_chat(
            [{"role": "user", "content": "Hello"}],
            on_done=completed.append,
        )
    ]

    assert output == ["done"]
    assert calls == 1
    assert len(completed) == 1
    usage = completed[0].usage
    assert isinstance(usage, KimiUsage)
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (10, 4, 500)
    assert usage.cached_tokens == 2
    assert usage.cache_write_tokens == 1
    assert tier.cache_read_in_per_token is not None
    assert tier.cache_write_in_per_token is not None
    assert completed[0].cost_input == pytest.approx(
        7 * tier.in_per_token
        + 2 * tier.cache_read_in_per_token
        + tier.cache_write_in_per_token,
    )
    expected_output_cost = 4 * tier.out_per_token
    assert completed[0].cost_output == pytest.approx(expected_output_cost)
    assert completed[0].cost_total == pytest.approx(
        completed[0].cost_input + expected_output_cost,
    )
