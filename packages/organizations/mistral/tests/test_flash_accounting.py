"""Usage and cost accounting contracts for routed GLM Flash."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
for source in (str(PACKAGE_ROOT / "src"), str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

import llm_api_adapter.adapters.base_adapter as base_adapter_module
import llm_api_adapter.universal_adapter as universal_module
from llm_api_adapter.errors.llm_api_error import LLMAPIClientError
from llm_api_adapter.llm_registry.llm_registry import RegistrySpec
from llm_api_adapter.llms.transports import JSONResponse, SSEEvent
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter
from llm_api_adapter_mistral.plugin import PLUGIN


MODEL = "glm-5.3-flash"
WIRE_MODEL = "zai-glm-5-3-flash"
MESSAGES = [{"role": "user", "content": "Answer briefly."}]
API_KINDS = ("chat", "achat", "stream_chat", "astream_chat")


@pytest.fixture
def mistral_runtime(monkeypatch):
    registry = RegistrySpec()
    assert PLUGIN.model_metadata is not None
    assert registry.register_organization_metadata(PLUGIN.model_metadata)
    providers = ServiceProviderRegistry()
    PLUGIN.register(providers)
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", registry)
    monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", providers)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", registry)
    return registry


def new_adapter():
    adapter = UniversalLLMAPIAdapter(
        organization="zai",
        service_provider="mistral",
        model=MODEL,
        api_key="mistral-test-key",
    )
    assert (adapter.organization, adapter.service_provider, adapter.model) == (
        "zai", "mistral", MODEL
    )
    return adapter


def response(usage, content="Answer."):
    payload = {
        "id": "flash-accounting",
        "model": WIRE_MODEL,
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }
    if usage is not ...:
        payload["usage"] = usage
    return payload


def stream_event(*, usage=..., content="Answer.", finish_reason="stop"):
    payload = {
        "id": "flash-accounting-stream",
        "model": WIRE_MODEL,
        "choices": [{
            "index": 0,
            "delta": {"content": content},
            "finish_reason": finish_reason,
        }],
    }
    if usage is not ...:
        payload["usage"] = usage
    return SSEEvent(event=None, data=payload)


class FakeTransport:
    def __init__(self, result, events=()):
        self.result = result
        self.events = events
        self.requests = []
        self.stream_requests = []

    def post_json(self, request, *, http_error_handler=None):
        self.requests.append(request)
        return JSONResponse(self.result)

    def post_sse(self, request, *, http_error_handler=None, stream_error_handler=None):
        self.stream_requests.append(request)
        return iter(self.events)


def configure(adapter, monkeypatch, *, result=None, events=()):
    transport = FakeTransport(result or {}, events)
    adapter.adapter._sync_transport = transport
    async_calls = []

    async def fake_chat(*, endpoint, payload, **kwargs):
        async_calls.append(("chat", payload))
        return result or {}

    def fake_stream_chat(*, endpoint, payload, **kwargs):
        async_calls.append(("stream", payload))

        async def source():
            for item in events:
                yield item

        return source()

    adapter.adapter._async_client.chat = fake_chat
    adapter.adapter._async_client.stream_chat = fake_stream_chat
    return transport, async_calls


def invoke(adapter, api_kind, monkeypatch, *, usage, content="Answer.", events=None):
    if events is None:
        events = [
            stream_event(content=content, finish_reason="stop"),
            stream_event(usage=usage, content=None, finish_reason=None),
        ]
    transport, async_calls = configure(
        adapter,
        monkeypatch,
        result=response(usage, content),
        events=events,
    )
    completed = []
    if api_kind == "chat":
        result = adapter.chat(MESSAGES, capture_reasoning=True)
    elif api_kind == "achat":
        result = asyncio.run(adapter.achat(MESSAGES, capture_reasoning=True))
    elif api_kind == "stream_chat":
        list(adapter.stream_chat(MESSAGES, capture_reasoning=True, on_done=completed.append))
        result = completed[0]
    else:
        async def collect():
            async def on_done(value):
                completed.append(value)

            async for _ in adapter.astream_chat(
                MESSAGES, capture_reasoning=True, on_done=on_done
            ):
                pass

        asyncio.run(collect())
        result = completed[0]
    assert result is not None
    return result, transport, async_calls


def rates(adapter):
    tier = adapter.adapter.pricing.tier_for_prompt_tokens(1000)
    return tier.in_per_token, tier.cache_read_in_per_token, tier.out_per_token


@pytest.mark.integration
@pytest.mark.parametrize("api_kind", API_KINDS)
def test_flash_confirmed_cache_reads_are_priced_once_and_reasoning_is_not_output(
    mistral_runtime, monkeypatch, api_kind
):
    adapter = new_adapter()
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
        "prompt_tokens_details": {"cached_tokens": 200},
    }
    content = [
        {"type": "thinking", "thinking": [{"type": "text", "text": "Private thought."}]},
        {"type": "text", "text": "Answer."},
    ]
    result, _, _ = invoke(adapter, api_kind, monkeypatch, usage=usage, content=content)

    input_rate, cache_rate, output_rate = rates(adapter)
    assert input_rate * 1_000_000 == pytest.approx(0.15)
    assert cache_rate * 1_000_000 == pytest.approx(0.03)
    assert output_rate * 1_000_000 == pytest.approx(0.50)
    assert result.usage.input_tokens == 1000
    assert result.usage.cached_tokens == 200
    assert result.usage.output_tokens == 500
    assert result.usage.total_tokens == 1500
    assert result.content == "Answer."
    assert [item.text for item in result.reasoning_events] == ["Private thought."]
    assert result.cost_input == pytest.approx(800 * input_rate + 200 * cache_rate)
    assert result.cost_output == pytest.approx(500 * output_rate)
    assert result.cost_total == pytest.approx(0.000376)


@pytest.mark.integration
@pytest.mark.parametrize("api_kind", API_KINDS)
@pytest.mark.parametrize(
    ("cache_detail", "expected_cache"),
    [
        ({"cached_tokens": 0}, 0),
        ({"cached_tokens": 200}, 200),
        ({}, None),
        (None, None),
    ],
    ids=["explicit-zero", "positive", "missing-count", "missing-details"],
)
def test_flash_cache_usage_states_preserve_unknown_costs(
    mistral_runtime, monkeypatch, api_kind, cache_detail, expected_cache
):
    adapter = new_adapter()
    usage = {"prompt_tokens": 1000, "completion_tokens": 500}
    if cache_detail is not None:
        usage["prompt_tokens_details"] = cache_detail
    result, _, _ = invoke(adapter, api_kind, monkeypatch, usage=usage)

    input_rate, cache_rate, output_rate = rates(adapter)
    assert result.usage.cached_tokens == expected_cache
    assert result.usage.total_tokens == 1500  # derived only from reported components
    assert result.cost_output == pytest.approx(500 * output_rate)
    if expected_cache is None:
        assert result.cost_input is None
        assert result.cost_total is None
    else:
        assert result.cost_input == pytest.approx((1000 - expected_cache) * input_rate + expected_cache * cache_rate)
        assert result.cost_total == pytest.approx(result.cost_input + result.cost_output)


@pytest.mark.integration
@pytest.mark.parametrize("api_kind", API_KINDS)
@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "total_tokens"])
def test_flash_malformed_counts_are_unknown_and_never_imputed(
    mistral_runtime, monkeypatch, api_kind, field
):
    adapter = new_adapter()
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    usage[field] = "not-a-count"
    result, _, _ = invoke(adapter, api_kind, monkeypatch, usage=usage)

    assert getattr(result.usage, {
        "prompt_tokens": "input_tokens",
        "completion_tokens": "output_tokens",
        "total_tokens": "total_tokens",
    }[field]) == (1500 if field == "total_tokens" else None)
    if field == "prompt_tokens":
        assert result.cost_input is None
        assert result.cost_total is None
    elif field == "completion_tokens":
        assert result.cost_output is None
        assert result.cost_total is None
    else:
        # Missing/invalid totals derive from valid components, but totals never
        # replace or impute component counts.
        assert result.usage.input_tokens == 1000
        assert result.usage.output_tokens == 500
        assert result.usage.total_tokens == 1500
        assert result.cost_total is not None


@pytest.mark.integration
@pytest.mark.parametrize("api_kind", API_KINDS)
@pytest.mark.parametrize(
    ("usage", "input_tokens", "output_tokens", "total_tokens"),
    [
        ({"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 0}}, 1000, None, None),
        ({"completion_tokens": 500}, None, 500, None),
        ({"total_tokens": 1500}, None, None, 1500),
        ({"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 999, "prompt_tokens_details": {"cached_tokens": 0}}, 1000, 500, 999),
        (None, None, None, None),
    ],
    ids=["missing-output", "missing-input", "total-only", "contradictory-total", "missing-usage"],
)
def test_flash_incomplete_or_contradictory_usage_does_not_invent_components(
    mistral_runtime, monkeypatch, api_kind, usage, input_tokens, output_tokens, total_tokens
):
    adapter = new_adapter()
    result, _, _ = invoke(adapter, api_kind, monkeypatch, usage=usage)
    input_rate, _, output_rate = rates(adapter)

    if usage is None:
        assert result.usage is None
        assert result.cost_input is None
        assert result.cost_output is None
        assert result.cost_total is None
    else:
        assert result.usage is not None
        assert result.usage.input_tokens == input_tokens
        assert result.usage.output_tokens == output_tokens
        assert result.usage.total_tokens == total_tokens
        if input_tokens is None:
            assert result.cost_input is None
            assert result.cost_output is None
            assert result.cost_total is None
        else:
            assert result.cost_input == pytest.approx(input_tokens * input_rate)
            if output_tokens is None:
                assert result.cost_output is None
                assert result.cost_total is None
            else:
                assert result.cost_output == pytest.approx(output_tokens * output_rate)
                assert result.cost_total == pytest.approx(result.cost_input + result.cost_output)
        if input_tokens is not None and output_tokens is not None:
            # The provider's contradictory total remains reported as usage,
            # while cost uses only the independently reported components.
            assert result.usage.total_tokens == 999


@pytest.mark.integration
@pytest.mark.parametrize("api_kind", API_KINDS)
def test_flash_cache_count_above_input_makes_input_and_total_cost_unknown(
    mistral_runtime, monkeypatch, api_kind
):
    adapter = new_adapter()
    usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 101},
    }
    result, _, _ = invoke(adapter, api_kind, monkeypatch, usage=usage)
    assert result.usage.cached_tokens == 101
    assert result.cost_input is None
    assert result.cost_output is not None
    assert result.cost_total is None


@pytest.mark.integration
@pytest.mark.parametrize("api_kind", API_KINDS)
@pytest.mark.parametrize("cached_tokens", ["200", -1, True, 1.5, object()])
def test_flash_invalid_cached_count_is_rejected(mistral_runtime, monkeypatch, api_kind, cached_tokens):
    adapter = new_adapter()
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "prompt_tokens_details": {"cached_tokens": cached_tokens},
    }
    _, _ = configure(adapter, monkeypatch, result=response(usage), events=[stream_event(usage=usage)])
    with pytest.raises(LLMAPIClientError, match="cached_tokens"):
        if api_kind == "chat":
            adapter.chat(MESSAGES)
        elif api_kind == "achat":
            asyncio.run(adapter.achat(MESSAGES))
        elif api_kind == "stream_chat":
            list(adapter.stream_chat(MESSAGES))
        else:
            async def collect():
                async for _ in adapter.astream_chat(MESSAGES):
                    pass
            asyncio.run(collect())


@pytest.mark.integration
@pytest.mark.parametrize("api_kind", ["stream_chat", "astream_chat"])
def test_flash_stream_uses_final_event_usage_only_and_interruption_has_no_completion(
    mistral_runtime, monkeypatch, api_kind
):
    adapter = new_adapter()
    final_usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
        "prompt_tokens_details": {"cached_tokens": 200},
    }
    final_events = [
        stream_event(content="Answer", finish_reason="stop"),
        SSEEvent(event=None, data={"id": "final-usage", "model": WIRE_MODEL, "choices": [], "usage": final_usage}),
    ]
    _, _ = configure(adapter, monkeypatch, result={}, events=final_events)
    completed = []
    if api_kind == "stream_chat":
        list(adapter.stream_chat(MESSAGES, on_done=completed.append))
    else:
        async def collect():
            async def on_done(value):
                completed.append(value)
            async for _ in adapter.astream_chat(MESSAGES, on_done=on_done):
                pass
        asyncio.run(collect())
    assert len(completed) == 1
    assert completed[0].usage.input_tokens == 1000
    assert completed[0].usage.output_tokens == 500
    assert completed[0].cost_total == pytest.approx(0.000376)

    # A transport failure after partial output must not finalize a response or
    # pass incomplete usage as a completed result.
    interrupted_events = [stream_event(content="Partial", finish_reason=None)]
    def failing_events():
        yield interrupted_events[0]
        raise LLMAPIClientError(detail="synthetic interrupted stream")
    configure(adapter, monkeypatch, result={}, events=failing_events())
    completed.clear()
    if api_kind == "stream_chat":
        with pytest.raises(LLMAPIClientError, match="interrupted"):
            list(adapter.stream_chat(MESSAGES, on_done=completed.append))
    else:
        async def collect_interrupted():
            with pytest.raises(LLMAPIClientError, match="interrupted"):
                async for _ in adapter.astream_chat(MESSAGES, on_done=completed.append):
                    pass
        asyncio.run(collect_interrupted())
    assert completed == []

