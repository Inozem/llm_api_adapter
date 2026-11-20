"""Observable usage and caller-tariff contract for generic Messages mode."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
from pathlib import Path
import socket

import httpx
import pytest
import requests
import requests_mock

from src.llm_api_adapter.llm_registry.llm_registry import Pricing, PricingTier
from src.llm_api_adapter.models.messages.chat_message import UserMessage


ROOT = "https://messages.example.invalid/proxy/v1"
URL = ROOT + "/messages"
MODEL = "vendor/usage-contract-model"
KEY = "synthetic-usage-key"
LANES = (
    "requests-chat",
    "requests-stream",
    "httpx-chat",
    "httpx-stream",
    "async-chat",
    "async-stream",
)
_ABSENT = object()


def _adapter(*, pricing=None, model=MODEL, transport="requests"):
    from src.llm_api_adapter.adapters.anthropic_compatible.adapter import (
        AnthropicCompatibleAdapter,
    )

    return AnthropicCompatibleAdapter(
        api_key=KEY,
        model=model,
        base_url=ROOT,
        transport=transport,
        pricing=pricing,
    )


def _tiered_pricing(*, write_rate=0.75):
    return Pricing.from_dict(
        [
            {
                "up_to_prompt_tokens": 129,
                "input_per_1m": 1,
                "output_per_1m": 2,
                "cache_read_input_per_1m": 0.25,
                "cache_write_input_per_1m": write_rate,
            },
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 3,
                "output_per_1m": 4,
                "cache_read_input_per_1m": 0.5,
                "cache_write_input_per_1m": write_rate,
            },
        ],
        currency="EUR",
    )


def _message(usage=_ABSENT, *, model=MODEL):
    result = {
        "id": "msg_usage_contract",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "Done"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
    }
    if usage is not _ABSENT:
        result["usage"] = usage
    return result


def _events(usage=_ABSENT, *, model=MODEL):
    """Build a valid completion with usage split between start and final delta."""
    from tests.fixtures.anthropic_compatible import messages_event_trace

    if usage is _ABSENT:
        return messages_event_trace(
            blocks=[{"type": "text", "text": "Done"}], model=model
        )
    if usage is None:
        usage_start, usage_delta = None, None
    else:
        usage_start = {key: usage[key] for key in ("input_tokens",) if key in usage}
        usage_delta = {
            key: value for key, value in usage.items() if key != "input_tokens"
        }
    return messages_event_trace(
        blocks=[{"type": "text", "text": "Done"}],
        usage_start=usage_start,
        usage_delta=usage_delta,
        model=model,
    )


def _sse_body(events):
    chunks = []
    for event in events:
        event_name = f"event: {event.event}\n" if event.event else ""
        chunks.append(event_name + "data: " + json.dumps(event.data) + "\n\n")
    return "".join(chunks).encode("utf-8")


def _httpx_sync_client(monkeypatch, handler):
    real_client = httpx.Client
    clients = []

    def make_client(*args, **kwargs):
        kwargs.update(transport=httpx.MockTransport(handler), trust_env=False)
        client = real_client(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", make_client)
    return clients


def _httpx_async_client(monkeypatch, handler):
    real_client = httpx.AsyncClient
    clients = []

    def make_client(*args, **kwargs):
        kwargs.update(transport=httpx.MockTransport(handler), trust_env=False)
        client = real_client(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", make_client)
    return clients


@contextmanager
def _mock_endpoint(lane, response_body, monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("usage/pricing tests must not resolve network hosts")
        ),
    )
    stream = lane.endswith("stream")
    requests_seen = []
    clients = []
    if lane.startswith("requests"):
        server = requests_mock.Mocker(real_http=False)
        server.start()
        options = (
            {"content": response_body, "headers": {"content-type": "text/event-stream"}}
            if stream
            else {"json": response_body}
        )
        server.register_uri("POST", URL, **options)
        try:
            yield server.request_history, clients
        finally:
            server.stop()
        return

    if lane.startswith("httpx"):
        def handler(request):
            requests_seen.append(request)
            return (
                httpx.Response(
                    200,
                    content=response_body,
                    headers={"content-type": "text/event-stream"},
                    request=request,
                )
                if stream
                else httpx.Response(200, json=response_body, request=request)
            )

        clients = _httpx_sync_client(monkeypatch, handler)
    else:
        async def handler(request):
            requests_seen.append(request)
            return (
                httpx.Response(
                    200,
                    content=response_body,
                    headers={"content-type": "text/event-stream"},
                    request=request,
                )
                if stream
                else httpx.Response(200, json=response_body, request=request)
            )

        clients = _httpx_async_client(monkeypatch, handler)
        monkeypatch.setattr(
            requests,
            "post",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("async Messages must not fall back to Requests")
            ),
        )
    try:
        yield requests_seen, clients
    finally:
        for client in clients:
            if lane.startswith("async"):
                assert client.is_closed
            else:
                assert client.is_closed


def _invoke(adapter, lane):
    stream = lane.endswith("stream")
    async_mode = lane.startswith("async")
    completed = []
    messages = [UserMessage("hi")]
    kwargs = {"max_tokens": 32}
    if stream:
        kwargs["on_done"] = completed.append

    if async_mode:
        async def run():
            if stream:
                output = [
                    chunk
                    async for chunk in adapter.astream_chat(messages, **kwargs)
                ]
                return output, completed[0]
            return [], await adapter.achat(messages, **kwargs)

        output, response = asyncio.run(run())
    elif stream:
        output = list(adapter.stream_chat(messages, **kwargs))
        response = completed[0]
    else:
        output = []
        response = adapter.chat(messages, **kwargs)
    return output, response


def _assert_wire(request, lane, *, model=MODEL):
    assert request.method == "POST"
    assert str(request.url) == URL
    headers = {name.lower(): value for name, value in request.headers.items()}
    assert headers["x-api-key"] == KEY
    assert headers["content-type"] == "application/json"
    assert headers["anthropic-version"] == "2023-06-01"
    raw_body = request.content if isinstance(request, httpx.Request) else request.body
    body = json.loads(raw_body)
    expected = {
        "model": model,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if lane.endswith("stream"):
        expected["stream"] = True
    assert body == expected


def _usage_for_case(case):
    values = {
        "complete": {
            "input_tokens": 100,
            "output_tokens": 5,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 20},
        },
        "partial-read-missing": {"input_tokens": 100, "output_tokens": 5,
                                  "cache_creation_input_tokens": 10},
        "partial-write-missing": {"input_tokens": 100, "output_tokens": 5,
                                   "cache_read_input_tokens": 20},
        "missing-input": {"output_tokens": 5, "cache_read_input_tokens": 0,
                          "cache_creation_input_tokens": 0},
        "null-input": {"input_tokens": None, "output_tokens": 5,
                       "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        "bool-input": {"input_tokens": True, "output_tokens": 5,
                       "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        "negative-input": {"input_tokens": -1, "output_tokens": 5,
                           "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        "fractional-input": {"input_tokens": 1.5, "output_tokens": 5,
                             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        "missing-output": {"input_tokens": 100, "cache_read_input_tokens": 0,
                           "cache_creation_input_tokens": 0},
        "null-output": {"input_tokens": 100, "output_tokens": None,
                         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        "negative-output": {"input_tokens": 100, "output_tokens": -1,
                            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        "fractional-output": {"input_tokens": 100, "output_tokens": 1.5,
                              "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        "null-read": {"input_tokens": 100, "output_tokens": 5,
                      "cache_read_input_tokens": None,
                      "cache_creation_input_tokens": 10},
        "bool-read": {"input_tokens": 100, "output_tokens": 5,
                      "cache_read_input_tokens": True, "cache_creation_input_tokens": 0},
        "negative-read": {"input_tokens": 100, "output_tokens": 5,
                          "cache_read_input_tokens": -1, "cache_creation_input_tokens": 0},
        "fractional-read": {"input_tokens": 100, "output_tokens": 5,
                            "cache_read_input_tokens": 1.5, "cache_creation_input_tokens": 0},
        "null-write": {"input_tokens": 100, "output_tokens": 5,
                       "cache_read_input_tokens": 0, "cache_creation_input_tokens": None},
        "bool-write": {"input_tokens": 100, "output_tokens": 5,
                       "cache_read_input_tokens": 0, "cache_creation_input_tokens": True},
        "negative-write": {"input_tokens": 100, "output_tokens": 5,
                           "cache_read_input_tokens": 0, "cache_creation_input_tokens": -1},
        "fractional-write": {"input_tokens": 100, "output_tokens": 5,
                             "cache_read_input_tokens": 0, "cache_creation_input_tokens": 1.5},
        "invalid-output": {"input_tokens": 100, "output_tokens": True,
                           "cache_read_input_tokens": 0,
                           "cache_creation_input_tokens": 0},
        "zero": {"input_tokens": 0, "output_tokens": 0,
                 "cache_read_input_tokens": 0,
                 "cache_creation_input_tokens": 0},
        "zero-cache": {"input_tokens": 100, "output_tokens": 5,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0},
        "no-usage": _ABSENT,
        "null-usage": None,
        "no-tariff": {
            "input_tokens": 100,
            "output_tokens": 5,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 10,
        },
        "no-tariff-native-name": {
            "input_tokens": 100,
            "output_tokens": 5,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 10,
        },
        "unpriced-write": {"input_tokens": 100, "output_tokens": 5,
                            "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 2},
        "unpriced-read": {"input_tokens": 100, "output_tokens": 5,
                          "cache_read_input_tokens": 2,
                          "cache_creation_input_tokens": 0},
        "alias-match": {"input_tokens": 100, "output_tokens": 5,
                        "cache_read_input_tokens": 20,
                        "cache_creation_input_tokens": 0,
                        "prompt_tokens_details": {"cached_tokens": 20}},
        "alias-conflict": {"input_tokens": 100, "output_tokens": 5,
                           "cache_read_input_tokens": 20,
                           "cache_creation_input_tokens": 0,
                           "prompt_tokens_details": {"cached_tokens": 21}},
    }
    return values[case]


def _price_for_case(case):
    if case in {"no-usage", "null-usage", "no-tariff", "no-tariff-native-name"}:
        return None
    if case in {"unpriced-write", "unpriced-read"}:
        return Pricing.from_dict(
            [{"up_to_prompt_tokens": None, "input_per_1m": 3, "output_per_1m": 4}],
            currency="USD",
        )
    if case == "zero":
        pricing = Pricing(
            tiers=(PricingTier(None, 0.0, 0.0, 0.0, 0.0),), currency="USD"
        )
        pricing.set_in_per_1m(0)
        pricing.set_out_per_1m(0)
        pricing.set_currency("USD")
        return pricing
    return _tiered_pricing()


def _expected_usage(case):
    """Expected public Usage fields for each controlled provider payload."""
    values = {
        "complete": (130, 5, 135, 20, 10),
        "partial-read-missing": (None, 5, None, None, 10),
        "partial-write-missing": (None, 5, None, 20, None),
        "null-read": (None, 5, None, None, 10),
        "missing-input": (None, 5, None, 0, 0),
        "null-input": (None, 5, None, 0, 0),
        "bool-input": (None, 5, None, 0, 0),
        "negative-input": (None, 5, None, 0, 0),
        "fractional-input": (None, 5, None, 0, 0),
        "missing-output": (100, None, None, 0, 0),
        "null-output": (100, None, None, 0, 0),
        "invalid-output": (100, None, None, 0, 0),
        "negative-output": (100, None, None, 0, 0),
        "fractional-output": (100, None, None, 0, 0),
        "bool-read": (None, 5, None, None, 0),
        "negative-read": (None, 5, None, None, 0),
        "fractional-read": (None, 5, None, None, 0),
        "null-write": (None, 5, None, 0, None),
        "bool-write": (None, 5, None, 0, None),
        "negative-write": (None, 5, None, 0, None),
        "fractional-write": (None, 5, None, 0, None),
        "zero": (0, 0, 0, 0, 0),
        "zero-cache": (100, 5, 105, 0, 0),
        "no-usage": None,
        "null-usage": None,
        "no-tariff": (130, 5, 135, 20, 10),
        "no-tariff-native-name": (130, 5, 135, 20, 10),
        "unpriced-write": (102, 5, 107, 0, 2),
        "unpriced-read": (102, 5, 107, 2, 0),
        "alias-match": (120, 5, 125, 20, 0),
        "alias-conflict": (None, 5, None, None, 0),
    }
    expected = values[case]
    if expected is None:
        return None
    return expected


@pytest.mark.unit
@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize(
    "case",
    [
        "complete", "partial-read-missing", "partial-write-missing", "null-read",
        "missing-input", "null-input", "bool-input", "negative-input", "fractional-input",
        "missing-output", "null-output", "invalid-output", "negative-output",
        "fractional-output", "bool-read", "negative-read", "fractional-read",
        "null-write", "bool-write", "negative-write", "fractional-write",
        "zero", "no-usage", "null-usage", "unpriced-write", "unpriced-read",
        "zero-cache", "alias-match", "alias-conflict", "no-tariff",
        "no-tariff-native-name",
    ],
)
def test_usage_and_tariff_results_through_each_real_mocked_transport(lane, case, monkeypatch):
    usage = _usage_for_case(case)
    stream = lane.endswith("stream")
    model = (
        "claude-sonnet-4-5-20250929"
        if case == "no-tariff-native-name"
        else MODEL
    )
    body = (
        _events(usage, model=model)
        if stream
        else _message(usage, model=model)
    )
    if stream:
        body = _sse_body(body)
    transport = "httpx" if lane.startswith("httpx") else "requests"
    adapter = _adapter(
        pricing=_price_for_case(case), transport=transport, model=model
    )

    with _mock_endpoint(lane, body, monkeypatch) as (requests_seen, clients):
        output, response = _invoke(adapter, lane)

    assert len(requests_seen) == 1
    _assert_wire(requests_seen[0], lane, model=model)
    if lane.endswith("stream"):
        assert output == ["Done"]

    expected_usage = _expected_usage(case)
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

    if case == "complete":
        assert response.usage.input_tokens == 130
        assert response.usage.output_tokens == 5
        assert response.usage.cached_tokens == 20
        assert response.usage.cache_write_tokens == 10
        assert response.cost_input == pytest.approx((100 * 3 + 20 * 0.5 + 10 * 0.75) / 1_000_000)
        assert response.cost_output == pytest.approx(5 * 4 / 1_000_000)
        assert response.cost_total == pytest.approx(0.0003375)
        assert response.currency == "EUR"
    elif case in {
        "partial-read-missing", "partial-write-missing", "null-read", "alias-conflict",
        "missing-input", "null-input", "bool-input", "negative-input", "fractional-input",
        "bool-read", "negative-read", "fractional-read", "null-write", "bool-write",
        "negative-write", "fractional-write",
    }:
        assert response.usage.input_tokens is None
        assert response.cost_input is None
        assert response.cost_total is None
    elif case in {"missing-output", "null-output", "invalid-output",
                  "negative-output", "fractional-output"}:
        assert response.usage.output_tokens is None
        assert response.cost_output is None
        assert response.cost_total is None
    elif case == "zero":
        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0
        assert response.cost_input == response.cost_output == response.cost_total == 0
        assert response.currency == "USD"
    elif case == "zero-cache":
        assert response.usage.input_tokens == 100
        assert response.usage.cached_tokens == 0
        assert response.usage.cache_write_tokens == 0
    elif case in {"no-usage", "null-usage"}:
        assert response.usage is None
        assert response.cost_input is response.cost_output is response.cost_total is None
        assert response.currency is None
    elif case in {"unpriced-write", "unpriced-read"}:
        assert response.usage.input_tokens == 102
        assert response.cost_input is None
        assert response.cost_output == pytest.approx(5 * 4 / 1_000_000)
        assert response.cost_total is None
        assert response.currency == "USD"
    elif case == "alias-match":
        assert response.usage.input_tokens == 120
        assert response.usage.cached_tokens == 20
    elif case == "no-tariff":
        assert response.usage.input_tokens == 130
        assert response.cost_input is response.cost_output is response.cost_total is None
        assert response.currency is None
    elif case == "no-tariff-native-name":
        assert response.usage.input_tokens == 130
        assert response.cost_input is response.cost_output is response.cost_total is None
        assert response.currency is None


@pytest.mark.unit
@pytest.mark.parametrize("lane", LANES)
@pytest.mark.parametrize(("ordinary", "read", "write", "expected_input_rate"), [
    (98, 20, 10, 1), (99, 20, 10, 1), (100, 20, 10, 3),
])
def test_tier_boundary_uses_inclusive_reported_input_on_every_lane(
    lane, ordinary, read, write, expected_input_rate, monkeypatch
):
    usage = {
        "input_tokens": ordinary,
        "output_tokens": 5,
        "cache_read_input_tokens": read,
        "cache_creation_input_tokens": write,
    }
    body = _events(usage) if lane.endswith("stream") else _message(usage)
    if lane.endswith("stream"):
        body = _sse_body(body)
    transport = "httpx" if lane.startswith("httpx") else "requests"
    adapter = _adapter(pricing=_tiered_pricing(), transport=transport)

    with _mock_endpoint(lane, body, monkeypatch) as (requests_seen, _clients):
        _output, response = _invoke(adapter, lane)

    assert len(requests_seen) == 1
    assert (
        response.usage.input_tokens,
        response.usage.output_tokens,
        response.usage.total_tokens,
        response.usage.cached_tokens,
        response.usage.cache_write_tokens,
    ) == (ordinary + read + write, 5, ordinary + read + write + 5, read, write)
    expected = (ordinary * expected_input_rate + read * (0.25 if expected_input_rate == 1 else 0.5)
                + write * 0.75) / 1_000_000
    assert response.cost_input == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize("lane", LANES)
def test_finite_uncovered_tier_does_not_extrapolate(lane, monkeypatch):
    pricing = Pricing.from_dict(
        [{"up_to_prompt_tokens": 129, "input_per_1m": 1, "output_per_1m": 2}],
        currency="CAD",
    )
    usage = {
        "input_tokens": 100,
        "output_tokens": 5,
        "cache_read_input_tokens": 20,
        "cache_creation_input_tokens": 10,
    }
    body = _events(usage) if lane.endswith("stream") else _message(usage)
    if lane.endswith("stream"):
        body = _sse_body(body)
    transport = "httpx" if lane.startswith("httpx") else "requests"
    adapter = _adapter(pricing=pricing, transport=transport)

    with _mock_endpoint(lane, body, monkeypatch) as (requests_seen, _clients):
        _output, response = _invoke(adapter, lane)

    assert len(requests_seen) == 1
    assert response.usage.input_tokens == 130
    assert response.cost_input is response.cost_output is response.cost_total is None
    assert response.currency is None


@pytest.mark.unit
@pytest.mark.parametrize("lane", LANES)
def test_invalid_mutated_or_assigned_tariff_fails_before_dispatch(lane, monkeypatch):
    import math

    from src.llm_api_adapter.errors.config_errors import LLMConfigError

    invalid = Pricing(
        tiers=(PricingTier(None, 1e-6, 2e-6),), currency="USD"
    )
    transport = "httpx" if lane.startswith("httpx") else "requests"
    adapter = _adapter(transport=transport)
    adapter.pricing = invalid
    object.__setattr__(invalid.tiers[0], "in_per_token", math.inf)
    body = _events({"input_tokens": 1, "output_tokens": 1,
                    "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})
    if lane.endswith("stream"):
        body = _sse_body(body)
    else:
        body = _message({"input_tokens": 1, "output_tokens": 1,
                         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})

    with _mock_endpoint(lane, body, monkeypatch) as (requests_seen, _clients):
        with pytest.raises(LLMConfigError):
            _invoke(adapter, lane)
        assert requests_seen == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "rate_field",
    [
        "in_per_token",
        "out_per_token",
        "cache_read_in_per_token",
        "cache_write_in_per_token",
    ],
)
@pytest.mark.parametrize("invalid_rate", [-1.0, float("nan"), float("inf"), True])
def test_invalid_direct_numeric_rates_fail_during_generic_construction(
    rate_field, invalid_rate
):
    from src.llm_api_adapter.errors.config_errors import LLMConfigError

    rates = {
        "in_per_token": 1e-6,
        "out_per_token": 2e-6,
        "cache_read_in_per_token": 0.5e-6,
        "cache_write_in_per_token": 0.75e-6,
    }
    rates[rate_field] = invalid_rate
    invalid = Pricing(
        tiers=(PricingTier(None, **rates),),
        currency="USD",
    )

    with pytest.raises(LLMConfigError):
        _adapter(pricing=invalid)


@pytest.mark.unit
@pytest.mark.parametrize("lane", LANES)
def test_file_loaded_dictionary_and_direct_tariffs_price_public_operations(
    lane, tmp_path, monkeypatch
):
    pricing_file = Path(tmp_path / "caller-rates.json")
    pricing_file.write_text(
        json.dumps({
            "currency": "CAD",
            "pricing_tiers": [{
                "up_to_prompt_tokens": None,
                "input_per_1m": 2,
                "output_per_1m": 5,
                "cache_read_input_per_1m": 0,
                "cache_write_input_per_1m": 0,
            }],
        }),
        encoding="utf-8",
    )
    rates = json.loads(Path(pricing_file).read_text(encoding="utf-8"))
    dictionary = Pricing.from_dict(rates["pricing_tiers"], currency=rates["currency"])
    direct = Pricing(
        tiers=(PricingTier(None, 2 / 1_000_000, 5 / 1_000_000, 0, 0),),
        currency="CAD",
    )
    usage = {
        "input_tokens": 3,
        "output_tokens": 2,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    stream = lane.endswith("stream")
    body = _events(usage) if stream else _message(usage)
    if stream:
        body = _sse_body(body)
    transport = "httpx" if lane.startswith("httpx") else "requests"
    adapter = _adapter(pricing=dictionary, transport=transport)

    with _mock_endpoint(lane, body, monkeypatch) as (requests_seen, _clients):
        _output, from_dictionary = _invoke(adapter, lane)
        adapter.pricing = direct
        _output, from_direct = _invoke(adapter, lane)

    assert len(requests_seen) == 2
    _assert_wire(requests_seen[0], lane)
    _assert_wire(requests_seen[1], lane)
    for response in (from_dictionary, from_direct):
        assert response.usage.input_tokens == 3
        assert response.cost_input == pytest.approx(6 / 1_000_000)
        assert response.cost_output == pytest.approx(10 / 1_000_000)
        assert response.cost_total == pytest.approx(16 / 1_000_000)
        assert response.currency == "CAD"

