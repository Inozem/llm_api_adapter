from unittest.mock import AsyncMock, patch

import pytest

from src.llm_api_adapter.adapters.openai_adapter import OpenAIAdapter
from src.llm_api_adapter.llm_registry.llm_registry import Pricing
from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent
from src.llm_api_adapter.models.messages.chat_message import UserMessage
from src.llm_api_adapter.models.responses.chat_response import (
    ChatResponse,
    CostLineItem,
    Usage,
)


def _tiered_pricing() -> Pricing:
    return Pricing.from_dict(
        [
            {
                "up_to_prompt_tokens": 200,
                "input_per_1m": 1.0,
                "output_per_1m": 2.0,
            },
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 3.0,
                "output_per_1m": 4.0,
            },
        ],
        currency="USD",
    )


def _adapter() -> OpenAIAdapter:
    adapter = OpenAIAdapter(api_key="test_api_key", model="gpt-5")
    adapter.pricing = _tiered_pricing()
    return adapter


def _cache_adapter(*, include_read_rate: bool = True) -> OpenAIAdapter:
    adapter = OpenAIAdapter(api_key="test_api_key", model="gpt-5.6-sol")
    first_tier = {
        "up_to_prompt_tokens": 100,
        "input_per_1m": 10.0,
        "output_per_1m": 8.0,
        "cache_write_input_per_1m": 4.0,
    }
    if include_read_rate:
        first_tier["cache_read_input_per_1m"] = 2.0
    second_tier = {
        "up_to_prompt_tokens": None,
        "input_per_1m": 30.0,
        "output_per_1m": 10.0,
        "cache_write_input_per_1m": 6.0,
    }
    if include_read_rate:
        second_tier["cache_read_input_per_1m"] = 5.0
    adapter.pricing = Pricing.from_dict([first_tier, second_tier], currency="USD")
    return adapter


def _response(
    input_tokens: int,
    *,
    output_tokens: int = 10,
    include_usage: bool = True,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
) -> dict:
    response = {
        "id": "resp_123",
        "model": "gpt-5",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Done"}],
            }
        ],
    }
    if include_usage:
        response["usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        cache_details = {}
        if cache_read_tokens is not None:
            cache_details["cached_tokens"] = cache_read_tokens
        if cache_write_tokens is not None:
            cache_details["cache_write_tokens"] = cache_write_tokens
        if cache_details:
            response["usage"]["input_tokens_details"] = cache_details
    return response


def _stream_events(
    input_tokens: int,
    *,
    include_usage: bool = True,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
) -> list[SSEEvent]:
    return [
        SSEEvent(
            event="response.output_text.delta",
            data={
                "type": "response.output_text.delta",
                "delta": "Done",
            },
        ),
        SSEEvent(
            event="response.completed",
            data={
                "type": "response.completed",
                "response": _response(
                    input_tokens,
                    include_usage=include_usage,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                ),
            },
        ),
    ]


async def _async_stream_events(
    input_tokens: int,
    *,
    include_usage: bool = True,
    cache_read_tokens: int | None = None,
    cache_write_tokens: int | None = None,
):
    for event in _stream_events(
        input_tokens,
        include_usage=include_usage,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    ):
        yield event


def _assert_costs(
    response,
    *,
    input_tokens: int,
    input_per_1m: float,
    output_per_1m: float,
) -> None:
    expected_input = input_tokens * input_per_1m / 1_000_000
    expected_output = 10 * output_per_1m / 1_000_000

    assert response.cost_input == pytest.approx(expected_input)
    assert response.cost_output == pytest.approx(expected_output)
    assert response.cost_total == pytest.approx(expected_input + expected_output)
    assert response.currency == "USD"


def _assert_cache_costs(response) -> None:
    assert response.usage.input_tokens == 101
    assert response.usage.cached_tokens == 20
    assert response.usage.cache_write_tokens == 10
    assert response.cost_input == pytest.approx(0.00229)
    assert response.cost_output == pytest.approx(0.0001)
    assert response.cost_total == pytest.approx(0.00239)
    assert response.currency == "USD"


def _apply_cache_pricing(response: ChatResponse, *, read_rate=2e-6, write_rate=4e-6):
    response.apply_pricing(
        price_input_per_token=10e-6,
        price_output_per_token=8e-6,
        currency="USD",
        price_cache_read_per_token=read_rate,
        price_cache_write_per_token=write_rate,
    )


def _direct_cache_response(
    *,
    input_tokens=100,
    output_tokens=5,
    total_tokens=105,
    cached_tokens=20,
    cache_write_tokens=10,
) -> ChatResponse:
    return ChatResponse(
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
        )
    )


def _ocr_item(*, currency="USD", quantity=2) -> CostLineItem:
    return CostLineItem(
        operation="ocr",
        model="mistral-ocr-4-1",
        unit="page",
        quantity=quantity,
        rate=0.004,
        currency=currency,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("input_tokens", "input_per_1m", "output_per_1m"),
    [
        (199, 1.0, 2.0),
        (200, 1.0, 2.0),
        (201, 3.0, 4.0),
    ],
)
def test_tiered_pricing_matches_sync_chat_and_stream(
    input_tokens,
    input_per_1m,
    output_per_1m,
):
    adapter = _adapter()

    with patch.object(
        OpenAISyncClient,
        "complete",
        return_value=_response(input_tokens),
    ):
        chat_response = adapter.chat([UserMessage("hi")])

    completed = []
    with patch.object(
        OpenAISyncClient,
        "stream",
        return_value=iter(_stream_events(input_tokens)),
    ):
        assert list(adapter.stream_chat([UserMessage("hi")], on_done=completed.append)) == [
            "Done"
        ]

    assert len(completed) == 1
    _assert_costs(
        chat_response,
        input_tokens=input_tokens,
        input_per_1m=input_per_1m,
        output_per_1m=output_per_1m,
    )
    _assert_costs(
        completed[0],
        input_tokens=input_tokens,
        input_per_1m=input_per_1m,
        output_per_1m=output_per_1m,
    )


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    ("input_tokens", "input_per_1m", "output_per_1m"),
    [
        (199, 1.0, 2.0),
        (200, 1.0, 2.0),
        (201, 3.0, 4.0),
    ],
)
async def test_tiered_pricing_matches_async_chat_and_stream(
    input_tokens,
    input_per_1m,
    output_per_1m,
):
    adapter = _adapter()

    with patch.object(
        OpenAIAsyncClient,
        "complete",
        new=AsyncMock(return_value=_response(input_tokens)),
    ):
        chat_response = await adapter.achat([UserMessage("hi")])

    completed = []
    with patch.object(
        OpenAIAsyncClient,
        "stream",
        return_value=_async_stream_events(input_tokens),
    ):
        output = [
            text
            async for text in adapter.astream_chat(
                [UserMessage("hi")],
                on_done=completed.append,
            )
        ]

    assert output == ["Done"]
    assert len(completed) == 1
    _assert_costs(
        chat_response,
        input_tokens=input_tokens,
        input_per_1m=input_per_1m,
        output_per_1m=output_per_1m,
    )
    _assert_costs(
        completed[0],
        input_tokens=input_tokens,
        input_per_1m=input_per_1m,
        output_per_1m=output_per_1m,
    )


@pytest.mark.unit
def test_cache_pricing_matches_sync_chat_and_stream_using_total_input_tier():
    adapter = _cache_adapter()
    response = _response(
        101,
        cache_read_tokens=20,
        cache_write_tokens=10,
    )

    with patch.object(OpenAISyncClient, "complete", return_value=response):
        chat_response = adapter.chat([UserMessage("hi")])

    completed = []
    with patch.object(
        OpenAISyncClient,
        "stream",
        return_value=iter(
            _stream_events(
                101,
                cache_read_tokens=20,
                cache_write_tokens=10,
            )
        ),
    ):
        assert list(adapter.stream_chat([UserMessage("hi")], on_done=completed.append)) == [
            "Done"
        ]

    assert len(completed) == 1
    _assert_cache_costs(chat_response)
    _assert_cache_costs(completed[0])


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cache_pricing_matches_async_chat_and_stream_using_total_input_tier():
    adapter = _cache_adapter()
    response = _response(
        101,
        cache_read_tokens=20,
        cache_write_tokens=10,
    )

    with patch.object(
        OpenAIAsyncClient,
        "complete",
        new=AsyncMock(return_value=response),
    ):
        chat_response = await adapter.achat([UserMessage("hi")])

    completed = []
    with patch.object(
        OpenAIAsyncClient,
        "stream",
        return_value=_async_stream_events(
            101,
            cache_read_tokens=20,
            cache_write_tokens=10,
        ),
    ):
        output = [
            text
            async for text in adapter.astream_chat(
                [UserMessage("hi")],
                on_done=completed.append,
            )
        ]

    assert output == ["Done"]
    assert len(completed) == 1
    _assert_cache_costs(chat_response)
    _assert_cache_costs(completed[0])


@pytest.mark.unit
@pytest.mark.parametrize(
    ("cache_read_tokens", "cache_write_tokens", "include_read_rate"),
    [
        (None, None, True),
        (80, 30, True),
        (20, 0, False),
    ],
)
def test_adapter_leaves_input_cost_unknown_for_unpriceable_cache_splits(
    cache_read_tokens,
    cache_write_tokens,
    include_read_rate,
):
    adapter = _cache_adapter(include_read_rate=include_read_rate)
    provider_response = _response(
        101,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )

    with patch.object(
        OpenAISyncClient,
        "complete",
        return_value=provider_response,
    ):
        response = adapter.chat([UserMessage("hi")])

    assert response.cost_input is None
    assert response.cost_output == pytest.approx(10 * 10e-6)
    assert response.cost_total is None


@pytest.mark.unit
def test_direct_pricing_keeps_existing_positional_call_compatible():
    response = ChatResponse(usage=Usage(100, 50, 150))

    response.apply_pricing(1e-6, 2e-6, "USD")

    assert response.cost_input == pytest.approx(100e-6)
    assert response.cost_output == pytest.approx(100e-6)
    assert response.cost_total == pytest.approx(200e-6)


@pytest.mark.unit
@pytest.mark.parametrize(
    (
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "cache_write_tokens",
        "read_rate",
        "write_rate",
        "expected_input",
        "expected_output",
        "expected_total",
    ),
    [
        (100, 5, 20, 10, 2e-6, 4e-6, 0.00078, 0.00004, 0.00082),
        (0, 0, 0, 0, 2e-6, 4e-6, 0, 0, 0),
        (100, 5, None, 10, 2e-6, 4e-6, None, 0.00004, None),
        (100, 5, 20, None, 2e-6, 4e-6, None, 0.00004, None),
        (100, 5, -1, 0, 2e-6, 4e-6, None, 0.00004, None),
        (100, 5, True, 0, 2e-6, 4e-6, None, 0.00004, None),
        (100, 5, 1.5, 0, 2e-6, 4e-6, None, 0.00004, None),
        (100, 5, 0, -1, 2e-6, 4e-6, None, 0.00004, None),
        (100, 5, 0, True, 2e-6, 4e-6, None, 0.00004, None),
        (100, 5, 0, 1.5, 2e-6, 4e-6, None, 0.00004, None),
        (100, 5, 80, 30, 2e-6, 4e-6, None, 0.00004, None),
        (None, 5, 0, 0, 2e-6, 4e-6, None, 0.00004, None),
        (100, None, 20, 10, 2e-6, 4e-6, 0.00078, None, None),
        (100, 5, 20, 0, None, 4e-6, None, 0.00004, None),
        (100, 5, 0, 20, 2e-6, None, None, 0.00004, None),
    ],
)
def test_direct_cache_pricing_accounts_only_valid_reported_components(
    input_tokens,
    output_tokens,
    cached_tokens,
    cache_write_tokens,
    read_rate,
    write_rate,
    expected_input,
    expected_output,
    expected_total,
):
    response = _direct_cache_response(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=None,
        cached_tokens=cached_tokens,
        cache_write_tokens=cache_write_tokens,
    )

    _apply_cache_pricing(response, read_rate=read_rate, write_rate=write_rate)

    if expected_input is None:
        assert response.cost_input is None
    else:
        assert response.cost_input == pytest.approx(expected_input)
    if expected_output is None:
        assert response.cost_output is None
    else:
        assert response.cost_output == pytest.approx(expected_output)
    if expected_total is None:
        assert response.cost_total is None
    else:
        assert response.cost_total == pytest.approx(expected_total)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("item_currency", "expected_total"),
    [("USD", 0.00882), ("EUR", None)],
)
def test_direct_cost_breakdown_preserves_cache_costs_and_currency_completeness(
    item_currency,
    expected_total,
):
    response = _direct_cache_response()
    _apply_cache_pricing(response)
    ocr = _ocr_item(currency=item_currency)

    response.apply_cost_breakdown([ocr])

    assert response.cost_input == pytest.approx(0.00078)
    assert response.cost_output == pytest.approx(0.00004)
    assert response.cost_breakdown == [ocr]
    if expected_total is None:
        assert response.cost_total is None
    else:
        assert response.cost_total == pytest.approx(expected_total)


@pytest.mark.unit
def test_direct_cost_breakdown_keeps_output_but_not_total_for_incomplete_cache_cost():
    response = _direct_cache_response(
        cache_write_tokens=None,
    )
    _apply_cache_pricing(response)
    ocr = _ocr_item(quantity=1)

    response.apply_cost_breakdown([ocr])

    assert response.cost_input is None
    assert response.cost_output == pytest.approx(0.00004)
    assert response.cost_breakdown == [ocr]
    assert response.cost_total is None


@pytest.mark.unit
def test_multi_tier_chat_without_provider_usage_leaves_costs_unset():
    adapter = _adapter()

    with patch.object(
        OpenAISyncClient,
        "complete",
        return_value=_response(200, include_usage=False),
    ):
        response = adapter.chat([UserMessage("hi")])

    assert response.usage is None
    assert response.currency is None
    assert response.cost_input is None
    assert response.cost_output is None
    assert response.cost_total is None


@pytest.mark.unit
def test_pricing_overrides_apply_to_the_selected_tier():
    adapter = _adapter()
    adapter.pricing.set_in_per_1m(7.0)
    adapter.pricing.set_out_per_1m(11.0)

    with patch.object(
        OpenAISyncClient,
        "complete",
        return_value=_response(201),
    ):
        response = adapter.chat([UserMessage("hi")])

    _assert_costs(
        response,
        input_tokens=201,
        input_per_1m=7.0,
        output_per_1m=11.0,
    )


@pytest.mark.unit
def test_single_tier_pricing_has_no_boundary():
    adapter = _adapter()
    adapter.pricing = Pricing.from_dict(
        [
            {
                "up_to_prompt_tokens": None,
                "input_per_1m": 7.0,
                "output_per_1m": 11.0,
            }
        ],
        currency="USD",
    )

    with patch.object(
        OpenAISyncClient,
        "complete",
        return_value=_response(201),
    ):
        response = adapter.chat([UserMessage("hi")])

    _assert_costs(
        response,
        input_tokens=201,
        input_per_1m=7.0,
        output_per_1m=11.0,
    )
