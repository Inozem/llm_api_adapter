from unittest.mock import AsyncMock, patch

import pytest

from src.llm_api_adapter.adapters.openai_adapter import OpenAIAdapter
from src.llm_api_adapter.llm_registry.llm_registry import Pricing
from src.llm_api_adapter.llms.openai.async_client import OpenAIAsyncClient
from src.llm_api_adapter.llms.openai.sync_client import OpenAISyncClient
from src.llm_api_adapter.llms.streaming import SSEEvent
from src.llm_api_adapter.models.messages.chat_message import UserMessage


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


def _response(
    input_tokens: int,
    *,
    output_tokens: int = 10,
    include_usage: bool = True,
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
    return response


def _stream_events(input_tokens: int, *, include_usage: bool = True) -> list[SSEEvent]:
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
                ),
            },
        ),
    ]


async def _async_stream_events(
    input_tokens: int,
    *,
    include_usage: bool = True,
):
    for event in _stream_events(input_tokens, include_usage=include_usage):
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
