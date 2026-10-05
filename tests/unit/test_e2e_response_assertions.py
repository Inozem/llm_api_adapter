from dataclasses import replace

import pytest

from llm_api_adapter.models.responses.chat_response import ChatResponse, Usage
from tests.e2e.harness import assert_usage_and_pricing


@pytest.mark.unit
@pytest.mark.parametrize("thoughts_tokens", [None, 0])
def test_e2e_assertions_accept_omitted_google_usage_fields(thoughts_tokens):
    usage_metadata = {
        "promptTokenCount": 10,
        "candidatesTokenCount": 2,
        "totalTokenCount": 12,
    }
    if thoughts_tokens is not None:
        usage_metadata["thoughtsTokenCount"] = thoughts_tokens
    response = ChatResponse.from_google_response({"usageMetadata": usage_metadata})
    response.apply_pricing(
        1e-6, 2e-6, "USD", price_cache_read_per_token=0.5e-6
    )

    assert_usage_and_pricing(response)
    assert response.usage.cached_tokens is None
    assert response.usage.output_tokens == (None if thoughts_tokens is None else 2)
    assert response.cost_input is None
    assert response.cost_total is None


@pytest.mark.unit
@pytest.mark.parametrize("component_costs", [(None, None), (0.001, 0.002)])
def test_e2e_assertions_accept_provider_total_or_complete_pricing(component_costs):
    assert_usage_and_pricing(
        ChatResponse(
            usage=Usage(10, 2, 12),
            currency="USD",
            cost_input=component_costs[0],
            cost_output=component_costs[1],
            cost_total=0.003,
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "changes",
    [
        {"usage": None},
        {"usage": Usage(-1, None, None)},
        {"usage": Usage(10, 2, 11)},
        {"cost_output": -0.001},
        {"cost_input": 0.001, "cost_output": 0.002, "cost_total": 0.004},
    ],
)
def test_e2e_assertions_still_reject_invalid_reported_metadata(changes):
    response = ChatResponse(usage=Usage(10, None, 12), currency="USD")

    with pytest.raises(AssertionError):
        assert_usage_and_pricing(replace(response, **changes))
