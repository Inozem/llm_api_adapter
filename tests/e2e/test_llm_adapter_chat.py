from math import isclose

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from tests.e2e.conftest import e2e_model_case_parameters


def _assert_pricing_contract(response) -> None:
    assert response.currency
    assert response.cost_total is not None and response.cost_total >= 0

    if response.cost_input is None or response.cost_output is None:
        assert response.cost_input is response.cost_output is None
        return

    assert response.cost_input >= 0
    assert response.cost_output >= 0
    assert isclose(
        response.cost_total,
        response.cost_input + response.cost_output,
        rel_tol=0,
        abs_tol=1e-9,
    )


@pytest.mark.e2e
@pytest.mark.e2e_capability("sync_chat", "usage_reporting")
@pytest.mark.parametrize("e2e_model_case", e2e_model_case_parameters())
def test_chat_accepts_basic_params_and_returns_contract(
    e2e_model_case,
    e2e_model_organization,
    chat_with_retry,
    e2e_adapter,
):
    model = e2e_model_case.model_spec
    assert model is not None
    adapter = e2e_adapter(e2e_model_organization, model.name)
    resp = chat_with_retry(
        adapter,
        messages=[UserMessage("Say 'OK'.")],
        max_tokens=1026,
        temperature=1.0,
        top_p=1.0,
        timeout_s=60,
    )

    assert isinstance(resp.content, str)
    assert isinstance(resp.finish_reason, str)

    assert resp.usage is not None
    assert resp.usage.input_tokens >= 0
    assert resp.usage.output_tokens >= 0
    assert resp.usage.total_tokens >= resp.usage.input_tokens

    _assert_pricing_contract(resp)


@pytest.mark.e2e
@pytest.mark.e2e_capability("reasoning_control")
@pytest.mark.parametrize("e2e_model_case", e2e_model_case_parameters())
def test_chat_with_reasoning_level_returns_valid_contract(
    e2e_model_case,
    e2e_model_organization,
    chat_with_retry,
    e2e_adapter,
):
    model = e2e_model_case.model_spec
    assert model is not None
    adapter = e2e_adapter(e2e_model_organization, model.name)
    resp = chat_with_retry(
        adapter,
        messages=[{"role": "user", "content": "Say 'OK'."}],
        max_tokens=2000,
        reasoning_level=1024,
        timeout_s=60,
    )

    assert isinstance(resp.content, str)
    assert isinstance(resp.finish_reason, str) and resp.finish_reason

    assert resp.usage is not None
    assert resp.usage.input_tokens >= 0
    assert resp.usage.output_tokens >= 0
    assert resp.usage.total_tokens >= 0
    assert resp.usage.total_tokens >= resp.usage.input_tokens + resp.usage.output_tokens

    _assert_pricing_contract(resp)
