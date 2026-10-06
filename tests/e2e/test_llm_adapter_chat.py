import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from tests.e2e.conftest import e2e_model_case_parameters
from tests.e2e.harness import assert_usage_and_pricing


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

    assert_usage_and_pricing(resp)


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

    assert_usage_and_pricing(resp)
