"""Bounded HTTPX sync and async contract checks for every provider."""

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


pytest.importorskip("httpx")


def _assert_usage_and_pricing(response):
    assert isinstance(response.content, str)
    assert response.content.strip()
    assert isinstance(response.finish_reason, str)
    assert response.usage is not None
    assert response.usage.input_tokens >= 0
    assert response.usage.output_tokens >= 0
    assert response.usage.total_tokens >= response.usage.input_tokens
    assert response.currency
    assert response.cost_total is not None and response.cost_total >= 0
    if response.cost_input is None or response.cost_output is None:
        assert response.cost_input is response.cost_output is None
    else:
        assert response.cost_input >= 0
        assert response.cost_output >= 0


def _adapter(organization, model: str, *, transport: str) -> UniversalLLMAPIAdapter:
    return UniversalLLMAPIAdapter(
        organization=organization["name"],
        model=model,
        api_key=organization["api_key"],
        transport=transport,
    )


def _chat_kwargs():
    return {
        "messages": [UserMessage("Reply with exactly: OK")],
        # Gemini 3.x may spend part of this shared output budget on default
        # thinking before returning visible text.
        "max_tokens": 512,
        "timeout_s": 60,
    }


@pytest.mark.e2e
def test_sync_httpx_chat_returns_contract_for_latest_provider_models(
    subtests,
    configured_sync_httpx_e2e_models,
    chat_with_retry,
):
    """Make one HTTPX-backed sync request for each configured provider."""
    if not configured_sync_httpx_e2e_models:
        pytest.skip("No provider API keys are configured")

    for organization, model in configured_sync_httpx_e2e_models:
        with subtests.test(organization=organization["name"], model=model):
            response = chat_with_retry(
                _adapter(organization, model, transport="httpx"),
                **_chat_kwargs(),
            )
            _assert_usage_and_pricing(response)


@pytest.mark.asyncio
@pytest.mark.e2e
async def test_async_httpx_chat_returns_contract_for_latest_provider_models(
    subtests,
    configured_sync_httpx_e2e_models,
    async_chat_with_retry,
):
    """Make one async HTTPX request for each configured provider.

    ``achat()`` always uses the shared asynchronous HTTPX transport.  Passing
    ``transport='httpx'`` also verifies that the opt-in sync transport setting
    remains compatible with an async facade instance.
    """
    if not configured_sync_httpx_e2e_models:
        pytest.skip("No provider API keys are configured")

    for organization, model in configured_sync_httpx_e2e_models:
        with subtests.test(organization=organization["name"], model=model):
            response = await async_chat_with_retry(
                _adapter(organization, model, transport="httpx"),
                **_chat_kwargs(),
            )
            _assert_usage_and_pricing(response)
