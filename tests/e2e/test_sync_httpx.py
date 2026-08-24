"""Bounded live verification for the opt-in synchronous HTTPX transport."""

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


pytest.importorskip("httpx")


@pytest.mark.e2e
def test_sync_httpx_chat_returns_contract_for_latest_provider_models(
    subtests,
    configured_sync_httpx_e2e_models,
    chat_with_retry,
):
    """Make one paid HTTPX sync request for each configured provider."""
    if not configured_sync_httpx_e2e_models:
        pytest.skip("No provider API keys are configured")

    for provider, model in configured_sync_httpx_e2e_models:
        with subtests.test(provider=provider["name"], model=model):
            adapter = UniversalLLMAPIAdapter(
                organization=provider["name"],
                model=model,
                api_key=provider["api_key"],
                transport="httpx",
            )
            response = chat_with_retry(
                adapter,
                messages=[UserMessage("Reply with exactly: OK")],
                # Gemini 3.x may spend part of this shared output budget on
                # default thinking before returning visible text.
                max_tokens=512,
                timeout_s=60,
            )

            assert isinstance(response.content, str)
            assert response.content.strip()
            assert isinstance(response.finish_reason, str)
            assert response.usage is not None
            assert response.usage.input_tokens >= 0
            assert response.usage.output_tokens >= 0
            assert response.usage.total_tokens >= response.usage.input_tokens
            assert response.currency
            assert response.cost_input >= 0
            assert response.cost_output >= 0
            assert response.cost_total >= 0
