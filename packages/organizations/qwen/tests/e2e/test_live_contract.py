"""Bounded live contracts that are specific to Qwen Model Studio."""

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from tests.e2e import conftest as core_e2e
from tests.e2e import harness as e2e_harness


_MAX_TOKENS = 128


@pytest.mark.e2e
@pytest.mark.e2e_qwen
def test_qwen_models_toggle_thinking_and_return_bounded_text():
    """Probe Qwen's thinking toggle without conflating usage with visible text."""
    profile = core_e2e.get_e2e_organization_profile("qwen")
    (organization,) = core_e2e.resolve_e2e_organizations(profile)

    for model in organization["models"]:
        for reasoning_level in (1024, "none"):
            response = e2e_harness.chat_with_transient_retry(
                e2e_harness.create_e2e_adapter(organization, model),
                messages=[UserMessage("Reply with exactly: OK")],
                max_tokens=_MAX_TOKENS,
                reasoning_level=reasoning_level,
                capture_reasoning=True,
                timeout_s=60,
            )

            assert response.content and response.content.strip() == "OK"
            assert response.finish_reason
            assert response.usage is not None
            if reasoning_level == "none":
                assert response.usage.output_tokens <= _MAX_TOKENS
                assert response.reasoning_events == []
