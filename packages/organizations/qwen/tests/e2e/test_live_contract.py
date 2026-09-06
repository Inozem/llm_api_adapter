"""Bounded live contracts that are specific to Qwen Model Studio."""

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from tests.e2e import conftest as core_e2e
from tests.e2e import harness as e2e_harness


_MAX_TOKENS = 128


@pytest.mark.e2e
@pytest.mark.e2e_qwen
def test_qwen_models_bound_output_and_toggle_thinking():
    """Probe every registered Qwen model's Frankfurt Messages behavior."""
    profile = core_e2e.get_e2e_organization_profile("qwen")
    (organization,) = core_e2e.resolve_e2e_organizations(profile)

    for model in organization["models"]:
        for reasoning_level in (1024, None):
            response = e2e_harness.chat_with_transient_retry(
                e2e_harness.create_e2e_adapter(organization, model),
                messages=[UserMessage("Reply with exactly: OK")],
                max_tokens=_MAX_TOKENS,
                reasoning_level=reasoning_level,
                capture_reasoning=True,
                timeout_s=60,
            )

            assert response.content and response.content.strip()
            assert response.finish_reason
            assert response.usage is not None
            assert response.usage.output_tokens <= _MAX_TOKENS
            if reasoning_level is None:
                assert response.reasoning is None
