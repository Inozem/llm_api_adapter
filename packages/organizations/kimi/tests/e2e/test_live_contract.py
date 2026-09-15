"""Bounded live contracts for Kimi's declared model and reasoning matrix."""

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from tests.e2e import conftest as core_e2e
from tests.e2e import harness as e2e_harness


_MAX_TOKENS = 128
_REASONING_LEVEL_BY_MODEL = {
    "kimi-k3": "high",
    "kimi-k2.6": "none",
}


@pytest.mark.e2e
@pytest.mark.e2e_kimi
def test_kimi_models_apply_their_declared_reasoning_mode_through_the_facade():
    """Exercise each model through the installed plugin, never the adapter directly."""
    profile = core_e2e.get_e2e_organization_profile("kimi")
    (organization,) = core_e2e.resolve_e2e_organizations(profile)

    for model in organization["models"]:
        response = e2e_harness.chat_with_transient_retry(
            e2e_harness.create_e2e_adapter(organization, model),
            messages=[UserMessage("Reply with exactly: OK")],
            max_tokens=_MAX_TOKENS,
            reasoning_level=_REASONING_LEVEL_BY_MODEL[model],
            timeout_s=60,
        )

        assert response.content and response.content.strip() == "OK"
        assert response.finish_reason
        assert response.usage is not None
        assert response.usage.output_tokens <= _MAX_TOKENS
