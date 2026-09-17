"""Bounded live checks for DeepSeek's canonical Flash facade contract."""

from __future__ import annotations

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from tests.e2e import conftest as core_e2e
from tests.e2e import harness as e2e_harness


_MODEL = "deepseek-flash"
_MAX_TOKENS = 128
_REASONING_LEVELS = ("none", "high")


@pytest.mark.e2e
@pytest.mark.e2e_deepseek
def test_deepseek_flash_bounded_live_facade_contract():
    """Exercise only the published Flash text and thinking modes."""
    profile = core_e2e.get_e2e_organization_profile("deepseek")
    (organization,) = core_e2e.resolve_e2e_organizations(profile)

    for reasoning_level in _REASONING_LEVELS:
        response = e2e_harness.chat_with_transient_retry(
            e2e_harness.create_e2e_adapter(organization, _MODEL),
            messages=[UserMessage("Reply with exactly: OK")],
            max_tokens=_MAX_TOKENS,
            reasoning_level=reasoning_level,
            timeout_s=60,
        )

        assert response.content and response.content.strip() == "OK"
        assert response.model == _MODEL
        assert response.finish_reason
        assert response.usage is not None
        assert response.usage.output_tokens <= _MAX_TOKENS
