"""Focused live checks for the Mistral-hosted GLM Flash route."""

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from tests.e2e.conftest import e2e_model_case_parameters
from tests.e2e.harness import assert_usage_and_pricing


@pytest.mark.e2e
@pytest.mark.e2e_capability("reasoning_control")
@pytest.mark.parametrize(
    "e2e_model_case",
    e2e_model_case_parameters("zai", service_provider="mistral"),
)
@pytest.mark.parametrize(
    ("reasoning_level", "capture_reasoning"),
    (
        ("low", False),
        ("low", True),
        ("high", False),
        ("max", False),
        ("none", False),
    ),
    ids=("low-uncaptured", "low-captured", "high", "max", "none-fallback"),
)
def test_flash_reasoning_control(
    e2e_model_case,
    e2e_model_organization,
    chat_with_retry,
    e2e_adapter,
    reasoning_level,
    capture_reasoning,
):
    model = e2e_model_case.model_spec
    assert model is not None
    assert e2e_model_case.organization == "zai"
    assert e2e_model_case.service_provider == "mistral"

    adapter = e2e_adapter(
        e2e_model_organization,
        model.name,
        organization_name=e2e_model_case.organization,
        service_provider=e2e_model_case.service_provider,
    )
    assert adapter.organization == "zai"
    assert adapter.service_provider == "mistral"
    assert adapter.model == "glm-5.3-flash"
    assert adapter.model_spec.provider_model == "zai-glm-5-3-flash"

    assert model.reasoning_capability.allowed_values == ("low", "high", "max")
    if reasoning_level == "none":
        with pytest.warns(UserWarning, match="cannot disable reasoning"):
            response = chat_with_retry(
                adapter,
                messages=[UserMessage("Reply with OK.")],
                max_tokens=256,
                reasoning_level=reasoning_level,
                capture_reasoning=capture_reasoning,
                timeout_s=60,
            )
    else:
        response = chat_with_retry(
            adapter,
            messages=[UserMessage("Reply with OK.")],
            max_tokens=256,
            reasoning_level=reasoning_level,
            capture_reasoning=capture_reasoning,
            timeout_s=60,
        )

    assert type(response.content) is str
    assert bool(response.content)
    assert type(response.finish_reason) is str
    assert bool(response.finish_reason)
    if capture_reasoning:
        assert all(isinstance(event.text, str) for event in response.reasoning_events)
    else:
        assert len(response.reasoning_events) == 0
    assert_usage_and_pricing(response)
