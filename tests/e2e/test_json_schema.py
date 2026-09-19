"""Live structured-output smoke test for every registered model."""

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage


_EXPECTED_JSON = {"contact": {"name": "Ada"}}
_PORTABLE_NESTED_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "contact": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    "required": ["contact"],
    "additionalProperties": False,
}


@pytest.mark.e2e
@pytest.mark.e2e_feature("structured_output")
def test_json_schema_returns_structured_output_for_every_configured_model(
    subtests,
    iter_organization_models,
    chat_with_retry,
    e2e_adapter,
):
    """Make one portable structured-output request for every configured model."""
    configured_models = 0
    for organization, model in iter_organization_models():
        if not organization["api_key"]:
            continue
        configured_models += 1

        with subtests.test(organization=organization["name"], model=model):
            adapter = e2e_adapter(organization, model)
            response = chat_with_retry(
                adapter,
                messages=[
                    UserMessage('Return exactly {"contact":{"name":"Ada"}}.')
                ],
                max_tokens=1000,
                json_schema=_PORTABLE_NESTED_OBJECT_SCHEMA,
                timeout_s=60,
            )

            assert response.refusal is None
            assert response.incomplete_reason is None
            assert response.parsed_json == _EXPECTED_JSON

    if not configured_models:
        pytest.skip("No provider API keys are configured")
