"""Live structured-output smoke test for every registered model."""

import pytest

from llm_api_adapter.models.messages.chat_message import UserMessage
from tests.e2e.conftest import e2e_model_case_parameters


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
@pytest.mark.e2e_capability("structured_output_schema")
@pytest.mark.parametrize("e2e_model_case", e2e_model_case_parameters())
def test_json_schema_returns_structured_output_for_every_configured_model(
    e2e_model_case,
    e2e_model_organization,
    chat_with_retry,
    e2e_adapter,
):
    """Make one portable structured-output request for every configured model."""
    model = e2e_model_case.model_spec
    assert model is not None
    if not e2e_model_organization["api_key"]:
        pytest.skip("No organization API key is configured")

    adapter = e2e_adapter(e2e_model_organization, model.name)
    response = chat_with_retry(
        adapter,
        messages=[UserMessage('Return exactly {"contact":{"name":"Ada"}}.')],
        max_tokens=1000,
        json_schema=_PORTABLE_NESTED_OBJECT_SCHEMA,
        timeout_s=60,
    )

    assert response.refusal is None
    assert response.incomplete_reason is None
    assert response.parsed_json == _EXPECTED_JSON
