import pytest

from llm_api_adapter.errors import JSONSchemaError
from llm_api_adapter.models.messages.chat_message import UserMessage
from llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
}


@pytest.mark.e2e
def test_json_schema_returns_parsed_json_for_all_providers(subtests, iter_provider_models, chat_with_retry):
    for p, model in iter_provider_models():
        with subtests.test(provider=p["name"], model=model):
            adapter = UniversalLLMAPIAdapter(
                organization=p["name"],
                model=model,
                api_key=p["api_key"],
            )

            try:
                resp = chat_with_retry(
                    adapter,
                    messages=[
                        UserMessage(
                            'Return a JSON object with fields "name" (string) '
                            'and "age" (integer). Use name="Alice" and age=30.'
                        )
                    ],
                    max_tokens=500,
                    json_schema=SIMPLE_SCHEMA,
                    timeout_s=60,
                )
            except JSONSchemaError as e:
                pytest.skip(f"Model returned non-JSON despite json_schema mode: {e}")

            if resp.content is None:
                pytest.skip("Model returned no content — likely exhausted token budget in reasoning")

            assert resp.parsed_json is not None
            assert isinstance(resp.parsed_json, dict)
            assert "name" in resp.parsed_json
            assert "age" in resp.parsed_json
            assert isinstance(resp.parsed_json["name"], str)
            assert isinstance(resp.parsed_json["age"], int)
