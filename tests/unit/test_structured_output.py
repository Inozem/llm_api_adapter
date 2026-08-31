from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from src.llm_api_adapter.errors.llm_api_error import JSONSchemaError
from src.llm_api_adapter.adapters.structured_output import (
    normalize_local_references,
    prepare_structured_output,
    validate_core_portable_schema,
)
from tests.fixtures.structured_output import (
    NestedPydanticResponse,
    PORTABLE_PROFILE_SCHEMAS,
)


def _contains_reference(node: Any) -> bool:
    if isinstance(node, dict):
        return "$ref" in node or any(_contains_reference(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_reference(value) for value in node)
    return False


@pytest.mark.unit
def test_preparation_inlines_local_definition_without_mutating_source_schema():
    source_schema = {
        "$defs": {
            "Contact": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
        "type": "object",
        "properties": {"contact": {"$ref": "#/$defs/Contact"}},
        "required": ["contact"],
    }
    original = deepcopy(source_schema)

    prepared = prepare_structured_output(
        source_schema,
        None,
        None,
        provider="openai",
    )

    assert source_schema == original
    assert prepared.source_schema == original
    assert prepared.source_schema is not source_schema
    assert prepared.provider_schema == {
        "type": "object",
        "properties": {
            "contact": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
        "required": ["contact"],
    }


@pytest.mark.unit
def test_normalization_resolves_local_references_inside_arrays():
    normalized = normalize_local_references(
        {
            "$defs": {
                "Tag": {"type": "string", "enum": ["new", "archived"]},
            },
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Tag"},
                },
            },
        },
        provider="anthropic",
    )

    assert normalized["properties"]["tags"]["items"] == {
        "type": "string",
        "enum": ["new", "archived"],
    }
    assert "$defs" not in normalized
    assert not _contains_reference(normalized)


@pytest.mark.unit
def test_pydantic_source_schema_and_model_remain_separate_from_provider_schema():
    prepared = prepare_structured_output(
        None,
        NestedPydanticResponse,
        None,
        provider="google",
    )

    assert prepared.response_model is NestedPydanticResponse
    assert prepared.source_schema == NestedPydanticResponse.model_json_schema()
    assert "$defs" in prepared.source_schema
    assert "$defs" not in prepared.provider_schema
    assert not _contains_reference(prepared.provider_schema)
    assert prepared.provider_schema["properties"]["contact"]["type"] == "object"


@pytest.mark.unit
@pytest.mark.parametrize("provider", ("openai", "anthropic", "google"))
@pytest.mark.parametrize("schema", PORTABLE_PROFILE_SCHEMAS.values())
def test_core_portable_profile_accepts_the_shared_fixture_vocabulary(provider, schema):
    original = deepcopy(schema)

    prepared = validate_core_portable_schema(schema, provider=provider)

    assert prepared == original
    assert prepared is not schema
    assert schema == original


@pytest.mark.unit
@pytest.mark.parametrize(
    ("schema", "expected_path", "expected_detail"),
    [
        (
            {"type": "array", "items": {"type": "string"}},
            "#",
            "root object",
        ),
        (
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": [],
                "additionalProperties": False,
            },
            "#/required",
            "every property",
        ),
        (
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            "#/additionalProperties",
            "additionalProperties",
        ),
        (
            {
                "type": "object",
                "properties": {
                    "answer": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                    },
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
            "#/properties/answer",
            "does not support anyOf",
        ),
    ],
)
def test_core_portable_profile_rejects_lossy_or_unsupported_schemas(
    schema,
    expected_path,
    expected_detail,
):
    with pytest.raises(JSONSchemaError) as error:
        validate_core_portable_schema(schema, provider="google")

    message = str(error.value)
    assert "google structured-output schema" in message
    assert expected_path in message
    assert expected_detail in message


@pytest.mark.unit
@pytest.mark.parametrize(
    ("schema", "expected_path", "expected_detail"),
    [
        (
            {"type": "object", "properties": {"value": {"$ref": "other.json"}}},
            "#/properties/value",
            "supports only local",
        ),
        (
            {"type": "object", "properties": {"value": {"$ref": "#/$defs/Missing"}}},
            "#/properties/value",
            "cannot resolve",
        ),
        (
            {
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {"next": {"$ref": "#/$defs/Node"}},
                    },
                },
                "$ref": "#/$defs/Node",
            },
            "#/$defs/Node/properties/next",
            "recursive",
        ),
    ],
)
def test_preparation_rejects_external_unresolved_and_recursive_references(
    schema,
    expected_path,
    expected_detail,
):
    with pytest.raises(JSONSchemaError) as error:
        prepare_structured_output(schema, None, None, provider="mistral")

    message = str(error.value)
    assert "mistral structured-output schema" in message
    assert expected_path in message
    assert expected_detail in message
