from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from src.llm_api_adapter.errors.llm_api_error import JSONSchemaError
from src.llm_api_adapter.adapters.structured_output import (
    normalize_local_references,
    prepare_structured_output,
)
from tests.fixtures.structured_output import NestedPydanticResponse


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
