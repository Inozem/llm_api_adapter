"""Provider-neutral structured-output fixtures for deterministic tests.

The fixtures describe the v0.9.2 portable-profile acceptance vocabulary.  They
intentionally contain no provider payload shapes, so core and organization
package suites can use the same inputs and expected terminal outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ConfigDict


FLAT_OBJECT_SCHEMA: Final = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

NULLABLE_REQUIRED_FIELD_SCHEMA: Final = {
    "type": "object",
    "properties": {"nickname": {"type": ["string", "null"]}},
    "required": ["nickname"],
    "additionalProperties": False,
}

ENUM_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["priority"],
    "additionalProperties": False,
}

ARRAY_SCHEMA: Final = {
    "type": "object",
    "properties": {
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["tags"],
    "additionalProperties": False,
}

INLINE_NESTED_OBJECT_SCHEMA: Final = {
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


class PortableContact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str


class NestedPydanticResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contact: PortableContact


NESTED_PYDANTIC_RESPONSE_JSON: Final = '{"contact": {"name": "Ada"}}'
INVALID_JSON_CONTENT: Final = '{"contact": '


@dataclass(frozen=True)
class StructuredOutputOutcomeFixture:
    """Named terminal outcome used by the portable-profile conformance suite."""

    name: str
    content: str | None


STRUCTURED_OUTPUT_OUTCOMES: Final = (
    StructuredOutputOutcomeFixture("valid_structured_result", '{"answer": "ok"}'),
    StructuredOutputOutcomeFixture("refusal", None),
    StructuredOutputOutcomeFixture("incomplete_result", None),
    StructuredOutputOutcomeFixture("invalid_json", INVALID_JSON_CONTENT),
    StructuredOutputOutcomeFixture("pydantic_validation_error", '{"contact": {}}'),
    StructuredOutputOutcomeFixture("unsupported_schema", None),
)

PORTABLE_PROFILE_SCHEMAS: Final = {
    "flat_object": FLAT_OBJECT_SCHEMA,
    "nullable_required_field": NULLABLE_REQUIRED_FIELD_SCHEMA,
    "enum": ENUM_SCHEMA,
    "array": ARRAY_SCHEMA,
    "inline_nested_object": INLINE_NESTED_OBJECT_SCHEMA,
}
