"""Internal preparation of provider-neutral structured-output schemas."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

from ..errors.llm_api_error import JSONSchemaError


_REFERENCE_METADATA_KEYS = frozenset(
    {"title", "description", "default", "examples", "$comment"},
)

_PORTABLE_SCHEMA_TYPES = frozenset(
    {"string", "number", "integer", "boolean", "object", "array"},
)
_PORTABLE_SCHEMA_KEYWORDS = frozenset(
    {
        "$id",
        "$schema",
        "additionalProperties",
        "description",
        "enum",
        "items",
        "properties",
        "required",
        "title",
        "type",
    },
)


@dataclass(frozen=True)
class PreparedStructuredOutput:
    """Keep source validation inputs distinct from the provider-ready schema."""

    source_schema: Optional[dict]
    provider_schema: Optional[dict]
    response_model: Optional[Any]


def prepare_structured_output(
    json_schema: Optional[dict],
    response_model: Optional[Any],
    tools: Optional[Any],
    *,
    provider: str,
) -> PreparedStructuredOutput:
    """Validate structured-output inputs and prepare a provider schema.

    ``response_model`` remains untouched for response validation.  The source
    schema is a deep copy of the caller-supplied JSON Schema or Pydantic's
    generated schema, while ``provider_schema`` has local references inlined.
    """
    if json_schema is not None and response_model is not None:
        raise JSONSchemaError(
            detail="json_schema and response_model cannot be used together",
        )
    if response_model is not None and tools is not None:
        raise JSONSchemaError(
            detail="response_model and tools cannot be used together",
        )
    if json_schema is not None and tools is not None:
        raise JSONSchemaError(
            detail="json_schema and tools cannot be used together",
        )

    if response_model is not None:
        source_schema = _schema_from_response_model(response_model)
    else:
        if json_schema is not None and not isinstance(json_schema, dict):
            raise JSONSchemaError(detail="json_schema must be a dict")
        source_schema = json_schema

    copied_source_schema = deepcopy(source_schema) if source_schema is not None else None
    provider_schema = (
        normalize_local_references(copied_source_schema, provider=provider)
        if copied_source_schema is not None
        else None
    )
    return PreparedStructuredOutput(
        source_schema=copied_source_schema,
        provider_schema=provider_schema,
        response_model=response_model,
    )


def normalize_local_references(schema: dict, *, provider: str) -> dict:
    """Inline supported ``#/$defs/...`` references without mutating ``schema``.

    The portable profile permits only direct, local references into the root
    ``$defs`` mapping.  External, unresolved, recursive, and semantically
    ambiguous references are rejected before a provider request is made.
    """
    source = deepcopy(schema)
    definitions = source.get("$defs", {})
    if definitions is not None and not isinstance(definitions, dict):
        raise _reference_error(
            provider,
            "#/$defs",
            "must be an object when local references are used",
        )

    normalized = _normalize_schema_node(
        source,
        provider=provider,
        path="#",
        definitions=definitions or {},
        active_references=frozenset(),
        is_root=True,
    )
    if not isinstance(normalized, dict):
        raise _reference_error(provider, "#", "must resolve to an object schema")
    return normalized


def validate_core_portable_schema(schema: dict, *, provider: str) -> dict:
    """Validate the schema profile shared by the built-in organizations.

    The Core profile is intentionally narrower than JSON Schema: it is the
    documented intersection used by OpenAI, Anthropic, and Google.  It never
    rewrites semantic constraints.  A caller must make strict-object and
    required-field intent explicit rather than relying on a provider-specific
    repair step.
    """
    portable_schema = deepcopy(schema)
    _validate_core_schema_node(
        portable_schema,
        provider=provider,
        path="#",
        is_root=True,
    )
    return portable_schema


def _validate_core_schema_node(
    node: Any,
    *,
    provider: str,
    path: str,
    is_root: bool = False,
) -> None:
    if not isinstance(node, dict):
        raise _core_profile_error(provider, path, "must be an object schema")

    unsupported_keywords = set(node) - _PORTABLE_SCHEMA_KEYWORDS
    if unsupported_keywords:
        names = ", ".join(sorted(unsupported_keywords))
        raise _core_profile_error(provider, path, f"does not support {names}")

    schema_types = _portable_schema_types(node.get("type"), provider=provider, path=path)
    if is_root and schema_types != {"object"}:
        raise _core_profile_error(provider, path, "requires a root object schema")

    if "enum" in node:
        enum = node["enum"]
        if not isinstance(enum, list) or not enum:
            raise _core_profile_error(provider, _pointer_path(path, "enum"), "requires a non-empty array")

    if "object" in schema_types:
        _validate_core_object_schema(node, provider=provider, path=path)
    elif any(key in node for key in ("properties", "required", "additionalProperties")):
        raise _core_profile_error(
            provider,
            path,
            "allows object keywords only on an object schema",
        )

    if "array" in schema_types:
        items = node.get("items")
        if not isinstance(items, dict):
            raise _core_profile_error(
                provider,
                _pointer_path(path, "items"),
                "requires one object item schema",
            )
        _validate_core_schema_node(
            items,
            provider=provider,
            path=_pointer_path(path, "items"),
        )
    elif "items" in node:
        raise _core_profile_error(
            provider,
            path,
            "allows items only on an array schema",
        )


def _portable_schema_types(value: Any, *, provider: str, path: str) -> set[str]:
    if isinstance(value, str):
        schema_types = {value}
    elif isinstance(value, list):
        if len(value) != 2 or len(set(value)) != 2 or "null" not in value:
            raise _core_profile_error(
                provider,
                _pointer_path(path, "type"),
                "permits nullable fields only as [<type>, \"null\"]",
            )
        schema_types = set(value)
    else:
        raise _core_profile_error(
            provider,
            _pointer_path(path, "type"),
            "requires an explicit type",
        )

    non_null_types = schema_types - {"null"}
    if not non_null_types or not non_null_types <= _PORTABLE_SCHEMA_TYPES:
        raise _core_profile_error(
            provider,
            _pointer_path(path, "type"),
            "uses a type outside the Core portable profile",
        )
    return schema_types


def _validate_core_object_schema(
    node: dict[str, Any],
    *,
    provider: str,
    path: str,
) -> None:
    if node.get("additionalProperties") is not False:
        raise _core_profile_error(
            provider,
            _pointer_path(path, "additionalProperties"),
            "requires additionalProperties to be false",
        )

    properties = node.get("properties")
    if properties is None:
        if "required" in node:
            raise _core_profile_error(
                provider,
                _pointer_path(path, "required"),
                "requires properties when required is present",
            )
        return
    if not isinstance(properties, dict):
        raise _core_profile_error(
            provider,
            _pointer_path(path, "properties"),
            "must be an object",
        )

    required = node.get("required")
    if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
        raise _core_profile_error(
            provider,
            _pointer_path(path, "required"),
            "must list every property name",
        )
    if len(set(required)) != len(required) or set(required) != set(properties):
        raise _core_profile_error(
            provider,
            _pointer_path(path, "required"),
            "must contain every property exactly once",
        )

    for name, property_schema in properties.items():
        _validate_core_schema_node(
            property_schema,
            provider=provider,
            path=_pointer_path(_pointer_path(path, "properties"), name),
        )


def _schema_from_response_model(response_model: Any) -> dict:
    try:
        schema = response_model.model_json_schema()
    except AttributeError:
        try:
            import pydantic  # noqa: F401
        except ImportError:
            raise JSONSchemaError(
                detail=(
                    "pydantic is required for response_model; install it with: "
                    "pip install pydantic"
                ),
            )
        raise JSONSchemaError(
            detail="response_model must be a Pydantic BaseModel subclass",
        )
    if not isinstance(schema, dict):
        raise JSONSchemaError(detail="response_model.model_json_schema() must return a dict")
    return schema


def _normalize_schema_node(
    node: Any,
    *,
    provider: str,
    path: str,
    definitions: dict[str, Any],
    active_references: frozenset[str],
    is_root: bool = False,
) -> Any:
    if isinstance(node, list):
        return [
            _normalize_schema_node(
                item,
                provider=provider,
                path=_pointer_path(path, str(index)),
                definitions=definitions,
                active_references=active_references,
            )
            for index, item in enumerate(node)
        ]
    if not isinstance(node, dict):
        return deepcopy(node)

    if "$ref" in node:
        return _resolve_reference_node(
            node,
            provider=provider,
            path=path,
            definitions=definitions,
            active_references=active_references,
        )

    return {
        key: _normalize_schema_node(
            value,
            provider=provider,
            path=_pointer_path(path, key),
            definitions=definitions,
            active_references=active_references,
        )
        for key, value in node.items()
        if not (is_root and key == "$defs")
    }


def _resolve_reference_node(
    node: dict[str, Any],
    *,
    provider: str,
    path: str,
    definitions: dict[str, Any],
    active_references: frozenset[str],
) -> Any:
    reference = node["$ref"]
    if not isinstance(reference, str):
        raise _reference_error(provider, path, "$ref must be a string")
    definition_name = _local_definition_name(reference, provider=provider, path=path)
    if definition_name in active_references:
        raise _reference_error(
            provider,
            path,
            f"contains recursive local reference {reference!r}",
        )
    if definition_name not in definitions:
        raise _reference_error(
            provider,
            path,
            f"cannot resolve local reference {reference!r}",
        )

    # A root Pydantic schema can declare ``$defs`` beside a root ``$ref``.
    # The definitions mapping is the resolution table, not an additional
    # constraint to merge into the inlined schema.
    sibling_keys = set(node) - {"$ref", "$defs"}
    unsupported_siblings = sibling_keys - _REFERENCE_METADATA_KEYS
    if unsupported_siblings:
        names = ", ".join(sorted(unsupported_siblings))
        raise _reference_error(
            provider,
            path,
            f"cannot combine local reference {reference!r} with {names}",
        )

    resolved = _normalize_schema_node(
        definitions[definition_name],
        provider=provider,
        path=_pointer_path(_pointer_path("#", "$defs"), definition_name),
        definitions=definitions,
        active_references=active_references | {definition_name},
    )
    if not sibling_keys:
        return resolved
    if not isinstance(resolved, dict):
        raise _reference_error(
            provider,
            path,
            f"cannot attach metadata to non-object local reference {reference!r}",
        )
    return {
        **resolved,
        **{
            key: _normalize_schema_node(
                node[key],
                provider=provider,
                path=_pointer_path(path, key),
                definitions=definitions,
                active_references=active_references,
            )
            for key in sibling_keys
        },
    }


def _local_definition_name(reference: str, *, provider: str, path: str) -> str:
    prefix = "#/$defs/"
    if not reference.startswith(prefix):
        raise _reference_error(
            provider,
            path,
            f"supports only local #/$defs/... references, got {reference!r}",
        )
    escaped_name = reference[len(prefix):]
    if not escaped_name or "/" in escaped_name:
        raise _reference_error(
            provider,
            path,
            f"supports only direct local $defs references, got {reference!r}",
        )
    return escaped_name.replace("~1", "/").replace("~0", "~")


def _pointer_path(parent: str, token: str) -> str:
    escaped_token = str(token).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{escaped_token}"


def _reference_error(provider: str, path: str, detail: str) -> JSONSchemaError:
    return JSONSchemaError(
        detail=f"{provider} structured-output schema at {path}: {detail}",
    )


def _core_profile_error(provider: str, path: str, detail: str) -> JSONSchemaError:
    return JSONSchemaError(
        detail=(
            f"{provider} structured-output schema at {path}: "
            f"Core portable profile {detail}"
        ),
    )


__all__ = [
    "PreparedStructuredOutput",
    "normalize_local_references",
    "prepare_structured_output",
    "validate_core_portable_schema",
]
