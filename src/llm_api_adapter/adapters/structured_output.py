"""Internal preparation of provider-neutral structured-output schemas."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

from ..errors.llm_api_error import JSONSchemaError


_REFERENCE_METADATA_KEYS = frozenset(
    {"title", "description", "default", "examples", "$comment"},
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


__all__ = ["PreparedStructuredOutput", "normalize_local_references", "prepare_structured_output"]
