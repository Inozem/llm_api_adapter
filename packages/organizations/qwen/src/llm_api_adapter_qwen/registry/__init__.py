"""Package-local, validated model metadata for the Qwen organization."""

from __future__ import annotations

import json
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from llm_api_adapter.llm_registry.llm_registry import OrganizationModelMetadata
from ..request_rules import validate_qwen_policy_catalogue


def _load_organization_data() -> dict[str, Any]:
    resource = files(__package__).joinpath("organizations/qwen.json")
    with resource.open("r", encoding="utf-8") as source:
        organization_data = json.load(source)
    if not isinstance(organization_data, dict):
        raise ValueError("Qwen organization metadata must be an object")
    return organization_data


_ORGANIZATION_DATA = _load_organization_data()
QWEN_MODEL_POLICIES = MappingProxyType(
    validate_qwen_policy_catalogue(_ORGANIZATION_DATA)
)

MODEL_METADATA = OrganizationModelMetadata(
    organization="qwen",
    organization_data=_ORGANIZATION_DATA,
)


__all__ = ["MODEL_METADATA"]
