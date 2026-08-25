"""Package-local, validated model metadata for the Mistral organization."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from llm_api_adapter.llm_registry.llm_registry import OrganizationModelMetadata


def _load_organization_data() -> dict[str, Any]:
    resource = files(__package__).joinpath("organizations/mistral.json")
    with resource.open("r", encoding="utf-8") as source:
        organization_data = json.load(source)
    if not isinstance(organization_data, dict):
        raise ValueError("Mistral organization metadata must be an object")
    return organization_data


MODEL_METADATA = OrganizationModelMetadata(
    organization="mistral",
    organization_data=_load_organization_data(),
)


__all__ = ["MODEL_METADATA"]
