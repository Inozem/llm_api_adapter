"""Package-local, validated model metadata for the Z.ai organization."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from llm_api_adapter.llm_registry.llm_registry import OrganizationModelMetadata

from .cache_pricing import CACHE_PRICING, ZaiCacheCost, ZaiCachePricing


def _load_organization_data() -> dict[str, Any]:
    resource = files(__package__).joinpath("organizations/zai.json")
    with resource.open("r", encoding="utf-8") as source:
        organization_data = json.load(source)
    if not isinstance(organization_data, dict):
        raise ValueError("Z.ai organization metadata must be an object")
    return organization_data


ORGANIZATION_DATA = _load_organization_data()
MODEL_METADATA = OrganizationModelMetadata(
    organization="zai",
    organization_data=ORGANIZATION_DATA,
)


__all__ = [
    "CACHE_PRICING",
    "MODEL_METADATA",
    "ORGANIZATION_DATA",
    "ZaiCacheCost",
    "ZaiCachePricing",
]
