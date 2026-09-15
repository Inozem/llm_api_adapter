"""Package-local, validated model metadata for Kimi / Moonshot."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from llm_api_adapter.llm_registry.llm_registry import OrganizationModelMetadata

from ..request_rules import KIMI_REQUEST_RULE_REGISTRY
from .cache_pricing import KimiCachePricing, load_cache_pricing


def _load_organization_data() -> dict[str, Any]:
    resource = files(__package__).joinpath("organizations/kimi.json")
    with resource.open("r", encoding="utf-8") as source:
        organization_data = json.load(source)
    if not isinstance(organization_data, dict):
        raise ValueError("Kimi organization metadata must be an object")
    return organization_data


ORGANIZATION_DATA = _load_organization_data()
CACHE_PRICING: dict[str, KimiCachePricing] = load_cache_pricing(ORGANIZATION_DATA)
MODEL_METADATA = OrganizationModelMetadata(
    organization="kimi",
    organization_data=ORGANIZATION_DATA,
    request_rule_registry=KIMI_REQUEST_RULE_REGISTRY,
)


__all__ = ["CACHE_PRICING", "KimiCachePricing", "MODEL_METADATA"]
