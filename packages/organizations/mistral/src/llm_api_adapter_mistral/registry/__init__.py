"""Package-local, validated model metadata for the Mistral provider."""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from llm_api_adapter.llm_registry.llm_registry import ProviderModelMetadata


def _load_provider_data() -> dict[str, Any]:
    resource = files(__package__).joinpath("providers/mistral.json")
    with resource.open("r", encoding="utf-8") as source:
        provider_data = json.load(source)
    if not isinstance(provider_data, dict):
        raise ValueError("Mistral provider metadata must be an object")
    return provider_data


MODEL_METADATA = ProviderModelMetadata(
    organization="mistral",
    provider_data=_load_provider_data(),
)


__all__ = ["MODEL_METADATA"]
