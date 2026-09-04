"""Entry point loaded by the Core organization-plugin discovery mechanism."""

from __future__ import annotations

from llm_api_adapter.organization_registry import (
    ORGANIZATION_PLUGIN_API_VERSION,
    OrganizationPlugin,
)
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry

from .adapter import QwenAdapter
from .registry import MODEL_METADATA


def register(registry: ServiceProviderRegistry) -> None:
    """Register the direct Qwen Model Studio service provider."""
    registry.register("qwen", QwenAdapter)


PLUGIN = OrganizationPlugin(
    api_version=ORGANIZATION_PLUGIN_API_VERSION,
    register=register,
    model_metadata=MODEL_METADATA,
)


__all__ = ["PLUGIN", "register"]
