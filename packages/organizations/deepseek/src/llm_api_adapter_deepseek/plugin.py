"""Entry point loaded by Core's organization-plugin discovery mechanism."""

from __future__ import annotations

from llm_api_adapter.organization_registry import (
    ORGANIZATION_PLUGIN_API_VERSION,
    OrganizationPlugin,
)
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry

from .adapter import DeepSeekAdapter
from .registry import MODEL_METADATA


def register(registry: ServiceProviderRegistry) -> None:
    """Register DeepSeek's direct official Responses API adapter."""
    registry.register("deepseek", DeepSeekAdapter)


PLUGIN = OrganizationPlugin(
    api_version=ORGANIZATION_PLUGIN_API_VERSION,
    register=register,
    model_metadata=MODEL_METADATA,
)


__all__ = ["PLUGIN", "register"]
