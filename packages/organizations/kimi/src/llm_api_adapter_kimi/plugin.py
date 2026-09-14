"""Kimi organization-plugin entry point.

The registration hook is intentionally empty until the Kimi adapter and model
metadata are introduced in the next implementation commits.
"""

from __future__ import annotations

from llm_api_adapter.organization_registry import (
    ORGANIZATION_PLUGIN_API_VERSION,
    OrganizationPlugin,
)
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry


def register(registry: ServiceProviderRegistry) -> None:
    """Reserve Kimi's direct service-provider registration hook."""
    _ = registry


PLUGIN = OrganizationPlugin(
    api_version=ORGANIZATION_PLUGIN_API_VERSION,
    register=register,
)


__all__ = ["PLUGIN", "register"]
