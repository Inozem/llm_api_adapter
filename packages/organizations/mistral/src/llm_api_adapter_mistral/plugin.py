"""Entry point loaded by the core provider-plugin discovery mechanism."""

from __future__ import annotations

from llm_api_adapter.provider_registry import (
    PROVIDER_PLUGIN_API_VERSION,
    ProviderPlugin,
    ServiceProviderRegistry,
)

from .adapter import MistralAdapter
from .registry import MODEL_METADATA


def register(registry: ServiceProviderRegistry) -> None:
    """Register Mistral's direct API as the ``mistral`` service provider."""
    registry.register("mistral", MistralAdapter)


PLUGIN = ProviderPlugin(
    api_version=PROVIDER_PLUGIN_API_VERSION,
    register=register,
    model_metadata=MODEL_METADATA,
)


__all__ = ["PLUGIN", "register"]
