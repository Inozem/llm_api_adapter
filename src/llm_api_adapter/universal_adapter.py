from dataclasses import dataclass
import logging
from typing import Any

from .adapters.base_adapter import LLMAdapterBase
from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.google_adapter import GoogleAdapter
from .errors.config_errors import ProviderNotInstalledError
from .llms.transports import validate_sync_transport
from .provider_registry import (
    KNOWN_PROVIDER_PACKAGES,
    ProviderPluginDiscovery,
    ServiceProviderRegistry,
)

logger = logging.getLogger(__name__)


SERVICE_PROVIDER_REGISTRY = ServiceProviderRegistry({
    AnthropicAdapter.company: AnthropicAdapter,
    OpenAIAdapter.company: OpenAIAdapter,
    GoogleAdapter.company: GoogleAdapter,
})
PROVIDER_PLUGIN_DISCOVERY = ProviderPluginDiscovery()


@dataclass
class UniversalLLMAPIAdapter:
    organization: str
    model: str
    api_key: str
    transport: str = "requests"
    service_provider: str | None = None

    def __repr__(self) -> str:
        masked = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
        return (
            f"UniversalLLMAPIAdapter(organization='{self.organization}', "
            f"service_provider='{self.service_provider}', model='{self.model}', "
            f"transport='{self.transport}', api_key='{masked}')"
        )

    def __post_init__(self) -> None:
        if not self.organization or not isinstance(self.organization, str):
            raise ValueError("Invalid organization")
        if self.service_provider is None:
            self.service_provider = self.organization
        elif (
            not isinstance(self.service_provider, str)
            or not self.service_provider
        ):
            raise ValueError("Invalid service provider")
        if not self.model or not isinstance(self.model, str):
            raise ValueError("Invalid model")
        if not self.api_key or not isinstance(self.api_key, str):
            raise ValueError("Invalid API key")
        self.transport = validate_sync_transport(self.transport)
        self.adapter = self._select_adapter(
            self.organization,
            self.service_provider,
            self.model,
            self.api_key,
            self.transport,
        )

    def _select_adapter(
        self,
        organization: str,
        service_provider: str,
        model: str,
        api_key: str,
        transport: str,
    ) -> LLMAdapterBase:
        """Select a built-in or lazily registered service-provider factory."""
        adapter_factory = SERVICE_PROVIDER_REGISTRY.get(service_provider)
        if adapter_factory is None:
            PROVIDER_PLUGIN_DISCOVERY.discover(SERVICE_PROVIDER_REGISTRY)
            adapter_factory = SERVICE_PROVIDER_REGISTRY.get(service_provider)
        if adapter_factory is not None:
            return adapter_factory(
                company=organization,
                model=model,
                api_key=api_key,
                transport=transport,
                service_provider=service_provider,
            )
        if service_provider == organization:
            known_provider = KNOWN_PROVIDER_PACKAGES.get(organization)
            if known_provider is not None:
                error = ProviderNotInstalledError(
                    organization=organization,
                    distribution=known_provider.distribution,
                )
                logger.error(str(error))
                raise error
            error_message = f"Unsupported organization: {organization}"
        else:
            error_message = f"Unsupported service provider: {service_provider}"
        logger.error(error_message)
        raise ValueError(error_message)

    def __getattr__(self, name: str) -> Any:
        """
        Redirects method calls to the selected adapter.
        """
        if hasattr(self.adapter, name):
            return getattr(self.adapter, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )
