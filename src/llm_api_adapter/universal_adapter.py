from dataclasses import dataclass
import logging
from typing import Any

from .adapters.base_adapter import LLMAdapterBase
from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.google_adapter import GoogleAdapter
from .adapters.openai_compatible.adapter import (
    OpenAICompatibleAdapter,
    _redact_compatible_diagnostic,
    _validate_api_root,
)
from .errors.config_errors import LLMConfigError, OrganizationNotInstalledError
from .llm_registry.llm_registry import LLM_REGISTRY
from .llms.transports import validate_sync_transport
from .organization_registry import (
    KNOWN_ORGANIZATION_PACKAGES,
    OrganizationPluginDiscovery,
)
from .service_provider_registry import ServiceProviderRegistry

logger = logging.getLogger(__name__)


SERVICE_PROVIDER_REGISTRY = ServiceProviderRegistry({
    AnthropicAdapter.company: AnthropicAdapter,
    OpenAIAdapter.company: OpenAIAdapter,
    GoogleAdapter.company: GoogleAdapter,
    OpenAICompatibleAdapter.company: OpenAICompatibleAdapter,
})
ORGANIZATION_PLUGIN_DISCOVERY = OrganizationPluginDiscovery()


@dataclass
class UniversalLLMAPIAdapter:
    organization: str
    model: str
    api_key: str
    transport: str = "requests"
    service_provider: str | None = None
    base_url: str | None = None

    def __repr__(self) -> str:
        masked = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
        result = (
            f"UniversalLLMAPIAdapter(organization='{self.organization}', "
            f"service_provider='{self.service_provider}', model='{self.model}', "
            f"transport='{self.transport}', api_key='{masked}')"
        )
        if self.organization == OpenAICompatibleAdapter.company:
            return _redact_compatible_diagnostic(result, self.api_key)
        return result

    def __post_init__(self) -> None:
        if not self.organization or not isinstance(self.organization, str):
            raise ValueError("Invalid organization")

        self._configure_connection()

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

    def _configure_connection(self) -> None:
        """Validate and normalize connection settings before factory selection or plugin discovery.

        Compatible roots reuse the adapter's canonical, side-effect-free validator.
        """
        if (
            self.organization != OpenAICompatibleAdapter.company
            and self.base_url is not None
        ):
            raise LLMConfigError(
                message=(
                    "base_url is supported only for "
                    "organization='openai_compatible'."
                )
            )
        if self.organization == OpenAICompatibleAdapter.company:
            if self.service_provider is None:
                self.service_provider = self.organization
            elif (
                not isinstance(self.service_provider, str)
                or self.service_provider != self.organization
            ):
                raise LLMConfigError(
                    message=(
                        "OpenAI-compatible mode requires "
                        "service_provider='openai_compatible'."
                    )
                )
            self.base_url = _validate_api_root(self.base_url)
        elif self.service_provider is None:
            self.service_provider = self.organization
        elif (
            not isinstance(self.service_provider, str)
            or not self.service_provider
        ):
            raise ValueError("Invalid service provider")

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
            ORGANIZATION_PLUGIN_DISCOVERY.discover(
                SERVICE_PROVIDER_REGISTRY,
                model_registry=LLM_REGISTRY,
            )
            adapter_factory = SERVICE_PROVIDER_REGISTRY.get(service_provider)
        if adapter_factory is not None:
            factory_kwargs = dict(
                company=organization,
                model=model,
                api_key=api_key,
                transport=transport,
                service_provider=service_provider,
            )
            if organization == OpenAICompatibleAdapter.company:
                factory_kwargs["base_url"] = self.base_url
            return adapter_factory(**factory_kwargs)
        if service_provider == organization:
            known_organization = KNOWN_ORGANIZATION_PACKAGES.get(organization)
            if known_organization is not None:
                error = OrganizationNotInstalledError(
                    organization=organization,
                    distribution=known_organization.distribution,
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
