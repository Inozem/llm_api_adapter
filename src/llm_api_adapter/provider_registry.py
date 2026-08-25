"""Service-provider factories and lazy external-plugin discovery."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
import logging
from typing import Any

from .llm_registry.llm_registry import ProviderModelMetadata, RegistrySpec

logger = logging.getLogger(__name__)

PROVIDER_PLUGIN_ENTRY_POINT_GROUP = "llm_api_adapter.providers"
PROVIDER_PLUGIN_API_VERSION = 1

ServiceProviderFactory = Callable[..., Any]
ProviderPluginRegister = Callable[["ServiceProviderRegistry"], None]


class DuplicateServiceProviderError(ValueError):
    """Raised when different factories claim the same service-provider key."""


@dataclass(frozen=True)
class KnownProviderPackage:
    """An external provider package that core can describe before installation."""

    organization: str
    distribution: str

    @property
    def install_command(self) -> str:
        """Return the exact command needed to install this provider package."""
        return f"pip install {self.distribution}"


KNOWN_PROVIDER_PACKAGES = {
    "mistral": KnownProviderPackage(
        organization="mistral",
        distribution="llm-api-adapter-mistral",
    ),
}


@dataclass(frozen=True)
class ProviderPlugin:
    """Versioned object exposed by an external provider entry point."""

    api_version: int
    register: ProviderPluginRegister
    model_metadata: ProviderModelMetadata | None = None


@dataclass(frozen=True)
class ProviderPluginFailure:
    """Diagnostic information for one plugin that could not register."""

    entry_point_name: str
    entry_point_value: str
    error_type: str
    message: str


class ServiceProviderRegistry:
    """Maps API service-provider keys to adapter factories."""

    def __init__(
        self,
        service_providers: Mapping[str, ServiceProviderFactory] | None = None,
    ) -> None:
        self._service_providers: dict[str, ServiceProviderFactory] = {}
        for service_provider, factory in (service_providers or {}).items():
            self.register(service_provider, factory)

    @property
    def service_providers(self) -> tuple[str, ...]:
        """Return service-provider keys in registration order."""
        return tuple(self._service_providers)

    def register(
        self,
        service_provider: str,
        factory: ServiceProviderFactory,
    ) -> bool:
        """Register a factory, returning false when it was already registered."""
        if not isinstance(service_provider, str) or not service_provider:
            raise ValueError("Service provider must be a non-empty string")
        if not callable(factory):
            raise TypeError("Service-provider factory must be callable")

        existing_factory = self._service_providers.get(service_provider)
        if existing_factory is None:
            self._service_providers[service_provider] = factory
            return True
        if existing_factory is factory:
            return False
        raise DuplicateServiceProviderError(
            "Service provider already registered: "
            f"{service_provider}",
        )

    def get(self, service_provider: str) -> ServiceProviderFactory | None:
        """Return the registered factory for a service provider, if present."""
        return self._service_providers.get(service_provider)


class ProviderPluginDiscovery:
    """Lazily load each installed provider entry point at most once."""

    def __init__(self) -> None:
        self._attempted_entry_points: set[tuple[str, str]] = set()
        self._failures: list[ProviderPluginFailure] = []

    @property
    def failures(self) -> tuple[ProviderPluginFailure, ...]:
        """Return diagnostics for plugins that failed to load or register."""
        return tuple(self._failures)

    def discover(
        self,
        registry: ServiceProviderRegistry,
        model_registry: RegistrySpec | None = None,
    ) -> None:
        """Load installed provider plugins without letting one break core usage."""
        try:
            provider_entry_points = entry_points(
                group=PROVIDER_PLUGIN_ENTRY_POINT_GROUP,
            )
        except Exception as error:
            self._record_failure(
                entry_point_name="<discovery>",
                entry_point_value=PROVIDER_PLUGIN_ENTRY_POINT_GROUP,
                error=error,
            )
            return

        for provider_entry_point in provider_entry_points:
            entry_point_key = (
                provider_entry_point.name,
                provider_entry_point.value,
            )
            if entry_point_key in self._attempted_entry_points:
                continue
            self._attempted_entry_points.add(entry_point_key)

            try:
                plugin = provider_entry_point.load()
                self._register_plugin(plugin, registry, model_registry)
            except Exception as error:
                self._record_failure(
                    entry_point_name=provider_entry_point.name,
                    entry_point_value=provider_entry_point.value,
                    error=error,
                )

    def _register_plugin(
        self,
        plugin: object,
        registry: ServiceProviderRegistry,
        model_registry: RegistrySpec | None,
    ) -> None:
        if not isinstance(plugin, ProviderPlugin):
            raise TypeError(
                "Provider entry point must resolve to a ProviderPlugin instance",
            )
        if plugin.api_version != PROVIDER_PLUGIN_API_VERSION:
            raise ValueError(
                "Unsupported provider plugin API version "
                f"{plugin.api_version!r}; expected {PROVIDER_PLUGIN_API_VERSION}",
            )
        if not callable(plugin.register):
            raise TypeError("Provider plugin register must be callable")
        if plugin.model_metadata is not None:
            if model_registry is None:
                raise ValueError(
                    "Provider model metadata requires a model registry"
                )
            model_registry.register_provider_metadata(plugin.model_metadata)
        plugin.register(registry)

    def _record_failure(
        self,
        *,
        entry_point_name: str,
        entry_point_value: str,
        error: Exception,
    ) -> None:
        failure = ProviderPluginFailure(
            entry_point_name=entry_point_name,
            entry_point_value=entry_point_value,
            error_type=type(error).__name__,
            message=str(error),
        )
        self._failures.append(failure)
        logger.warning(
            "Unable to load provider plugin %r (%s): %s",
            entry_point_name,
            entry_point_value,
            failure.message,
        )
