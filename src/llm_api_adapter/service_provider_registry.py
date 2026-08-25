"""Service-provider adapter factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


ServiceProviderFactory = Callable[..., Any]


class DuplicateServiceProviderError(ValueError):
    """Raised when different factories claim the same service-provider key."""


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
