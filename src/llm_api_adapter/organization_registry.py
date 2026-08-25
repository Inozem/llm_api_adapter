"""Organization package metadata and lazy plugin discovery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
import logging

from .llm_registry.llm_registry import OrganizationModelMetadata, RegistrySpec
from .service_provider_registry import ServiceProviderRegistry


logger = logging.getLogger(__name__)

ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP = "llm_api_adapter.organizations"
ORGANIZATION_PLUGIN_API_VERSION = 1

OrganizationPluginRegister = Callable[[ServiceProviderRegistry], None]


@dataclass(frozen=True)
class KnownOrganizationPackage:
    """An external organization package that core can describe before installation."""

    organization: str
    distribution: str

    @property
    def install_command(self) -> str:
        """Return the exact command needed to install this organization package."""
        return f"pip install {self.distribution}"


KNOWN_ORGANIZATION_PACKAGES = {
    "mistral": KnownOrganizationPackage(
        organization="mistral",
        distribution="llm-api-adapter-mistral",
    ),
}


@dataclass(frozen=True)
class OrganizationPlugin:
    """Versioned object exposed by an external organization entry point."""

    api_version: int
    register: OrganizationPluginRegister
    model_metadata: OrganizationModelMetadata | None = None


@dataclass(frozen=True)
class OrganizationPluginFailure:
    """Diagnostic information for one organization plugin that could not register."""

    entry_point_name: str
    entry_point_value: str
    error_type: str
    message: str


class OrganizationPluginDiscovery:
    """Lazily load each installed organization entry point at most once."""

    def __init__(self) -> None:
        self._attempted_entry_points: set[tuple[str, str]] = set()
        self._failures: list[OrganizationPluginFailure] = []

    @property
    def failures(self) -> tuple[OrganizationPluginFailure, ...]:
        """Return diagnostics for plugins that failed to load or register."""
        return tuple(self._failures)

    def discover(
        self,
        service_provider_registry: ServiceProviderRegistry,
        model_registry: RegistrySpec | None = None,
    ) -> None:
        """Load organization plugins without letting one break core usage."""
        try:
            organization_entry_points = entry_points(
                group=ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP,
            )
        except Exception as error:
            self._record_failure(
                entry_point_name="<discovery>",
                entry_point_value=ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP,
                error=error,
            )
            return

        for organization_entry_point in organization_entry_points:
            entry_point_key = (
                organization_entry_point.name,
                organization_entry_point.value,
            )
            if entry_point_key in self._attempted_entry_points:
                continue
            self._attempted_entry_points.add(entry_point_key)

            try:
                plugin = organization_entry_point.load()
                self._register_plugin(
                    plugin,
                    service_provider_registry,
                    model_registry,
                )
            except Exception as error:
                self._record_failure(
                    entry_point_name=organization_entry_point.name,
                    entry_point_value=organization_entry_point.value,
                    error=error,
                )

    def _register_plugin(
        self,
        plugin: object,
        service_provider_registry: ServiceProviderRegistry,
        model_registry: RegistrySpec | None,
    ) -> None:
        if not isinstance(plugin, OrganizationPlugin):
            raise TypeError(
                "Organization entry point must resolve to an OrganizationPlugin instance",
            )
        if plugin.api_version != ORGANIZATION_PLUGIN_API_VERSION:
            raise ValueError(
                "Unsupported organization plugin API version "
                f"{plugin.api_version!r}; expected {ORGANIZATION_PLUGIN_API_VERSION}",
            )
        if not callable(plugin.register):
            raise TypeError("Organization plugin register must be callable")
        if plugin.model_metadata is not None:
            if model_registry is None:
                raise ValueError(
                    "Organization model metadata requires a model registry"
                )
            model_registry.register_organization_metadata(plugin.model_metadata)
        plugin.register(service_provider_registry)

    def _record_failure(
        self,
        *,
        entry_point_name: str,
        entry_point_value: str,
        error: Exception,
    ) -> None:
        failure = OrganizationPluginFailure(
            entry_point_name=entry_point_name,
            entry_point_value=entry_point_value,
            error_type=type(error).__name__,
            message=str(error),
        )
        self._failures.append(failure)
        logger.warning(
            "Unable to load organization plugin %r (%s): %s",
            entry_point_name,
            entry_point_value,
            failure.message,
        )
