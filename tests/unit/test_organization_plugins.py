"""Deterministic tests for the lazy external-organization plugin contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import src.llm_api_adapter.adapters.base_adapter as base_adapter_module
import src.llm_api_adapter.organization_registry as registry_module
import src.llm_api_adapter.universal_adapter as universal_module
from src.llm_api_adapter.adapters.anthropic_adapter import AnthropicAdapter
from src.llm_api_adapter.adapters.base_adapter import LLMAdapterBase
from src.llm_api_adapter.errors import OrganizationNotInstalledError
from src.llm_api_adapter.llm_registry.llm_registry import (
    OrganizationModelMetadata,
    RegistrySpec,
    resolve_model_spec,
)
from src.llm_api_adapter.organization_registry import (
    ORGANIZATION_PLUGIN_API_VERSION,
    ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP,
    OrganizationPlugin,
    OrganizationPluginDiscovery,
)
from src.llm_api_adapter.service_provider_registry import (
    DuplicateServiceProviderError,
    ServiceProviderRegistry,
)
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


@dataclass
class PluginTestAdapter(LLMAdapterBase):
    company: str = "mistral"
    model: str = ""
    api_key: str = ""

    def chat(self, *args: Any, **kwargs: Any) -> dict[str, str]:
        return {"response": "ok"}

    def stream_chat(self, *args: Any, **kwargs: Any):
        yield "streamed"


class FakeEntryPoint:
    def __init__(
        self,
        name: str,
        value: str,
        plugin: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self._plugin = plugin
        self._error = error
        self.load_calls = 0

    def load(self) -> object:
        self.load_calls += 1
        if self._error is not None:
            raise self._error
        return self._plugin


def _organization_model_metadata(organization: str) -> OrganizationModelMetadata:
    return OrganizationModelMetadata(
        organization=organization,
        organization_data={
            "currency": "USD",
            "models": {
                "test-model": {
                    "limits": {
                        "context_window_tokens": 128_000,
                        "max_output_tokens": 16_384,
                    },
                    "pricing_tiers": [
                        {
                            "up_to_prompt_tokens": None,
                            "input_per_1m": 1.0,
                            "output_per_1m": 2.0,
                        }
                    ],
                }
            },
        },
    )


def _test_plugin(
    organization: str = "mistral",
    model_metadata: OrganizationModelMetadata | None = None,
) -> OrganizationPlugin:
    def register(registry: ServiceProviderRegistry) -> None:
        registry.register(organization, PluginTestAdapter)

    return OrganizationPlugin(
        api_version=ORGANIZATION_PLUGIN_API_VERSION,
        register=register,
        model_metadata=model_metadata,
    )


@pytest.fixture
def isolated_plugin_runtime(monkeypatch):
    registry = ServiceProviderRegistry({"anthropic": AnthropicAdapter})
    discovery = OrganizationPluginDiscovery()
    model_registry = RegistrySpec()
    monkeypatch.setattr(universal_module, "SERVICE_PROVIDER_REGISTRY", registry)
    monkeypatch.setattr(
        universal_module,
        "ORGANIZATION_PLUGIN_DISCOVERY",
        discovery,
    )
    monkeypatch.setattr(universal_module, "LLM_REGISTRY", model_registry)
    monkeypatch.setattr(base_adapter_module, "LLM_REGISTRY", model_registry)
    return registry, discovery, model_registry


@pytest.mark.unit
def test_external_organization_is_discovered_after_its_distribution_is_available(
    monkeypatch,
    isolated_plugin_runtime,
):
    installed_entry_points: list[FakeEntryPoint] = []

    def get_entry_points(*, group: str):
        assert group == ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP
        return tuple(installed_entry_points)

    monkeypatch.setattr(registry_module, "entry_points", get_entry_points)

    with pytest.raises(
        OrganizationNotInstalledError,
        match="pip install llm-api-adapter-mistral",
    ):
        UniversalLLMAPIAdapter(
            organization="mistral",
            model="test-model",
            api_key="test-key",
        )

    entry_point = FakeEntryPoint(
        name="mistral",
        value="llm_api_adapter_mistral.plugin:PLUGIN",
        plugin=_test_plugin(
            model_metadata=_organization_model_metadata("mistral"),
        ),
    )
    installed_entry_points.append(entry_point)

    adapter = UniversalLLMAPIAdapter(
        organization="mistral",
        model="test-model",
        api_key="test-key",
    )

    assert isinstance(adapter.adapter, PluginTestAdapter)
    assert adapter.adapter.company == "mistral"
    assert adapter.adapter.service_provider == "mistral"
    assert adapter.adapter.transport == "requests"
    assert adapter.adapter.model_spec is not None
    assert adapter.adapter.pricing is not None
    assert resolve_model_spec(
        isolated_plugin_runtime[2],
        "mistral",
        "test-model",
    ) is adapter.adapter.model_spec
    assert entry_point.load_calls == 1


@pytest.mark.unit
def test_known_xai_organization_is_installable_before_its_package_is_available(
    monkeypatch,
    isolated_plugin_runtime,
):
    installed_entry_points: list[FakeEntryPoint] = []

    def get_entry_points(*, group: str):
        assert group == ORGANIZATION_PLUGIN_ENTRY_POINT_GROUP
        return tuple(installed_entry_points)

    monkeypatch.setattr(registry_module, "entry_points", get_entry_points)

    with pytest.raises(OrganizationNotInstalledError) as raised:
        UniversalLLMAPIAdapter(
            organization="xai",
            model="test-model",
            api_key="test-key",
        )

    assert str(raised.value) == (
        "Organization 'xai' is not installed. "
        "Install it with: pip install llm-api-adapter-xai"
    )

    entry_point = FakeEntryPoint(
        name="xai",
        value="test_plugins.xai:PLUGIN",
        plugin=_test_plugin(
            organization="xai",
            model_metadata=_organization_model_metadata("xai"),
        ),
    )
    installed_entry_points.append(entry_point)

    adapter = UniversalLLMAPIAdapter(
        organization="xai",
        model="test-model",
        api_key="test-key",
    )

    assert isinstance(adapter.adapter, PluginTestAdapter)
    assert adapter.adapter.company == adapter.adapter.service_provider == "xai"
    assert resolve_model_spec(
        isolated_plugin_runtime[2],
        "xai",
        "test-model",
    ) is adapter.adapter.model_spec
    assert entry_point.load_calls == 1


@pytest.mark.unit
def test_builtin_organization_does_not_load_external_plugins(
    monkeypatch,
    isolated_plugin_runtime,
):
    entry_point = FakeEntryPoint(
        name="broken-organization",
        value="broken.plugin:PLUGIN",
        error=RuntimeError("plugin should remain unloaded"),
    )
    monkeypatch.setattr(
        registry_module,
        "entry_points",
        lambda *, group: (entry_point,),
    )

    adapter = UniversalLLMAPIAdapter(
        organization="anthropic",
        model="claude-sonnet-4-5",
        api_key="test-key",
    )

    assert isinstance(adapter.adapter, AnthropicAdapter)
    assert entry_point.load_calls == 0


@pytest.mark.unit
def test_plugin_registration_is_idempotent_and_rejects_duplicates():
    registry = ServiceProviderRegistry()

    assert registry.register("test-service", PluginTestAdapter) is True
    assert registry.register("test-service", PluginTestAdapter) is False
    with pytest.raises(
        DuplicateServiceProviderError,
        match="already registered: test-service",
    ):
        registry.register("test-service", lambda **_: PluginTestAdapter)


@pytest.mark.unit
def test_plugin_failures_are_recorded_without_breaking_registered_organizations(
    monkeypatch,
    isolated_plugin_runtime,
):
    registry, discovery, _ = isolated_plugin_runtime
    broken_entry_point = FakeEntryPoint(
        name="broken-organization",
        value="broken.plugin:PLUGIN",
        error=RuntimeError("broken plugin"),
    )
    duplicate_entry_point = FakeEntryPoint(
        name="duplicate-organization",
        value="duplicate.plugin:PLUGIN",
        plugin=OrganizationPlugin(
            api_version=ORGANIZATION_PLUGIN_API_VERSION,
            register=lambda organizations: organizations.register(
                "anthropic",
                PluginTestAdapter,
            ),
        ),
    )
    monkeypatch.setattr(
        registry_module,
        "entry_points",
        lambda *, group: (broken_entry_point, duplicate_entry_point),
    )

    discovery.discover(registry)
    discovery.discover(registry)

    assert registry.get("anthropic") is AnthropicAdapter
    assert [failure.entry_point_name for failure in discovery.failures] == [
        "broken-organization",
        "duplicate-organization",
    ]
    assert [failure.error_type for failure in discovery.failures] == [
        "RuntimeError",
        "DuplicateServiceProviderError",
    ]
    assert broken_entry_point.load_calls == duplicate_entry_point.load_calls == 1


@pytest.mark.unit
def test_incompatible_plugin_contract_is_reported_as_a_diagnostic(
    monkeypatch,
    isolated_plugin_runtime,
):
    registry, discovery, _ = isolated_plugin_runtime
    entry_point = FakeEntryPoint(
        name="legacy-organization",
        value="legacy.plugin:PLUGIN",
        plugin=OrganizationPlugin(
            api_version=ORGANIZATION_PLUGIN_API_VERSION + 1,
            register=lambda organizations: None,
        ),
    )
    monkeypatch.setattr(
        registry_module,
        "entry_points",
        lambda *, group: (entry_point,),
    )

    discovery.discover(registry)

    assert registry.get("legacy-organization") is None
    assert discovery.failures[0].error_type == "ValueError"
    assert "Unsupported organization plugin API version" in discovery.failures[0].message


@pytest.mark.unit
def test_invalid_plugin_model_metadata_is_reported_before_organization_registration(
    monkeypatch,
    isolated_plugin_runtime,
):
    registry, discovery, model_registry = isolated_plugin_runtime
    invalid_metadata = OrganizationModelMetadata(
        organization="mistral",
        organization_data={"models": {"test-model": {}}},
    )
    entry_point = FakeEntryPoint(
        name="mistral",
        value="llm_api_adapter_mistral.plugin:PLUGIN",
        plugin=_test_plugin(model_metadata=invalid_metadata),
    )
    monkeypatch.setattr(
        registry_module,
        "entry_points",
        lambda *, group: (entry_point,),
    )

    discovery.discover(registry, model_registry=model_registry)

    assert registry.get("mistral") is None
    assert resolve_model_spec(model_registry, "mistral", "test-model") is None
    assert discovery.failures[0].error_type == "ValueError"
    assert "Invalid organization metadata for 'mistral'" in discovery.failures[0].message
