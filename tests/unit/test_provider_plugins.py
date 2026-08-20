"""Deterministic tests for the lazy external-provider plugin contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import src.llm_api_adapter.provider_registry as registry_module
import src.llm_api_adapter.universal_adapter as universal_module
from src.llm_api_adapter.adapters.anthropic_adapter import AnthropicAdapter
from src.llm_api_adapter.errors import ProviderNotInstalledError
from src.llm_api_adapter.provider_registry import (
    DuplicateServiceProviderError,
    PROVIDER_PLUGIN_API_VERSION,
    PROVIDER_PLUGIN_ENTRY_POINT_GROUP,
    ProviderPlugin,
    ProviderPluginDiscovery,
    ServiceProviderRegistry,
)
from src.llm_api_adapter.universal_adapter import UniversalLLMAPIAdapter


@dataclass
class PluginTestAdapter:
    company: str
    model: str
    api_key: str
    transport: str
    service_provider: str

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


def _test_plugin(service_provider: str = "mistral") -> ProviderPlugin:
    def register(registry: ServiceProviderRegistry) -> None:
        registry.register(service_provider, PluginTestAdapter)

    return ProviderPlugin(
        api_version=PROVIDER_PLUGIN_API_VERSION,
        register=register,
    )


@pytest.fixture
def isolated_plugin_runtime(monkeypatch):
    registry = ServiceProviderRegistry({"anthropic": AnthropicAdapter})
    discovery = ProviderPluginDiscovery()
    monkeypatch.setattr(
        universal_module,
        "SERVICE_PROVIDER_REGISTRY",
        registry,
    )
    monkeypatch.setattr(
        universal_module,
        "PROVIDER_PLUGIN_DISCOVERY",
        discovery,
    )
    return registry, discovery


@pytest.mark.unit
def test_external_provider_is_discovered_only_after_its_distribution_is_available(
    monkeypatch,
    isolated_plugin_runtime,
):
    installed_entry_points: list[FakeEntryPoint] = []

    def get_entry_points(*, group: str):
        assert group == PROVIDER_PLUGIN_ENTRY_POINT_GROUP
        return tuple(installed_entry_points)

    monkeypatch.setattr(registry_module, "entry_points", get_entry_points)

    with pytest.raises(
        ProviderNotInstalledError,
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
        plugin=_test_plugin(),
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
    assert entry_point.load_calls == 1


@pytest.mark.unit
def test_builtin_provider_does_not_load_external_plugins(monkeypatch, isolated_plugin_runtime):
    entry_point = FakeEntryPoint(
        name="broken-provider",
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

    assert registry.register("test-provider", PluginTestAdapter) is True
    assert registry.register("test-provider", PluginTestAdapter) is False
    with pytest.raises(
        DuplicateServiceProviderError,
        match="already registered: test-provider",
    ):
        registry.register("test-provider", lambda **_: PluginTestAdapter)


@pytest.mark.unit
def test_plugin_failures_are_recorded_without_breaking_registered_providers(
    monkeypatch,
    isolated_plugin_runtime,
):
    registry, discovery = isolated_plugin_runtime
    broken_entry_point = FakeEntryPoint(
        name="broken-provider",
        value="broken.plugin:PLUGIN",
        error=RuntimeError("broken plugin"),
    )
    duplicate_entry_point = FakeEntryPoint(
        name="duplicate-provider",
        value="duplicate.plugin:PLUGIN",
        plugin=ProviderPlugin(
            api_version=PROVIDER_PLUGIN_API_VERSION,
            register=lambda providers: providers.register(
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
        "broken-provider",
        "duplicate-provider",
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
    registry, discovery = isolated_plugin_runtime
    entry_point = FakeEntryPoint(
        name="legacy-provider",
        value="legacy.plugin:PLUGIN",
        plugin=ProviderPlugin(
            api_version=PROVIDER_PLUGIN_API_VERSION + 1,
            register=lambda providers: None,
        ),
    )
    monkeypatch.setattr(
        registry_module,
        "entry_points",
        lambda *, group: (entry_point,),
    )

    discovery.discover(registry)

    assert registry.get("legacy-provider") is None
    assert discovery.failures[0].error_type == "ValueError"
    assert "Unsupported provider plugin API version" in discovery.failures[0].message
