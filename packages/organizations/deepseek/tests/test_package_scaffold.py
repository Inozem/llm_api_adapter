"""Deterministic checks for the independently buildable DeepSeek package scaffold."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
CORE_SOURCE = REPOSITORY_ROOT / "src"
PACKAGE_SOURCE = PACKAGE_ROOT / "src"
for source in (str(PACKAGE_SOURCE), str(CORE_SOURCE), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

from llm_api_adapter.organization_registry import (
    ORGANIZATION_PLUGIN_API_VERSION,
    OrganizationPlugin,
)
from llm_api_adapter.service_provider_registry import ServiceProviderRegistry


@pytest.mark.unit
def test_deepseek_project_declares_core_dependency_extras_and_entry_point():
    metadata = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'version = "0.1.0"' in metadata
    assert 'dependencies = ["llm-api-adapter>=0.9.6,<1.0.0"]' in metadata
    assert 'async = ["llm-api-adapter[async]>=0.9.6,<1.0.0"]' in metadata
    assert 'httpx = ["llm-api-adapter[httpx]>=0.9.6,<1.0.0"]' in metadata
    assert 'deepseek = "llm_api_adapter_deepseek.plugin:PLUGIN"' in metadata


@pytest.mark.unit
def test_deepseek_plugin_entry_point_matches_the_core_contract():
    from llm_api_adapter_deepseek.plugin import PLUGIN

    assert isinstance(PLUGIN, OrganizationPlugin)
    assert PLUGIN.api_version == ORGANIZATION_PLUGIN_API_VERSION
    assert callable(PLUGIN.register)


@pytest.mark.unit
def test_deepseek_plugin_registers_the_deepseek_service_provider():
    from llm_api_adapter_deepseek.plugin import PLUGIN

    registry = ServiceProviderRegistry()
    PLUGIN.register(registry)

    assert registry.get("deepseek") is not None
