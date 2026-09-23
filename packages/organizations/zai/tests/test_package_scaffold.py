"""Deterministic checks for the independently buildable Z.ai package scaffold."""

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
def test_zai_package_layout_contains_distributable_files():
    required_files = (
        PACKAGE_ROOT / "pyproject.toml",
        PACKAGE_ROOT / "MANIFEST.in",
        PACKAGE_ROOT / "LICENSE",
        PACKAGE_ROOT / "README.md",
        PACKAGE_SOURCE / "llm_api_adapter_zai" / "__init__.py",
        PACKAGE_SOURCE / "llm_api_adapter_zai" / "py.typed",
        PACKAGE_SOURCE / "llm_api_adapter_zai" / "request_rules.py",
        PACKAGE_SOURCE / "llm_api_adapter_zai" / "clients" / "__init__.py",
    )

    missing = [
        str(path.relative_to(PACKAGE_ROOT))
        for path in required_files
        if not path.is_file()
    ]
    assert not missing, f"Missing Z.ai package files: {missing}"


@pytest.mark.unit
def test_zai_project_declares_core_dependency_extras_and_entry_point():
    metadata = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "llm-api-adapter-zai"' in metadata
    assert 'version = "0.1.0"' in metadata
    assert 'dependencies = ["llm-api-adapter>=0.9.7,<1.0.0"]' in metadata
    assert 'async = ["llm-api-adapter[async]>=0.9.7,<1.0.0"]' in metadata
    assert 'httpx = ["llm-api-adapter[httpx]>=0.9.7,<1.0.0"]' in metadata
    assert 'zai = "llm_api_adapter_zai.plugin:PLUGIN"' in metadata
    assert 'dependencies = ["llm-api-adapter-zai' not in metadata


@pytest.mark.unit
def test_zai_plugin_entry_point_matches_the_core_contract():
    from llm_api_adapter_zai.plugin import PLUGIN

    assert isinstance(PLUGIN, OrganizationPlugin)
    assert PLUGIN.api_version == ORGANIZATION_PLUGIN_API_VERSION
    assert callable(PLUGIN.register)


@pytest.mark.unit
def test_zai_plugin_entry_point_registers_the_zai_service_provider():
    from llm_api_adapter_zai.plugin import PLUGIN

    registry = ServiceProviderRegistry()
    PLUGIN.register(registry)

    assert registry.get("zai") is not None
