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
from tests.external_organization_metadata import (
    read_project_metadata,
    requirement_target,
)


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
    project = read_project_metadata(PACKAGE_ROOT / "pyproject.toml")

    assert project["name"] == "llm-api-adapter-zai"
    assert [requirement_target(item) for item in project["dependencies"]] == [
        "llm-api-adapter"
    ]
    for extra in ("async", "httpx"):
        assert [
            requirement_target(item)
            for item in project["optional-dependencies"][extra]
        ] == [f"llm-api-adapter[{extra}]"]
    assert project["entry-points"]["llm_api_adapter.organizations"]["zai"] == (
        "llm_api_adapter_zai.plugin:PLUGIN"
    )


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
