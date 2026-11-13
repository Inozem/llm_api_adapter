"""Deterministic checks for the independently buildable Z.ai package scaffold."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


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


CORE_REQUIREMENT_SPECIFIER = SpecifierSet(">=0.10.0,<1.0.0")
ROOT_ZAI_REQUIREMENT_SPECIFIER = SpecifierSet(">=0.1.2,<0.2.0")


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
def test_zai_project_declares_migrated_package_version():
    """The migrated Z.ai package declares its planned release version."""
    project = read_project_metadata(PACKAGE_ROOT / "pyproject.toml")

    assert project["version"] == "0.1.3"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dependency_group", "expected_extra"),
    [("dependencies", None), ("async", "async"), ("httpx", "httpx")],
    ids=["base", "async", "httpx"],
)
def test_zai_core_dependency_groups_require_supported_release(
    dependency_group: str, expected_extra: str | None
):
    """Every declared Core path covers the shared foundation's supported range."""
    project = read_project_metadata(PACKAGE_ROOT / "pyproject.toml")
    if dependency_group == "dependencies":
        requirements = project["dependencies"]
    else:
        requirements = project["optional-dependencies"][dependency_group]

    assert len(requirements) == 1
    parsed = Requirement(requirements[0])
    assert parsed.name.lower().replace("_", "-") == "llm-api-adapter"
    assert parsed.extras == ({expected_extra} if expected_extra else set())
    assert parsed.url is None
    assert parsed.marker is None
    assert parsed.specifier == CORE_REQUIREMENT_SPECIFIER
    assert all(
        parsed.specifier.contains(version)
        for version in ("0.10.0", "0.10.1", "0.11.0", "0.99.0")
    )
    assert not any(
        parsed.specifier.contains(version)
        for version in ("0.9.8", "0.9.9", "0.9.10", "1.0.0", "1.1.0")
    )


@pytest.mark.unit
def test_zai_root_metadata_keeps_core_and_extra_compatible():
    """The root release stays fixed while its Z.ai extra admits this package."""
    root_project = read_project_metadata(REPOSITORY_ROOT / "pyproject.toml")
    assert root_project["version"] == "0.10.0"
    assert root_project["dependencies"] == ["requests>=2.32"]

    raw_requirement = root_project["optional-dependencies"]["zai"]
    assert raw_requirement == ["llm-api-adapter-zai>=0.1.2,<0.2.0"]
    parsed = Requirement(raw_requirement[0])
    assert parsed.name.lower().replace("_", "-") == "llm-api-adapter-zai"
    assert parsed.extras == set()
    assert parsed.url is None
    assert parsed.marker is None
    assert parsed.specifier == ROOT_ZAI_REQUIREMENT_SPECIFIER
    assert parsed.specifier.contains("0.1.2")
    assert parsed.specifier.contains("0.1.3")


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
