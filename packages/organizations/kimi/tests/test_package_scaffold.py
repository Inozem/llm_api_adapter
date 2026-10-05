"""Deterministic checks for the independently buildable Kimi package scaffold."""

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
from tests.external_organization_metadata import (
    read_project_metadata,
    requirement_target,
)


@pytest.mark.unit
def test_kimi_plugin_entry_point_matches_the_core_contract():
    from llm_api_adapter_kimi.plugin import PLUGIN

    assert isinstance(PLUGIN, OrganizationPlugin)
    assert PLUGIN.api_version == ORGANIZATION_PLUGIN_API_VERSION
    assert callable(PLUGIN.register)


@pytest.mark.unit
def test_kimi_project_forwards_only_core_transport_extras():
    project = read_project_metadata(PACKAGE_ROOT / "pyproject.toml")

    assert [requirement_target(item) for item in project["dependencies"]] == [
        "llm-api-adapter"
    ]
    for extra in ("async", "httpx"):
        assert [
            requirement_target(item)
            for item in project["optional-dependencies"][extra]
        ] == [f"llm-api-adapter[{extra}]"]
    assert project["entry-points"]["llm_api_adapter.organizations"]["kimi"] == (
        "llm_api_adapter_kimi.plugin:PLUGIN"
    )
