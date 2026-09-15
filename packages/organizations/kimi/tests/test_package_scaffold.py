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


@pytest.mark.unit
def test_kimi_plugin_entry_point_matches_the_core_contract():
    from llm_api_adapter_kimi.plugin import PLUGIN

    assert isinstance(PLUGIN, OrganizationPlugin)
    assert PLUGIN.api_version == ORGANIZATION_PLUGIN_API_VERSION
    assert callable(PLUGIN.register)


@pytest.mark.unit
def test_kimi_project_forwards_only_core_transport_extras():
    metadata = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'dependencies = ["llm-api-adapter>=0.9.5,<1.0.0"]' in metadata
    assert 'async = ["llm-api-adapter[async]>=0.9.5,<1.0.0"]' in metadata
    assert 'httpx = ["llm-api-adapter[httpx]>=0.9.5,<1.0.0"]' in metadata
    assert 'kimi = "llm_api_adapter_kimi.plugin:PLUGIN"' in metadata
