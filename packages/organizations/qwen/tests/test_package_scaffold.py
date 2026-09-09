"""Deterministic checks for the Qwen package scaffold."""

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
def test_qwen_plugin_entry_point_matches_the_core_contract():
    from llm_api_adapter_qwen.plugin import PLUGIN

    assert isinstance(PLUGIN, OrganizationPlugin)
    assert PLUGIN.api_version == ORGANIZATION_PLUGIN_API_VERSION
    assert callable(PLUGIN.register)
