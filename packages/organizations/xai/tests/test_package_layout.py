"""Tests for the independently buildable xAI package."""

from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_distribution_metadata_declares_the_core_compatibility_range():
    metadata = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "llm-api-adapter-xai"' in metadata
    assert 'version = "0.1.0"' in metadata
    assert 'llm-api-adapter>=0.9.0,<1.0.0' in metadata


@pytest.mark.unit
def test_transport_extras_are_forwarded_and_composable():
    metadata = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'async = ["llm-api-adapter[async]>=0.9.0,<1.0.0"]' in metadata
    assert 'httpx = ["llm-api-adapter[httpx]>=0.9.0,<1.0.0"]' in metadata


@pytest.mark.unit
def test_distribution_exposes_the_organization_plugin_entry_point():
    metadata = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '[project.entry-points."llm_api_adapter.organizations"]' in metadata
    assert 'xai = "llm_api_adapter_xai.plugin:PLUGIN"' in metadata


@pytest.mark.unit
def test_source_layout_uses_the_core_transport_contract():
    source_root = PACKAGE_ROOT / "src" / "llm_api_adapter_xai"
    source_files = list(source_root.rglob("*.py"))
    source_text = "\n".join(
        path.read_text(encoding="utf-8") for path in source_files
    )

    assert "create_sync_transport" in source_text
    assert "import requests" not in source_text
    assert "import httpx" not in source_text
    assert (source_root / "py.typed").is_file()
