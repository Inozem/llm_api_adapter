"""Tests for the independently buildable Mistral package skeleton."""

from __future__ import annotations

from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_distribution_metadata_declares_the_core_compatibility_range():
    metadata = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "llm-api-adapter-mistral"' in metadata
    assert 'version = "0.1.0"' in metadata
    assert 'llm-api-adapter>=0.9.0,<1.0.0' in metadata


@pytest.mark.unit
def test_transport_extras_are_forwarded_and_composable():
    metadata = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'async = ["llm-api-adapter[async]>=0.9.0,<1.0.0"]' in metadata
    assert 'httpx = ["llm-api-adapter[httpx]>=0.9.0,<1.0.0"]' in metadata


@pytest.mark.unit
def test_distribution_exposes_the_provider_plugin_entry_point():
    metadata = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '[project.entry-points."llm_api_adapter.providers"]' in metadata
    assert 'mistral = "llm_api_adapter_mistral.plugin:PLUGIN"' in metadata


@pytest.mark.unit
def test_source_layout_uses_dedicated_clients_over_core_transports():
    source_root = PACKAGE_ROOT / "src" / "llm_api_adapter_mistral"
    implementation_paths = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*.py")
    }
    transport_implementation_paths = {
        path
        for path in implementation_paths
        if path.endswith((
            "transport.py",
            "transports.py",
        ))
    }

    assert transport_implementation_paths == set()
    assert {
        "clients/sync_client.py",
        "clients/async_client.py",
    }.issubset(implementation_paths)
    assert (source_root / "py.typed").is_file()
