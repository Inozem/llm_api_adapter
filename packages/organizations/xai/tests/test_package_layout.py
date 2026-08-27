"""Tests for the independently buildable xAI package skeleton."""

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
def test_source_layout_has_no_direct_transport_implementation():
    source_root = PACKAGE_ROOT / "src" / "llm_api_adapter_xai"
    implementation_paths = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*.py")
    }
    direct_transport_paths = {
        path
        for path in implementation_paths
        if path.endswith((
            "transport.py",
            "transports.py",
            "sync_client.py",
            "async_client.py",
            "streaming.py",
            "async_streaming.py",
        ))
    }

    assert direct_transport_paths == set()
    assert (source_root / "py.typed").is_file()
