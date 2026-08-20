"""Tests for the independently buildable Mistral package skeleton."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _write_pure_python_wheel(
    directory: Path,
    *,
    distribution: str,
    package: str,
    version: str,
    requires_dist: str | None = None,
) -> Path:
    """Create a minimal valid wheel for the isolated-install check."""
    normalized_name = distribution.replace("-", "_")
    wheel_path = directory / f"{normalized_name}-{version}-py3-none-any.whl"
    dist_info = f"{normalized_name}-{version}.dist-info"
    metadata_lines = [
        "Metadata-Version: 2.1",
        f"Name: {distribution}",
        f"Version: {version}",
    ]
    if requires_dist is not None:
        metadata_lines.append(f"Requires-Dist: {requires_dist}")

    contents = {
        f"{package}/__init__.py": "",
        f"{dist_info}/METADATA": "\n".join(metadata_lines) + "\n",
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\n"
            "Generator: llm-api-adapter tests\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ),
        f"{dist_info}/RECORD": "",
    }
    with ZipFile(wheel_path, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    return wheel_path


def _load_wheel_check_module():
    script_path = PACKAGE_ROOT / "scripts" / "check_isolated_wheel_install.py"
    specification = importlib.util.spec_from_file_location(
        "mistral_wheel_check",
        script_path,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


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
def test_source_layout_has_no_direct_transport_implementation():
    source_root = PACKAGE_ROOT / "src" / "llm_api_adapter_mistral"
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


@pytest.mark.unit
def test_isolated_wheel_check_installs_compatible_candidate_wheels(tmp_path):
    core_wheel = _write_pure_python_wheel(
        tmp_path,
        distribution="llm-api-adapter",
        package="llm_api_adapter",
        version="0.9.0",
    )
    mistral_wheel = _write_pure_python_wheel(
        tmp_path,
        distribution="llm-api-adapter-mistral",
        package="llm_api_adapter_mistral",
        version="0.1.0",
        requires_dist="llm-api-adapter>=0.9.0,<1.0.0",
    )

    wheel_check = _load_wheel_check_module()
    wheel_check.check_installation(core_wheel, mistral_wheel)
