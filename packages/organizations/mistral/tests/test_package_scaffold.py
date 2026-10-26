"""Deterministic checks for the independently buildable Mistral package scaffold."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 test jobs use the backport.
    import tomli as tomllib


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parents[2]
CORE_SOURCE = REPOSITORY_ROOT / "src"
PACKAGE_SOURCE = PACKAGE_ROOT / "src"
for source in (str(PACKAGE_SOURCE), str(CORE_SOURCE), str(REPOSITORY_ROOT)):
    if source not in sys.path:
        sys.path.insert(0, source)

from llm_api_adapter.llm_registry.llm_registry import ModelSpec
from llm_api_adapter.organization_registry import (
    ORGANIZATION_PLUGIN_API_VERSION,
    OrganizationPlugin,
)
from tests.external_organization_metadata import (
    read_project_metadata,
    requirement_target,
)


REGISTRY_PATH = (
    PACKAGE_SOURCE
    / "llm_api_adapter_mistral"
    / "registry"
    / "organizations"
    / "mistral.json"
)
CORE_REQUIREMENT_SPECIFIER = SpecifierSet(">=0.9.10,<1.0.0")


@pytest.mark.unit
def test_mistral_plugin_entry_point_matches_the_core_contract():
    project = read_project_metadata(PACKAGE_ROOT / "pyproject.toml")

    assert project["entry-points"]["llm_api_adapter.organizations"]["mistral"] == (
        "llm_api_adapter_mistral.plugin:PLUGIN"
    )

    from llm_api_adapter_mistral.plugin import PLUGIN

    assert isinstance(PLUGIN, OrganizationPlugin)
    assert PLUGIN.api_version == ORGANIZATION_PLUGIN_API_VERSION
    assert callable(PLUGIN.register)


@pytest.mark.unit
def test_mistral_registry_resource_is_packaged_by_existing_rules():
    with (PACKAGE_ROOT / "pyproject.toml").open("rb") as source:
        package_toml = tomllib.load(source)

    package_data = package_toml["tool"]["setuptools"]["package-data"][
        "llm_api_adapter_mistral"
    ]
    assert "py.typed" in package_data
    assert "registry/organizations/*.json" in package_data
    manifest = (PACKAGE_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert (
        "recursive-include src/llm_api_adapter_mistral/registry/organizations *.json"
        in manifest
    )
    assert REGISTRY_PATH.is_file()


@pytest.mark.unit
def test_mistral_registry_contains_native_and_hosted_model_specs():
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    model_data = registry["models"]
    model_specs = {
        name: ModelSpec.from_dict(name, data) for name, data in model_data.items()
    }

    assert any(
        name.startswith("mistral-") and spec.provider_model is None
        for name, spec in model_specs.items()
    )
    hosted_glm = model_specs["glm-5.3-flash"]
    assert hosted_glm.provider_model == "zai-glm-5-3-flash"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("dependency_group", "expected_target"),
    [
        ("dependencies", "llm-api-adapter"),
        ("async", "llm-api-adapter[async]"),
        ("httpx", "llm-api-adapter[httpx]"),
    ],
)
def test_mistral_core_dependency_groups_target_supported_release(
    dependency_group: str, expected_target: str
):
    project = read_project_metadata(PACKAGE_ROOT / "pyproject.toml")
    if dependency_group == "dependencies":
        requirements = project["dependencies"]
    else:
        requirements = project["optional-dependencies"][dependency_group]

    assert len(requirements) == 1
    raw_requirement = requirements[0]
    assert requirement_target(raw_requirement) == expected_target
    parsed = Requirement(raw_requirement)
    assert parsed.specifier == CORE_REQUIREMENT_SPECIFIER
    assert parsed.name.lower().replace("_", "-") != "llm-api-adapter-zai"


@pytest.mark.unit
def test_mistral_project_supports_python_310_and_declares_no_zai_dependency():
    project = read_project_metadata(PACKAGE_ROOT / "pyproject.toml")

    assert SpecifierSet(project["requires-python"]).contains("3.10")
    requirements = list(project["dependencies"])
    for extra_requirements in project["optional-dependencies"].values():
        requirements.extend(extra_requirements)
    assert all(
        Requirement(item).name.lower().replace("_", "-") != "llm-api-adapter-zai"
        for item in requirements
    )
