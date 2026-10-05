"""Repository metadata checks for independently released organization packages."""

from __future__ import annotations

from pathlib import Path
import shutil

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ORGANIZATIONS = ("kimi", "mistral", "qwen", "xai", "deepseek", "zai")
REGISTRY = "src/llm_api_adapter/organization_registry.py"
E2E_PROFILES = "tests/e2e/conftest.py"
CI_LANES = ".github/scripts/select_e2e_lanes.py"


@pytest.fixture
def metadata_repository(tmp_path: Path) -> Path:
    paths = [REGISTRY, "pyproject.toml", E2E_PROFILES, CI_LANES]
    paths.extend(
        f"packages/organizations/{organization}/pyproject.toml"
        for organization in ORGANIZATIONS
    )
    for relative_path in paths:
        destination = tmp_path / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY_ROOT / relative_path, destination)
    return tmp_path


def _replace_once(root: Path, relative_path: str, old: str, new: str) -> None:
    path = root / relative_path
    source = path.read_text(encoding="utf-8")
    assert source.count(old) == 1, f"Expected one occurrence of {old!r} in {path}"
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def _validate(root: Path) -> tuple[str, ...]:
    from tests.external_organization_metadata import (
        validate_external_organization_metadata,
    )

    return tuple(validate_external_organization_metadata(root))


def _assert_issue(issues: tuple[str, ...], organization: str, source: str) -> None:
    assert any(
        organization in issue and source in issue.replace("\\", "/")
        for issue in issues
    ), issues


@pytest.mark.unit
def test_six_external_organizations_have_consistent_metadata(
    metadata_repository: Path,
) -> None:
    assert not _validate(metadata_repository)


@pytest.mark.unit
def test_checked_out_repository_has_all_external_organization_metadata() -> None:
    from tests.external_organization_metadata import (
        read_external_organization_metadata,
    )

    metadata = read_external_organization_metadata(REPOSITORY_ROOT)
    for source in (
        metadata.known_packages,
        metadata.extras,
        metadata.packages,
        metadata.entry_points,
    ):
        assert set(source) == set(ORGANIZATIONS)
    assert not _validate(REPOSITORY_ROOT)


@pytest.mark.unit
@pytest.mark.parametrize("organization", ORGANIZATIONS)
@pytest.mark.parametrize(
    "missing_source",
    ("core_package", "package", "extra", "entry_point", "e2e_profile", "ci_lane"),
)
def test_missing_external_metadata_reports_organization_and_source(
    metadata_repository: Path,
    organization: str,
    missing_source: str,
) -> None:
    package_manifest = f"packages/organizations/{organization}/pyproject.toml"
    expected_source = {
        "core_package": REGISTRY,
        "package": package_manifest,
        "extra": "pyproject.toml",
        "entry_point": package_manifest,
        "e2e_profile": E2E_PROFILES,
        "ci_lane": CI_LANES,
    }[missing_source]

    if missing_source == "core_package":
        path = metadata_repository / REGISTRY
        source = path.read_text(encoding="utf-8")
        start = source.index(f'    "{organization}": KnownOrganizationPackage(')
        end = source.index("    ),\n", start) + len("    ),\n")
        _replace_once(metadata_repository, REGISTRY, source[start:end], "")
    elif missing_source == "package":
        (metadata_repository / package_manifest).unlink()
    elif missing_source == "extra":
        path = metadata_repository / "pyproject.toml"
        line = next(
            line for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
            if line.startswith(f"{organization} = [")
        )
        _replace_once(metadata_repository, "pyproject.toml", line, "")
    elif missing_source == "entry_point":
        _replace_once(
            metadata_repository,
            package_manifest,
            f'{organization} = "llm_api_adapter_{organization}.plugin:PLUGIN"',
            "",
        )
    elif missing_source == "e2e_profile":
        _replace_once(
            metadata_repository,
            E2E_PROFILES,
            f"    _{organization.upper()}_E2E_PROFILE,\n",
            "",
        )
    else:
        _replace_once(
            metadata_repository,
            CI_LANES,
            f'            "{organization}_e2e": str(self.{organization}_e2e).lower(),',
            "",
        )

    _assert_issue(_validate(metadata_repository), organization, expected_source)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("old", "new", "expected"),
    (
        ("    qwen_e2e: bool\n", "", "missing field"),
        (
            "        qwen_e2e=(\n",
            "        unused_qwen_e2e=(\n",
            "missing selector route",
        ),
    ),
    ids=("selection-field", "selection-route"),
)
def test_missing_ci_selection_metadata_is_reported(
    metadata_repository: Path,
    old: str,
    new: str,
    expected: str,
) -> None:
    _replace_once(metadata_repository, CI_LANES, old, new)

    issues = _validate(metadata_repository)
    _assert_issue(issues, "qwen", CI_LANES)
    assert any(expected in issue for issue in issues), issues


@pytest.mark.unit
def test_duplicate_core_organization_key_is_reported(metadata_repository: Path) -> None:
    path = metadata_repository / REGISTRY
    source = path.read_text(encoding="utf-8")
    start = source.index('    "kimi": KnownOrganizationPackage(')
    end = source.index("    ),\n", start) + len("    ),\n")
    _replace_once(metadata_repository, REGISTRY, source[start:end], source[start:end] * 2)

    issues = _validate(metadata_repository)
    _assert_issue(issues, "kimi", REGISTRY)
    assert any("duplicate" in issue.lower() for issue in issues), issues


@pytest.mark.unit
@pytest.mark.parametrize("source", ("package", "extra"))
def test_equivalent_normalized_distribution_names_are_accepted(
    metadata_repository: Path,
    source: str,
) -> None:
    if source == "package":
        _replace_once(
            metadata_repository,
            "packages/organizations/qwen/pyproject.toml",
            'name = "llm-api-adapter-qwen"',
            'name = "llm_api_adapter_qwen"',
        )
    else:
        _replace_once(
            metadata_repository,
            "pyproject.toml",
            'qwen = ["llm-api-adapter-qwen',
            'qwen = ["llm_api_adapter_qwen',
        )

    assert not _validate(metadata_repository)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source", "relative_path", "old", "new"),
    (
        (
            "package",
            "packages/organizations/qwen/pyproject.toml",
            'name = "llm-api-adapter-qwen"',
            'name = "llm-api-adapter-other"',
        ),
        (
            "extra",
            "pyproject.toml",
            'qwen = ["llm-api-adapter-qwen',
            'qwen = ["llm-api-adapter-other',
        ),
        (
            "e2e_profile",
            E2E_PROFILES,
            '    distribution=KNOWN_ORGANIZATION_PACKAGES["qwen"].distribution,',
            '    distribution="llm-api-adapter-other",',
        ),
        (
            "core_package",
            REGISTRY,
            '        distribution="llm-api-adapter-qwen",',
            '        distribution="llm-api-adapter-other",',
        ),
    ),
    ids=("package", "extra", "e2e-profile", "core-package"),
)
def test_distribution_mismatch_reports_organization_and_source(
    metadata_repository: Path,
    source: str,
    relative_path: str,
    old: str,
    new: str,
) -> None:
    _replace_once(metadata_repository, relative_path, old, new)

    issues = _validate(metadata_repository)
    _assert_issue(issues, "qwen", relative_path)
    assert any("distribution" in issue.lower() for issue in issues), issues
