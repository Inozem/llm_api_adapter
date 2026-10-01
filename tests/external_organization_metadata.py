"""Read repository metadata without importing organization packages or runtime code."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 test jobs use the backport.
    import tomli as tomllib


REGISTRY = "src/llm_api_adapter/organization_registry.py"
E2E_PROFILES = "tests/e2e/conftest.py"
CI_LANES = ".github/scripts/select_e2e_lanes.py"
ENTRY_POINT_GROUP = "llm_api_adapter.organizations"


@dataclass
class ExternalOrganizationMetadata:
    """Values observed in the independent repository metadata sources."""

    known_packages: dict[str, str] = field(default_factory=dict)
    extras: dict[str, str] = field(default_factory=dict)
    package_directories: set[str] = field(default_factory=set)
    packages: dict[str, str] = field(default_factory=dict)
    entry_points: dict[str, str] = field(default_factory=dict)
    e2e_profiles: dict[str, str | None] = field(default_factory=dict)
    ci_fields: set[str] = field(default_factory=set)
    ci_outputs: set[str] = field(default_factory=set)
    ci_routes: set[str] = field(default_factory=set)
    issues: list[str] = field(default_factory=list)


def _assignment(body: list[ast.stmt], name: str) -> ast.expr | None:
    for statement in body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in statement.targets
        ):
            return statement.value
    return None


def _python_tree(root: Path, relative_path: str, issues: list[str]) -> ast.Module:
    try:
        return ast.parse((root / relative_path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as error:
        issues.append(f"{relative_path}: cannot read Python metadata: {error}")
        return ast.parse("")


def _toml(root: Path, relative_path: str, issues: list[str]) -> dict[str, Any]:
    try:
        with (root / relative_path).open("rb") as source:
            return tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        issues.append(f"{relative_path}: cannot read TOML metadata: {error}")
        return {}


def _normalized_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str | None:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    return match.group(1) if match else None


def _read_known_packages(root: Path, metadata: ExternalOrganizationMetadata) -> None:
    tree = _python_tree(root, REGISTRY, metadata.issues)
    mapping = _assignment(tree.body, "KNOWN_ORGANIZATION_PACKAGES")
    if not isinstance(mapping, ast.Dict):
        metadata.issues.append(
            f"{REGISTRY}: missing KNOWN_ORGANIZATION_PACKAGES mapping"
        )
        return

    for key, value in zip(mapping.keys, mapping.values):
        try:
            organization = ast.literal_eval(key)
            assert isinstance(value, ast.Call)
            keywords = {
                item.arg: ast.literal_eval(item.value) for item in value.keywords
            }
            declared_organization = keywords["organization"]
            distribution = keywords["distribution"]
            assert isinstance(organization, str) and isinstance(distribution, str)
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            metadata.issues.append(f"{REGISTRY}: invalid known package entry: {error}")
            continue
        if organization in metadata.known_packages:
            metadata.issues.append(
                f"{REGISTRY}: duplicate organization key {organization!r}"
            )
        if declared_organization != organization:
            metadata.issues.append(
                f"{REGISTRY}: {organization}: declared organization "
                f"{declared_organization!r} differs from its key"
            )
        metadata.known_packages[organization] = distribution


def _read_core_extras(root: Path, metadata: ExternalOrganizationMetadata) -> None:
    project = _toml(root, "pyproject.toml", metadata.issues).get("project", {})
    for organization, requirements in project.get("optional-dependencies", {}).items():
        if organization in {"async", "httpx"}:
            continue
        if not isinstance(requirements, list) or len(requirements) != 1:
            metadata.issues.append(
                f"pyproject.toml: {organization}: expected one distribution requirement"
            )
            continue
        distribution = _requirement_name(requirements[0])
        if distribution is None:
            metadata.issues.append(
                f"pyproject.toml: {organization}: invalid distribution requirement"
            )
            continue
        metadata.extras[organization] = distribution


def _read_package_manifests(root: Path, metadata: ExternalOrganizationMetadata) -> None:
    package_root = root / "packages" / "organizations"
    if not package_root.is_dir():
        metadata.issues.append("packages/organizations: missing package directory")
        return
    for directory in sorted(path for path in package_root.iterdir() if path.is_dir()):
        organization = directory.name
        metadata.package_directories.add(organization)
        relative_path = f"packages/organizations/{organization}/pyproject.toml"
        if not (root / relative_path).is_file():
            continue
        project = _toml(root, relative_path, metadata.issues).get("project", {})
        distribution = project.get("name")
        if isinstance(distribution, str):
            metadata.packages[organization] = distribution
        entry_points = project.get("entry-points", {}).get(ENTRY_POINT_GROUP, {})
        if isinstance(entry_points, dict):
            for entry_name, target in entry_points.items():
                if entry_name != organization:
                    metadata.issues.append(
                        f"{relative_path}: {organization}: unexpected entry point "
                        f"{entry_name!r}"
                    )
                elif isinstance(target, str):
                    metadata.entry_points[organization] = target


def _profile_distribution(
    node: ast.expr | None,
    known_packages: dict[str, str],
) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Constant) and (
        node.value is None or isinstance(node.value, str)
    ):
        return node.value
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "distribution"
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "KNOWN_ORGANIZATION_PACKAGES"
    ):
        organization = ast.literal_eval(node.value.slice)
        return known_packages.get(organization)
    raise ValueError("unsupported E2E distribution expression")


def _read_e2e_profiles(root: Path, metadata: ExternalOrganizationMetadata) -> None:
    tree = _python_tree(root, E2E_PROFILES, metadata.issues)
    assignments = {
        target.id: statement.value
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }
    active = assignments.get("_E2E_PROFILES")
    if not isinstance(active, (ast.Tuple, ast.List)):
        metadata.issues.append(f"{E2E_PROFILES}: missing _E2E_PROFILES sequence")
        return
    for item in active.elts:
        if not isinstance(item, ast.Name) or not isinstance(
            assignments.get(item.id), ast.Call
        ):
            metadata.issues.append(f"{E2E_PROFILES}: invalid active E2E profile")
            continue
        profile = assignments[item.id]
        assert isinstance(profile, ast.Call)
        keywords = {keyword.arg: keyword.value for keyword in profile.keywords}
        try:
            organization = ast.literal_eval(keywords["name"])
            distribution = _profile_distribution(
                keywords.get("distribution"), metadata.known_packages
            )
        except (KeyError, TypeError, ValueError) as error:
            metadata.issues.append(
                f"{E2E_PROFILES}: invalid profile {item.id}: {error}"
            )
            continue
        if organization in metadata.e2e_profiles:
            metadata.issues.append(
                f"{E2E_PROFILES}: duplicate active E2E profile {organization!r}"
            )
        metadata.e2e_profiles[organization] = distribution


def _read_ci_lanes(root: Path, metadata: ExternalOrganizationMetadata) -> None:
    tree = _python_tree(root, CI_LANES, metadata.issues)
    lane_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "E2ELaneSelection"
        ),
        None,
    )
    if lane_class is None:
        metadata.issues.append(f"{CI_LANES}: missing E2ELaneSelection")
        return
    metadata.ci_fields = {
        node.target.id
        for node in lane_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    github_outputs = next(
        (
            node
            for node in lane_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "github_outputs"
        ),
        None,
    )
    output_mapping = (
        next(
            (
                node.value
                for node in github_outputs.body
                if isinstance(node, ast.Return)
            ),
            None,
        )
        if github_outputs
        else None
    )
    if isinstance(output_mapping, ast.Dict):
        for key in output_mapping.keys:
            try:
                output = ast.literal_eval(key)
            except (TypeError, ValueError):
                metadata.issues.append(f"{CI_LANES}: invalid GitHub output key")
                continue
            if output in metadata.ci_outputs:
                metadata.issues.append(
                    f"{CI_LANES}: duplicate GitHub output {output!r}"
                )
            metadata.ci_outputs.add(output)
    else:
        metadata.issues.append(f"{CI_LANES}: missing github_outputs mapping")

    selector = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "select_e2e_lanes"
        ),
        None,
    )
    if selector is None:
        metadata.issues.append(f"{CI_LANES}: missing select_e2e_lanes function")
        return
    selections = [
        node
        for node in ast.walk(selector)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "E2ELaneSelection"
    ]
    if len(selections) != 1:
        metadata.issues.append(f"{CI_LANES}: expected one E2ELaneSelection call")
        return
    metadata.ci_routes = {keyword.arg for keyword in selections[0].keywords}


def read_external_organization_metadata(root: Path) -> ExternalOrganizationMetadata:
    """Read the five independent metadata sources without executing their code."""
    metadata = ExternalOrganizationMetadata()
    _read_known_packages(root, metadata)
    _read_core_extras(root, metadata)
    _read_package_manifests(root, metadata)
    _read_e2e_profiles(root, metadata)
    _read_ci_lanes(root, metadata)
    return metadata


def validate_external_organization_metadata(root: Path) -> tuple[str, ...]:
    """Report missing or conflicting metadata with organization and source paths."""
    metadata = read_external_organization_metadata(root)
    issues = list(metadata.issues)
    organizations = (
        set(metadata.known_packages)
        | set(metadata.extras)
        | metadata.package_directories
        | {name for name, distribution in metadata.e2e_profiles.items() if distribution}
        | {
            output.removesuffix("_e2e")
            for output in metadata.ci_outputs
            if output.endswith("_e2e") and not output.startswith("core_")
        }
    )

    for organization in sorted(organizations):
        package_path = f"packages/organizations/{organization}/pyproject.toml"
        sources = {
            REGISTRY: metadata.known_packages.get(organization),
            "pyproject.toml": metadata.extras.get(organization),
            package_path: metadata.packages.get(organization),
            E2E_PROFILES: metadata.e2e_profiles.get(organization),
        }
        for source, distribution in sources.items():
            if distribution is None:
                issues.append(
                    f"{source}: {organization}: missing distribution metadata"
                )
        if organization not in metadata.entry_points:
            issues.append(
                f"{package_path}: {organization}: missing "
                f"{ENTRY_POINT_GROUP} entry point"
            )
        else:
            expected_target = f"llm_api_adapter_{organization}.plugin:PLUGIN"
            if metadata.entry_points[organization] != expected_target:
                issues.append(
                    f"{package_path}: {organization}: entry point target "
                    f"{metadata.entry_points[organization]!r} differs from "
                    f"{expected_target!r}"
                )
        for lane in (organization, f"{organization}_e2e"):
            for label, values in (
                ("field", metadata.ci_fields),
                ("GitHub output", metadata.ci_outputs),
                ("selector route", metadata.ci_routes),
            ):
                if lane not in values:
                    issues.append(
                        f"{CI_LANES}: {organization}: missing {label} {lane!r}"
                    )

        present = {
            source: _normalized_distribution(distribution)
            for source, distribution in sources.items()
            if distribution is not None
        }
        counts = Counter(present.values())
        if len(counts) > 1:
            highest = max(counts.values())
            leaders = {name for name, count in counts.items() if count == highest}
            for source, distribution in present.items():
                if len(leaders) != 1 or distribution not in leaders:
                    issues.append(
                        f"{source}: {organization}: distribution {distribution!r} "
                        f"conflicts with other metadata sources"
                    )

    return tuple(issues)
