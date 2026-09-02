"""Select post-publish E2E lanes from a newline-delimited changed-path list."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Iterable


_CORE_CANDIDATE_PATHS = (
    "pyproject.toml",
    "README.md",
    "src/llm_api_adapter/",
)
_SHARED_CORE_PATHS = (
    "pyproject.toml",
    "src/llm_api_adapter/organization_registry.py",
    "src/llm_api_adapter/service_provider_registry.py",
    "src/llm_api_adapter/universal_adapter.py",
    "src/llm_api_adapter/adapters/base_adapter.py",
    "src/llm_api_adapter/adapters/structured_output.py",
    "src/llm_api_adapter/errors/",
    "src/llm_api_adapter/llm_registry/llm_registry.py",
    "src/llm_api_adapter/llm_registry/reasoning.py",
    "src/llm_api_adapter/llm_registry/request_rules.py",
    "src/llm_api_adapter/llms/async_streaming.py",
    "src/llm_api_adapter/llms/httpx_transport.py",
    "src/llm_api_adapter/llms/request_rules.py",
    "src/llm_api_adapter/llms/requests_transport.py",
    "src/llm_api_adapter/llms/streaming.py",
    "src/llm_api_adapter/llms/transports.py",
    "src/llm_api_adapter/models/",
)
_CORE_ONLY_SHARED_PATHS = (
    "README.md",
    "src/llm_api_adapter/__init__.py",
    "src/llm_api_adapter/adapters/__init__.py",
    "src/llm_api_adapter/llm_registry/__init__.py",
    "src/llm_api_adapter/llm_registry/llm_registry.json",
    "src/llm_api_adapter/llms/__init__.py",
)
_E2E_HARNESS_PATHS = (
    "tests/e2e/",
    "tests/fixtures/",
    "tests/requirements-test.txt",
    "pytest.ini",
    ".github/workflows/ci-dev-release.yml",
    ".github/scripts/",
)
_CORE_ORGANIZATION_PATHS = {
    "openai": (
        "src/llm_api_adapter/adapters/openai/",
        "src/llm_api_adapter/adapters/openai_adapter.py",
        "src/llm_api_adapter/adapters/openai_payloads.py",
        "src/llm_api_adapter/adapters/openai_streaming.py",
        "src/llm_api_adapter/llms/openai/",
        "src/llm_api_adapter/llm_registry/organizations/openai.json",
    ),
    "anthropic": (
        "src/llm_api_adapter/adapters/anthropic/",
        "src/llm_api_adapter/adapters/anthropic_adapter.py",
        "src/llm_api_adapter/adapters/anthropic_payloads.py",
        "src/llm_api_adapter/adapters/anthropic_streaming.py",
        "src/llm_api_adapter/llms/anthropic/",
        "src/llm_api_adapter/llm_registry/organizations/anthropic.json",
    ),
    "google": (
        "src/llm_api_adapter/adapters/google/",
        "src/llm_api_adapter/adapters/google_adapter.py",
        "src/llm_api_adapter/adapters/google_payloads.py",
        "src/llm_api_adapter/adapters/google_streaming.py",
        "src/llm_api_adapter/llms/google/",
        "src/llm_api_adapter/llm_registry/organizations/google.json",
    ),
}
_CORE_LANE_METADATA = {
    "openai": {
        "organization": "OpenAI",
        "marker": "e2e_openai",
        "api_key_env": "OPENAI_API_KEY",
    },
    "anthropic": {
        "organization": "Anthropic",
        "marker": "e2e_anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "google": {
        "organization": "Google",
        "marker": "e2e_google",
        "api_key_env": "GOOGLE_API_KEY",
    },
}
_CORE_ORGANIZATIONS = tuple(_CORE_LANE_METADATA)


def _matches(paths: tuple[str, ...], prefixes: tuple[str, ...]) -> bool:
    return any(path.startswith(prefix) for path in paths for prefix in prefixes)


def _organization_package_changed(paths: tuple[str, ...], organization: str) -> bool:
    root = f"packages/organizations/{organization}/"
    package_files = ("MANIFEST.in", "README.md", "pyproject.toml")
    return any(
        path.startswith(f"{root}src/")
        or path in {f"{root}{name}" for name in package_files}
        for path in paths
    )


@dataclass(frozen=True)
class E2ELaneSelection:
    """Release-candidate distributions and E2E lanes selected by changed paths."""

    core: bool
    shared_core: bool
    core_organizations: tuple[str, ...]
    mistral: bool
    xai: bool
    mistral_e2e: bool
    xai_e2e: bool

    def github_outputs(self) -> dict[str, str]:
        matrix = {
            "include": [
                _CORE_LANE_METADATA[organization]
                for organization in self.core_organizations
            ]
        }
        return {
            "core": str(self.core).lower(),
            "shared_core": str(self.shared_core).lower(),
            "core_e2e_matrix": json.dumps(matrix, separators=(",", ":")),
            "mistral": str(self.mistral).lower(),
            "mistral_e2e": str(self.mistral_e2e).lower(),
            "xai": str(self.xai).lower(),
            "xai_e2e": str(self.xai_e2e).lower(),
        }


def select_e2e_lanes(changed_paths: Iterable[str]) -> E2ELaneSelection:
    """Select only the release-candidate lanes affected by ``changed_paths``."""
    paths = tuple(path.strip() for path in changed_paths if path.strip())
    core = _matches(paths, _CORE_CANDIDATE_PATHS)
    shared_core = False
    core_organizations: tuple[str, ...] = ()

    if core:
        if _matches(paths, _SHARED_CORE_PATHS):
            shared_core = True
            core_organizations = _CORE_ORGANIZATIONS
        elif _matches(paths, _CORE_ONLY_SHARED_PATHS + _E2E_HARNESS_PATHS):
            core_organizations = _CORE_ORGANIZATIONS
        else:
            core_organizations = tuple(
                organization
                for organization in _CORE_ORGANIZATIONS
                if _matches(paths, _CORE_ORGANIZATION_PATHS[organization])
            )
            if not core_organizations:
                shared_core = True
                core_organizations = _CORE_ORGANIZATIONS

    e2e_harness = _matches(paths, _E2E_HARNESS_PATHS)
    mistral = _organization_package_changed(paths, "mistral")
    xai = _organization_package_changed(paths, "xai")
    return E2ELaneSelection(
        core=core,
        shared_core=shared_core,
        core_organizations=core_organizations,
        mistral=mistral,
        xai=xai,
        mistral_e2e=shared_core or mistral or e2e_harness,
        xai_e2e=shared_core or xai or e2e_harness,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()

    outputs = select_e2e_lanes(sys.stdin.read().splitlines()).github_outputs()
    with args.github_output.open("a", encoding="utf-8") as output_file:
        for name, value in outputs.items():
            output_file.write(f"{name}={value}\n")


if __name__ == "__main__":
    main()
