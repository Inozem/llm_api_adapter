import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SELECTOR_PATH = _REPOSITORY_ROOT / ".github" / "scripts" / "select_e2e_lanes.py"
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "ci_e2e_lane_selector", _SELECTOR_PATH
)
assert _MODULE_SPEC is not None
assert _MODULE_SPEC.loader is not None
_SELECTOR = importlib.util.module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = _SELECTOR
_MODULE_SPEC.loader.exec_module(_SELECTOR)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("changed_paths", "core", "shared_core", "organizations", "mistral_e2e", "xai_e2e"),
    [
        (
            ["src/llm_api_adapter/llm_registry/organizations/anthropic.json"],
            True,
            False,
            ("anthropic",),
            False,
            False,
        ),
        (
            ["src/llm_api_adapter/llms/google/async_client.py"],
            True,
            False,
            ("google",),
            False,
            False,
        ),
        (
            ["src/llm_api_adapter/adapters/structured_output.py"],
            True,
            True,
            ("openai", "anthropic", "google"),
            True,
            True,
        ),
        (
            ["src/llm_api_adapter/llm_registry/llm_registry.json"],
            True,
            False,
            ("openai", "anthropic", "google"),
            False,
            False,
        ),
        (
            ["src/llm_api_adapter/new_shared_module.py"],
            True,
            True,
            ("openai", "anthropic", "google"),
            True,
            True,
        ),
        (
            ["packages/organizations/mistral/src/llm_api_adapter_mistral/adapter.py"],
            False,
            False,
            (),
            True,
            False,
        ),
        (
            ["tests/e2e/test_tools_auto_loop.py"],
            False,
            False,
            (),
            True,
            True,
        ),
    ],
)
def test_select_e2e_lanes_classifies_shared_and_organization_paths(
    changed_paths,
    core,
    shared_core,
    organizations,
    mistral_e2e,
    xai_e2e,
):
    selection = _SELECTOR.select_e2e_lanes(changed_paths)

    assert selection.core is core
    assert selection.shared_core is shared_core
    assert selection.core_organizations == organizations
    assert selection.mistral_e2e is mistral_e2e
    assert selection.xai_e2e is xai_e2e
    outputs = selection.github_outputs()
    for organization in ("openai", "anthropic", "google"):
        assert outputs[f"core_{organization}_e2e"] == str(
            organization in organizations
        ).lower()


@pytest.mark.unit
def test_selector_cli_writes_github_outputs(tmp_path):
    github_output = tmp_path / "github_output"
    result = subprocess.run(
        [
            sys.executable,
            str(_SELECTOR_PATH),
            "--github-output",
            str(github_output),
        ],
        input="src/llm_api_adapter/llm_registry/organizations/openai.json\n",
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    outputs = dict(
        line.split("=", maxsplit=1)
        for line in github_output.read_text(encoding="utf-8").splitlines()
    )
    assert outputs["core"] == "true"
    assert outputs["shared_core"] == "false"
    assert outputs["core_openai_e2e"] == "true"
    assert outputs["core_anthropic_e2e"] == "false"
    assert outputs["core_google_e2e"] == "false"
    assert "core_e2e_matrix" not in outputs
    assert outputs["mistral_e2e"] == "false"
    assert outputs["xai_e2e"] == "false"
