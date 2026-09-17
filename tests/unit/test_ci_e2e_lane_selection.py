import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SELECTOR_PATH = _REPOSITORY_ROOT / ".github" / "scripts" / "select_e2e_lanes.py"
_WORKFLOW_PATH = _REPOSITORY_ROOT / ".github" / "workflows" / "ci-dev-release.yml"
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
    (
        "changed_paths",
        "core",
        "shared_core",
        "organizations",
        "mistral_e2e",
        "xai_e2e",
        "qwen_e2e",
    ),
    [
        (
            ["src/llm_api_adapter/llm_registry/organizations/anthropic.json"],
            True,
            False,
            ("anthropic",),
            False,
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
            False,
        ),
        (
            ["src/llm_api_adapter/adapters/structured_output.py"],
            True,
            True,
            ("openai", "anthropic", "google"),
            True,
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
            False,
        ),
        (
            ["src/llm_api_adapter/new_shared_module.py"],
            True,
            True,
            ("openai", "anthropic", "google"),
            True,
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
            False,
        ),
        (
            ["tests/e2e/test_tools_auto_loop.py"],
            False,
            False,
            ("openai", "anthropic", "google"),
            True,
            True,
            True,
        ),
        (
            ["packages/organizations/qwen/tests/e2e/test_live_contract.py"],
            False,
            False,
            (),
            False,
            False,
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
    qwen_e2e,
):
    selection = _SELECTOR.select_e2e_lanes(changed_paths)

    assert selection.core is core
    assert selection.shared_core is shared_core
    assert selection.core_organizations == organizations
    assert selection.mistral_e2e is mistral_e2e
    assert selection.xai_e2e is xai_e2e
    assert selection.qwen_e2e is qwen_e2e
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
    assert outputs["kimi"] == "false"
    assert outputs["kimi_e2e"] == "false"
    assert outputs["mistral_e2e"] == "false"
    assert outputs["xai_e2e"] == "false"
    assert outputs["qwen"] == "false"
    assert outputs["qwen_e2e"] == "false"


@pytest.mark.unit
def test_core_e2e_jobs_run_after_a_skipped_core_publish_for_harness_changes():
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")

    for organization in ("openai", "anthropic", "google"):
        job = workflow.split(f"  post-publish-{organization}-e2e:\n", maxsplit=1)[1]
        job = job.split("\n  post-publish-", maxsplit=1)[0]

        assert f"needs.changes.outputs.core_{organization}_e2e == 'true'" in job
        assert "needs.publish-core-testpypi.result == 'success'" in job
        assert "needs.publish-core-testpypi.result == 'skipped'" in job
        assert "needs.changes.outputs.core == 'true'" not in job


@pytest.mark.unit
def test_qwen_package_changes_select_candidate_and_e2e_lane():
    selection = _SELECTOR.select_e2e_lanes(
        ["packages/organizations/qwen/src/llm_api_adapter_qwen/adapter.py"]
    )

    assert selection.qwen is True
    assert selection.qwen_e2e is True


@pytest.mark.unit
def test_deepseek_package_changes_select_only_its_candidate_and_e2e_lane():
    selection = _SELECTOR.select_e2e_lanes(
        ["packages/organizations/deepseek/src/llm_api_adapter_deepseek/adapter.py"]
    )

    assert selection.core is False
    assert selection.deepseek is True
    assert selection.deepseek_e2e is True
    assert selection.kimi_e2e is False
    assert selection.mistral_e2e is False
    assert selection.xai_e2e is False
    assert selection.qwen_e2e is False


@pytest.mark.unit
def test_deepseek_package_e2e_changes_select_only_its_live_lane():
    selection = _SELECTOR.select_e2e_lanes(
        ["packages/organizations/deepseek/tests/e2e/test_live_contract.py"]
    )

    assert selection.core is False
    assert selection.deepseek is False
    assert selection.deepseek_e2e is True
    assert selection.kimi_e2e is False
    assert selection.mistral_e2e is False
    assert selection.xai_e2e is False
    assert selection.qwen_e2e is False


@pytest.mark.unit
def test_kimi_package_changes_select_only_its_candidate_and_e2e_lane():
    selection = _SELECTOR.select_e2e_lanes(
        ["packages/organizations/kimi/src/llm_api_adapter_kimi/adapter.py"]
    )

    assert selection.kimi is True
    assert selection.kimi_e2e is True
    assert selection.mistral_e2e is False
    assert selection.xai_e2e is False
    assert selection.qwen_e2e is False


@pytest.mark.unit
def test_kimi_package_e2e_changes_select_only_the_kimi_live_lane():
    selection = _SELECTOR.select_e2e_lanes(
        ["packages/organizations/kimi/tests/e2e/test_live_contract.py"]
    )

    assert selection.kimi is False
    assert selection.kimi_e2e is True
    assert selection.mistral_e2e is False
    assert selection.xai_e2e is False
    assert selection.qwen_e2e is False


@pytest.mark.unit
def test_qwen_candidate_e2e_job_uses_only_qwen_credentials_and_candidates():
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    job = workflow.split("  post-publish-qwen-e2e:\n", maxsplit=1)[1]

    assert "needs.changes.outputs.qwen_e2e == 'true'" in job
    assert "needs.publish-core-testpypi.result == 'success'" in job
    assert "needs.publish-core-testpypi.result == 'skipped'" in job
    assert "needs.publish-qwen-testpypi.result == 'success'" in job
    assert "needs.publish-qwen-testpypi.result == 'skipped'" in job
    assert "QWEN_API_KEY: ${{ secrets.QWEN_API_KEY }}" in job
    assert "QWEN_WORKSPACE_ID: ${{ secrets.QWEN_WORKSPACE_ID }}" in job
    assert "llm-api-adapter[async,httpx]==${CORE_CANDIDATE_VERSION}" in job
    assert "llm-api-adapter-qwen[async,httpx]==${QWEN_CANDIDATE_VERSION}" in job
    assert "organization='qwen'" in job
    assert "model='qwen3.8-max'" in job
    assert "pytest -v --import-mode=importlib -m e2e_qwen" in job
    assert "MISTRAL_API_KEY" not in job
    assert "XAI_API_KEY" not in job


@pytest.mark.unit
def test_kimi_candidate_e2e_job_uses_only_kimi_credentials_and_candidates():
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    job = workflow.split("  post-publish-kimi-e2e:\n", maxsplit=1)[1]
    job = job.split("\n  post-publish-", maxsplit=1)[0]

    assert "needs.changes.outputs.kimi_e2e == 'true'" in job
    assert "needs.publish-core-testpypi.result == 'success'" in job
    assert "needs.publish-core-testpypi.result == 'skipped'" in job
    assert "needs.publish-kimi-testpypi.result == 'success'" in job
    assert "needs.publish-kimi-testpypi.result == 'skipped'" in job
    assert "KIMI_API_KEY: ${{ secrets.KIMI_API_KEY }}" in job
    assert "llm-api-adapter[async,httpx]==${CORE_CANDIDATE_VERSION}" in job
    assert "llm-api-adapter-kimi[async,httpx]==${KIMI_CANDIDATE_VERSION}" in job
    assert "organization='kimi'" in job
    assert "model='kimi-k3'" in job
    assert "pytest -v --import-mode=importlib -m e2e_kimi" in job
    assert "MISTRAL_API_KEY" not in job
    assert "XAI_API_KEY" not in job
    assert "QWEN_API_KEY" not in job


@pytest.mark.unit
def test_deepseek_candidate_e2e_job_uses_only_deepseek_credentials_and_candidates():
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    job = workflow.split("  post-publish-deepseek-e2e:\n", maxsplit=1)[1]

    assert "needs.changes.outputs.deepseek_e2e == 'true'" in job
    assert "needs.publish-core-testpypi.result == 'success'" in job
    assert "needs.publish-core-testpypi.result == 'skipped'" in job
    assert "needs.publish-deepseek-testpypi.result == 'success'" in job
    assert "needs.publish-deepseek-testpypi.result == 'skipped'" in job
    assert "DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}" in job
    assert "llm-api-adapter[async,httpx]==${CORE_CANDIDATE_VERSION}" in job
    assert (
        "llm-api-adapter-deepseek[async,httpx]==${DEEPSEEK_CANDIDATE_VERSION}"
        in job
    )
    assert "organization='deepseek'" in job
    assert "model='deepseek-flash'" in job
    assert "pytest -v --import-mode=importlib -m e2e_deepseek" in job
    assert (
        "--rootdir=. tests/e2e packages/organizations/deepseek/tests/e2e"
        in job
    )
    assert "KIMI_API_KEY" not in job
    assert "QWEN_API_KEY" not in job


@pytest.mark.unit
def test_post_publish_e2e_jobs_use_importlib_collection_mode():
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")

    for marker in (
        "e2e_openai",
        "e2e_anthropic",
        "e2e_google",
        "e2e_kimi",
        "e2e_mistral",
        "e2e_xai",
        "e2e_qwen",
        "e2e_deepseek",
    ):
        assert f"pytest -v --import-mode=importlib -m {marker}" in workflow


@pytest.mark.unit
def test_qwen_deterministic_and_tag_workflows_cover_unit_and_integration_tests():
    workflow_dir = _REPOSITORY_ROOT / ".github" / "workflows"

    for filename in ("ci-qwen-dev.yml", "ci-qwen-main.yml"):
        workflow = (workflow_dir / filename).read_text(encoding="utf-8")
        assert "packages/organizations/qwen/**" in workflow
        assert (
            "pytest -v --ignore=packages/organizations/qwen/tests/e2e "
            "-m unit packages/organizations/qwen/tests"
        ) in workflow
        assert (
            "pytest -v --ignore=packages/organizations/qwen/tests/e2e "
            "-m integration packages/organizations/qwen/tests"
        ) in workflow
        assert "e2e_qwen" not in workflow
        assert "secrets." not in workflow

    main_workflow = (workflow_dir / "ci-main.yml").read_text(encoding="utf-8")
    assert '"qwen-v*"' in main_workflow
    assert "test-qwen:" in main_workflow
    assert "publish-qwen-pypi:" in main_workflow


@pytest.mark.unit
def test_kimi_deterministic_and_tag_workflows_cover_unit_and_integration_tests():
    workflow_dir = _REPOSITORY_ROOT / ".github" / "workflows"

    for filename in ("ci-kimi-dev.yml", "ci-kimi-main.yml"):
        workflow = (workflow_dir / filename).read_text(encoding="utf-8")
        assert "packages/organizations/kimi/**" in workflow
        assert (
            "pytest -v --ignore=packages/organizations/kimi/tests/e2e "
            "-m unit packages/organizations/kimi/tests"
        ) in workflow
        assert (
            "pytest -v --ignore=packages/organizations/kimi/tests/e2e "
            "-m integration packages/organizations/kimi/tests"
        ) in workflow
        assert "e2e_kimi" not in workflow
        assert "secrets." not in workflow

    main_workflow = (workflow_dir / "ci-main.yml").read_text(encoding="utf-8")
    assert '"kimi-v*"' in main_workflow
    assert "test-kimi:" in main_workflow
    assert "publish-kimi-pypi:" in main_workflow


@pytest.mark.unit
def test_deepseek_deterministic_workflows_cover_unit_and_integration_tests():
    workflow_dir = _REPOSITORY_ROOT / ".github" / "workflows"

    for filename in ("ci-deepseek-dev.yml", "ci-deepseek-main.yml"):
        workflow = (workflow_dir / filename).read_text(encoding="utf-8")
        assert "packages/organizations/deepseek/**" in workflow
        assert (
            "pytest -v --ignore=packages/organizations/deepseek/tests/e2e "
            "-m unit packages/organizations/deepseek/tests"
        ) in workflow
        assert (
            "pytest -v --ignore=packages/organizations/deepseek/tests/e2e "
            "-m integration packages/organizations/deepseek/tests"
        ) in workflow
        assert "e2e_deepseek" not in workflow
        assert "secrets." not in workflow

    dev_workflow = (workflow_dir / "ci-dev.yml").read_text(encoding="utf-8")
    assert ".github/workflows/ci-deepseek-dev.yml" in dev_workflow
    assert ".github/workflows/ci-deepseek-main.yml" in dev_workflow

    main_workflow = (workflow_dir / "ci-main.yml").read_text(encoding="utf-8")
    assert '"deepseek-v*"' in main_workflow
    assert "test-deepseek:" in main_workflow
    assert "publish-deepseek-pypi:" in main_workflow
