# Quickstart: Validate the DeepSeek Release Candidate

This guide validates the planned Core `0.9.6` and independently versioned
`llm-api-adapter-deepseek` `0.1.0` integration. It intentionally separates safe
deterministic checks from paid live checks.

## 1. Deterministic development environment

From the repository root, install the Core and package in editable mode with both optional transports:

```powershell
python -m pip install -r tests/requirements-test.txt
python -m pip install -e ".[async,httpx]"
python -m pip install -e "packages/organizations/deepseek[async,httpx]"
```

These commands are credential-free. Do not set `DEEPSEEK_API_KEY` for the
deterministic suite; the mocked transports and discovery checks must never make
provider requests or incur charges.

Run the Core discovery/extra and CI-selection checks, then the package's credential-free unit and mocked-integration suites:

```powershell
python -m pytest -v -m unit tests/unit/test_organization_plugins.py tests/unit/test_ci_e2e_lane_selection.py
python -m pytest -v --ignore=packages/organizations/deepseek/tests/e2e -m unit packages/organizations/deepseek/tests
python -m pytest -v --ignore=packages/organizations/deepseek/tests/e2e -m integration packages/organizations/deepseek/tests
```

Expected outcomes:

- `organization="deepseek"` is actionable-but-not-installed before the package is available and discoverable after entry-point installation.
- `deepseek-flash` passes declared sync, async, streaming, tools, structured output, reasoning-continuation, image, usage/cost, and normalized-error fixtures with both supported synchronous transports.
- PDF/document inputs and every unsupported capability fail before any mock transport sees a request.
- The DeepSeek selector change selects exactly its candidate and live-E2E lane; it does not accidentally use another provider's credentials.

## 2. Build and isolated installation

Build the two changed distributions:

```powershell
python -m pip install build
python -m build
python -m build packages/organizations/deepseek
```

Install the generated wheels into a clean virtual environment with the matching
Core `deepseek`, `async`, and `httpx` extras. Verify that the package discovers
through `UniversalLLMAPIAdapter` without importing package modules from the
repository checkout. Repeat the missing-package case with only the Core wheel;
the error must recommend the exact DeepSeek distribution. Core and the
DeepSeek package are released independently, so keep the candidate pair exact
(`0.9.6` with `0.1.0`) and never substitute an older TestPyPI artifact.

## 3. Release-candidate E2E (protected `dev` gate and manual handoff)

Do not run live calls in a pull request or local deterministic suite. Once a
maintainer merges the reviewed candidate into protected `dev`,
`ci-dev-release.yml` automatically publishes only the changed Core/organization
distributions to TestPyPI. When DeepSeek or shared Core changes, its dedicated
post-publish job installs the exact candidate versions, verifies plugin discovery,
and runs the bounded DeepSeek lane with `DEEPSEEK_API_KEY` supplied only through
GitHub Actions Secrets.

The E2E test reads `DEEPSEEK_API_KEY` directly from its environment. Do not put a
key in a command, test argument, log, or result. If a maintainer requests an
additional final manual check after the TestPyPI candidates exist, perform this
preflight at the final handoff. It verifies only that the key is present and
prints the command; it does **not** run the paid test:

```powershell
if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
    throw "DEEPSEEK_API_KEY is not configured in this environment."
}

$e2eCommand = 'python -m pytest -v --import-mode=importlib -m e2e_deepseek packages/organizations/deepseek/tests/e2e/test_live_contract.py'
Write-Output 'DEEPSEEK_API_KEY is configured. Run this command yourself:'
Write-Output $e2eCommand
```

The maintainer decides whether to run the displayed command; this optional manual
check supplements rather than replaces the automatic `dev` post-publish gate.
The bounded live profile exercises only `deepseek-flash` and only declared
features. It excludes document input. Before final PyPI tags, a maintainer
performs the same explicit candidate installation and verifies the documented
public install path, text/tool/structured response, image boundary, reasoning
continuation, and absence of credential or reasoning leakage.
