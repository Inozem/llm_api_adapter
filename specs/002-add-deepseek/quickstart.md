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

## 3. Maintainer-controlled E2E before promotion to `dev`

Do not run live calls in a pull request or local deterministic suite. After the
deterministic checks and wheel installation checks pass, but **before** the
reviewed candidate is promoted through a staging pull request to protected
`dev`, perform this final handoff. The E2E test reads `DEEPSEEK_API_KEY` directly
from its environment. Do not put a key in a command, test argument, log, or
result. The preflight verifies only that the key is present and prints the
command; it does **not** run the paid test:

```powershell
if ([string]::IsNullOrWhiteSpace($env:DEEPSEEK_API_KEY)) {
    throw "DEEPSEEK_API_KEY is not configured in this environment."
}

$e2eCommand = 'python -m pytest -v --import-mode=importlib -m e2e_deepseek --rootdir=. tests/e2e packages/organizations/deepseek/tests/e2e'
Write-Output 'DEEPSEEK_API_KEY is configured. Run this command yourself:'
Write-Output $e2eCommand
```

The maintainer runs the displayed command and records only a sanitized result. A
passing result is required before promoting the exact candidate commit to `dev`.
The full applicable DeepSeek E2E suite exercises the shared facade and package
contracts for `deepseek-flash`. Document-input scenarios are excluded by the
declared capability gate.

## 4. Post-publish TestPyPI gate

Once the staging pull request is merged into protected `dev`,
`ci-dev-release.yml` automatically publishes only the changed Core/organization
distributions to TestPyPI. When DeepSeek or shared Core changes, its dedicated
post-publish job installs the exact candidate versions, verifies plugin discovery,
and runs the same full applicable DeepSeek E2E lane with `DEEPSEEK_API_KEY` supplied only
through GitHub Actions Secrets. This independent second gate verifies the
published artifacts; it does not replace the pre-promotion check.

Before final PyPI tags, a maintainer performs the documented public TestPyPI
installation and verifies the public install path, text/tool/structured response,
image boundary, reasoning continuation, and absence of credential or reasoning
leakage.
