# Validation Quickstart: Core / Plugin Refactor 0.9.8

Run these checks **after implementation**, from the actual repository root (`llm_api_adapter/llm_api_adapter` in this workspace). This guide validates the feature; it does not authorize a live provider call or a release.

## Prerequisites

- Use a local development environment with Core, test requirements, and the six organization packages installed in editable mode, as in the project's deterministic CI jobs.
- Use no provider API keys for the commands below. The local `.venv\Scripts\python.exe` in this workspace is available; use the equivalent interpreter in another environment.
- Review [contract](./contracts/core-plugin-refactor.md) and [data model](./data-model.md) for expected behavior.

## 1. Validate registry profiles and shared selection

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -m unit tests/unit/llm_registry tests/unit/conformance tests/unit/test_organization_plugins.py
```

Expected: every first-party registered model has an explicit `capability_exceptions` list, including an empty list when it follows the baseline, and every declared exception has a valid `behavior_id`. An unlisted capability and a `pass` exception both select the baseline-positive scenario; another `(capability_id, behavior_id)` pair selects its documented deviation check. Mistral PDF via OCR remains a visible `pass` exception, while package tests verify the normalization and OCR costs. A missing profile, invalid behavior ID, or missing/duplicate route fails deterministically. A legacy third-party plugin can still register without a profile, but profile-based certification fails clearly.

## 2. Validate automatic cache usage and prices

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -m unit tests/unit/models/responses tests/unit/streaming/test_chunk_buffer.py tests/unit/adapters/test_base_adapter.py tests/unit/adapters/test_pricing_lifecycle.py tests/unit/adapters/test_openai_adapter.py tests/unit/adapters/test_anthropic_adapter.py tests/unit/adapters/test_google_adapter.py
& .\.venv\Scripts\python.exe -m pytest -q --import-mode=importlib --ignore=packages/organizations/zai/tests/e2e --ignore=packages/organizations/kimi/tests/e2e --ignore=packages/organizations/deepseek/tests/e2e -m "unit or integration" packages/organizations/zai/tests packages/organizations/kimi/tests packages/organizations/deepseek/tests
& .\.venv\Scripts\python.exe -m pytest -q -m "unit or integration" packages/organizations/mistral/tests/test_mistral_adapter.py packages/organizations/xai/tests/test_xai_adapter.py packages/organizations/qwen/tests/test_qwen_adapter.py
```

Expected: complete provider-confirmed automatic cache-read/cache-write splits price ordinary, read, and write input once at the correct tier; missing or inconsistent component splits yield no invented input/total cost. Registry fixtures contain rates only for exact models whose ordinary adapter requests can automatically incur and report those components; opt-in-only cache modes remain absent. Direct `ChatResponse.from_*` calls and provider adapters expose `None` for omitted counts in parsed partial usage and retain reported `0`; direct `Usage` construction retains its zero defaults and existing `cached_tokens` cache-read meaning. DeepSeek peak/off-peak examples retain their dispatch-time behavior, and its separate streaming usage parser preserves partial counts. Synchronous, asynchronous, and streaming final responses agree on accounting. No live request is made.

## 3. Validate organization metadata and CI line selection

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -m unit tests/unit/test_external_organization_metadata.py tests/unit/test_ci_e2e_lane_selection.py tests/unit/test_organization_plugins.py
```

Expected: all known external organizations match their Core extras, package manifests and entry points, E2E profiles, and CI selector. Mutation fixtures that remove a package or change a distribution name fail with the affected source identified. Existing package-specific outputs and secret-bearing workflow jobs remain separate.

## 4. Check E2E collection without running it

```powershell
& .\.venv\Scripts\python.exe -m pytest --collect-only -q --import-mode=importlib -m e2e_zai tests/e2e
```

Expected: baseline-positive scenarios are collected for each exact model with no exception or a `pass` exception. Other exception pairs route to documented deviation scenarios. Missing profiles, invalid behavior IDs, and missing/duplicate routes fail collection instead of silently deselecting tests. Collection must not require `ZAI_API_KEY` or make a provider call. Repeat for the other provider markers in `pytest.ini` as part of release preparation.

## 5. Run the full deterministic gate

```powershell
& .\.venv\Scripts\python.exe -m pytest -v --import-mode=importlib -m "unit or integration"
```

Expected: Core and affected package deterministic suites pass with no network or provider credentials. The Python 3.10–3.14 CI matrix remains authoritative for cross-version compatibility; its canonical Python 3.10 lane keeps the 90% coverage floor. Inspect the final diff and matching README/contributor/package pricing documentation before review.

## T057 validation evidence (2026-10-04)

Executed from the repository root with Python 3.14.3. Every pytest subprocess had `PYTHON_DOTENV_DISABLED=1`; environment variables ending in `_API_KEY` or `_WORKSPACE_ID`, and variables named like tokens, secrets, or credentials, were removed. No provider calls were made.

Focused checks passed:

- Section 1 registry, conformance, and organization plugin command: **436 passed**.
- Section 2 first command, response models and adapter/pricing/streaming checks: **326 passed**.
- Section 2 second command, Z.ai/Kimi/DeepSeek package tests with package E2E ignored: **198 passed, 6 deselected**.
- Section 2 third command, Mistral/xAI/Qwen adapter tests: **229 passed, 3 deselected**.
- Section 3 metadata, CI lane selection, and organization plugin command: **109 passed**.
- Affected integration files (`tests/integration/test_llm_adapter_chat.py` and `tests/integration/test_streaming.py`) with `--import-mode=importlib -m integration`: **35 passed**.

Provider E2E collection used `python -m pytest --collect-only -q --import-mode=importlib --rootdir=. -m e2e_<provider> tests/e2e` for each marker below. When present, `packages/organizations/<provider>/tests/e2e` was included in that provider's collection. These counts include shared and existing package-local scenarios; the collected cases include real model parameters.

| Provider marker | Collected |
| --- | ---: |
| `e2e_openai` | 204 |
| `e2e_anthropic` | 124 |
| `e2e_google` | 114 |
| `e2e_mistral` | 34 |
| `e2e_xai` | 34 |
| `e2e_kimi` | 24 |
| `e2e_qwen` | 42 |
| `e2e_deepseek` | 15 |
| `e2e_zai` | 15 |

The full deterministic command, `python -m pytest -q --disable-warnings --import-mode=importlib -m "unit or integration"`, passed: **1,868 passed, 796 deselected, 21 warnings**. Full output is retained in the ignored `.pytest_cache/t057/full-suite-final.log` file.

CI configuration in `.github/workflows/ci-dev.yml` covers Python 3.10, 3.11, 3.12, 3.13, and 3.14; unit and integration coverage is accumulated, and `coverage report --show-missing --fail-under=90` is gated to Python 3.10. Only Python 3.14.3 was executed locally. Python 3.10-3.13 matrix results and the canonical Python 3.10 coverage result remain pending CI evidence; local validation does not claim those lanes.

## Later release gate

After a staging pull request merges to `dev`, the established post-publish workflow builds and installs exact TestPyPI candidates, then runs each affected provider's applicable common and package-local E2E scenarios in its own job with only its own credential. A pre-merge or local live run is a preflight and does not replace that gate. No E2E or publication is performed by `$speckit-plan`.

## T058 documentation audit and remaining gates (2026-10-04)

The constitution clarifies cache-read/write accounting, partial usage, pricing compatibility, and
authoritative provider totals. The architecture records exact-model profile certification,
test-only scenario selection, AST/TOML metadata validation, and usage/pricing ownership. T057's
local evidence above remains the recorded validation; remote CI and TestPyPI status were not
verified for this audit.

Primary review confirmed this documentation-only scope. Independent targeted validation passed:
**327 passed, 17 warnings** (`.pytest_cache/t058/primary-validation.log`).

Pending gates remain the Python 3.10-3.13 CI results and canonical Python 3.10 coverage result
noted above; merge the staging PR to `dev`; install the exact prepared TestPyPI candidates cleanly
and verify plugin discovery; then pass every applicable shared and package-local E2E scenario in
each affected provider's own lane with only that provider's key. Promote to `main` only after all
gates pass. Local and pre-merge live results are preflight only. No publication or provider calls
were performed.
