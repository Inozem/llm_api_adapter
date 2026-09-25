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

Expected: every first-party registered model has an explicit `capability_exceptions` list, including an empty list when it follows the baseline. Each unlisted capability selects its baseline-positive scenario; each declared exception selects its documented check. A missing profile, invalid exception, or missing evidence fails deterministically. A legacy third-party plugin can still register without a profile, but profile-based certification fails clearly. An exception check replaces only the positive scenario for that same capability.

## 2. Validate cached usage and prices

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -m unit tests/unit/models/responses tests/unit/streaming/test_chunk_buffer.py tests/unit/adapters/test_base_adapter.py tests/unit/adapters/test_pricing_lifecycle.py tests/unit/adapters/test_openai_adapter.py tests/unit/adapters/test_anthropic_adapter.py tests/unit/adapters/test_google_adapter.py
& .\.venv\Scripts\python.exe -m pytest -q --ignore=packages/organizations/zai/tests/e2e --ignore=packages/organizations/kimi/tests/e2e --ignore=packages/organizations/deepseek/tests/e2e -m "unit or integration" packages/organizations/zai/tests packages/organizations/kimi/tests packages/organizations/deepseek/tests
& .\.venv\Scripts\python.exe -m pytest -q -m "unit or integration" packages/organizations/mistral/tests/test_mistral_adapter.py packages/organizations/xai/tests/test_xai_adapter.py packages/organizations/qwen/tests/test_qwen_adapter.py
```

Expected: complete provider-confirmed cache splits price ordinary and cached input once at the correct tier; missing or inconsistent splits yield no invented input/total cost. Direct `ChatResponse.from_*` calls and provider adapters expose `None` for omitted counts in parsed partial usage and retain reported `0`; direct `Usage` construction retains its zero defaults. DeepSeek peak/off-peak examples retain their dispatch-time behavior, and its separate streaming usage parser preserves partial counts. Synchronous, asynchronous, and streaming final responses agree on accounting. No live request is made.

## 3. Validate organization metadata and CI line selection

```powershell
& .\.venv\Scripts\python.exe -m pytest -q -m unit tests/unit/test_external_organization_metadata.py tests/unit/test_ci_e2e_lane_selection.py tests/unit/test_organization_plugins.py
```

Expected: all known external organizations match their Core extras, package manifests and entry points, E2E profiles, and CI selector. Mutation fixtures that remove a package or change a distribution name fail with the affected source identified. Existing package-specific outputs and secret-bearing workflow jobs remain separate.

## 4. Check E2E collection without running it

```powershell
& .\.venv\Scripts\python.exe -m pytest --collect-only -q --import-mode=importlib -m e2e_zai tests/e2e
```

Expected: baseline-positive scenarios are collected for each exact model unless its exception list routes that capability to a documented exception scenario. Missing profiles, invalid exceptions, and missing evidence fail collection instead of silently deselecting tests. Collection must not require `ZAI_API_KEY` or make a provider call. Repeat for the other provider markers in `pytest.ini` as part of release preparation.

## 5. Run the full deterministic gate

```powershell
& .\.venv\Scripts\python.exe -m pytest -v --import-mode=importlib -m "unit or integration"
```

Expected: Core and affected package deterministic suites pass with no network or provider credentials. The Python 3.10–3.14 CI matrix remains authoritative for cross-version compatibility; its canonical Python 3.10 lane keeps the 90% coverage floor. Inspect the final diff and matching README/contributor/package pricing documentation before review.

## Later release gate

After a staging pull request merges to `dev`, the established post-publish workflow builds and installs exact TestPyPI candidates, then runs each affected provider's applicable common and package-local E2E scenarios in its own job with only its own credential. A pre-merge or local live run is a preflight and does not replace that gate. No E2E or publication is performed by `$speckit-plan`.
