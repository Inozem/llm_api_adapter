---

description: "Dependency-ordered implementation tasks for the Z.ai / GLM provider release"
---

# Tasks: Z.ai / GLM Provider Release

**Input**: `specs/003-zai-glm-provider/{spec,plan,research,data-model,quickstart}.md` and `contracts/zai-provider.md`

**Tests**: Required: deterministic credential-free suites, applicable shared Core E2E,
package-local boundary E2E, and focused authorized live verification.

## Phase 1: Setup

**Purpose**: Create the independently versioned provider-package skeleton.

- [X] T001 Create package metadata in `packages/organizations/zai/pyproject.toml` with version `0.1.0`, Core range `>=0.9.7,<1.0.0`, forwarded `async`/`httpx` extras, and the `zai` entry point.
- [X] T002 Create distributable package files in `packages/organizations/zai/MANIFEST.in`, `packages/organizations/zai/LICENSE`, `packages/organizations/zai/README.md`, `packages/organizations/zai/src/llm_api_adapter_zai/__init__.py`, `packages/organizations/zai/src/llm_api_adapter_zai/py.typed`, and `packages/organizations/zai/src/llm_api_adapter_zai/clients/__init__.py`.

---

## Phase 2: Foundational

**Purpose**: Complete the Core discovery and E2E-profile prerequisites that block every story.

- [X] T003 Update Core version, Z.ai keyword, and optional `zai` extra in `pyproject.toml` without adding `llm-api-adapter-zai` to base dependencies.
- [X] T004 Add the `zai` known-package record and `llm-api-adapter-zai` installation remedy in `src/llm_api_adapter/organization_registry.py`.
- [X] T005 Add Core regression coverage for Z.ai's absent-package error, lazy discovery, and optional extra in `tests/unit/test_organization_plugins.py`.
- [X] T006 Mark shared portable JSON-schema scenarios with `e2e_feature("structured_output")` in `tests/e2e/test_json_schema.py` and `tests/e2e/test_async.py`.
- [X] T007 Add the `e2e_zai` marker, `ZAI_API_KEY`, and a named explicit Z.ai capability profile in `pytest.ini` and `tests/e2e/conftest.py`.

**Checkpoint**: Core distinguishes absent Z.ai from an unknown organization, and shared E2E can
select only the profile's declared capability set.

---

## Phase 3: User Story 1 — Use a verified GLM model (Priority: P1) 🎯 MVP

**Goal**: A developer installs the provider, selects `organization="zai"` and
`glm-5.3-flash`, and receives standard normalized output through the existing facade.

**Independent Test**: With Core `0.9.7` and only the Z.ai package installed, plugin discovery
registers Flash and mocked sync/async/stream responses normalize through the existing facade.

- [X] T008 [P] [US1] Write package-layout, dependency-range, forwarded-extra, and entry-point tests in `packages/organizations/zai/tests/test_package_scaffold.py`.
- [X] T009 [P] [US1] Write closed model-metadata tests in `packages/organizations/zai/tests/fixtures/zai_capability_discovery.py` and `packages/organizations/zai/tests/test_capability_discovery.py` for exact Flash ID, 1,000,000/131,072 limits, USD rates, and `low`/`high`/`max` reasoning.
- [X] T010 [P] [US1] Write mocked facade, sync, async, SSE, HTTPX-parity, error, and usage tests in `packages/organizations/zai/tests/test_zai_adapter.py`.
- [X] T011 [US1] Implement Flash registry metadata and standard pricing in `packages/organizations/zai/src/llm_api_adapter_zai/registry/organizations/zai.json` and `packages/organizations/zai/src/llm_api_adapter_zai/registry/__init__.py`.
- [X] T012 [US1] Implement the versioned `OrganizationPlugin`, model metadata export, and `zai` factory registration in `packages/organizations/zai/src/llm_api_adapter_zai/plugin.py`.
- [X] T013 [US1] Implement official endpoint construction, bearer auth, requests dispatch, response parsing, and normalized errors in `packages/organizations/zai/src/llm_api_adapter_zai/clients/sync_client.py`.
- [X] T014 [US1] Implement equivalent async dispatch, parsing, cancellation, and cleanup in `packages/organizations/zai/src/llm_api_adapter_zai/clients/async_client.py`.
- [X] T015 [US1] Implement SSE delta parsing, tool-delta accumulation, visible-text isolation, and final response assembly in `packages/organizations/zai/src/llm_api_adapter_zai/streaming.py`.
- [X] T016 [US1] Implement the facade-compatible Z.ai adapter using existing model registry and transport contracts in `packages/organizations/zai/src/llm_api_adapter_zai/adapter.py`.
- [X] T017 [US1] Run deterministic package and Core discovery suites from `specs/003-zai-glm-provider/quickstart.md` and resolve failures in `packages/organizations/zai/tests/` and `tests/unit/test_organization_plugins.py`.

**Checkpoint**: The minimal Z.ai package is independently installable and returns normalized Flash
chat, async, and streaming results without another provider package.

---

## Phase 4: User Story 2 — Reliably use declared capabilities (Priority: P2)

**Goal**: Developers receive a precise, tested Flash capability boundary, with unsupported requests
rejected locally before an outbound provider call.

**Independent Test**: Mocked transports prove declared supported behavior and pre-transport
rejection of unlisted models, schemas, tool modes, reasoning levels, and file forms.

- [X] T018 [P] [US2] Add pre-transport rejection tests for schemas, response models, unsupported `tool_choice`, invalid reasoning levels, unknown models, documents, and unverified combinations in `packages/organizations/zai/tests/test_zai_adapter.py`.
- [X] T019 [P] [US2] Add payload and normalization tests for auto-only tools with at most 128 declarations, tool-result history, separate reasoning events, and image URL/data-URL parts in `packages/organizations/zai/tests/test_zai_adapter.py`.
- [X] T020 [P] [US2] Add cache-hit/cache-miss pricing and malformed/incomplete-usage tests in `packages/organizations/zai/tests/test_zai_adapter.py`.
- [X] T021 [US2] Implement validated cached-input pricing without changing the Core registry schema in `packages/organizations/zai/src/llm_api_adapter_zai/registry/cache_pricing.py`.
- [X] T022 [US2] Implement exact-model validation, tool/reasoning rules, structured-output and document rejection, image serialization, and usage/cost handling in `packages/organizations/zai/src/llm_api_adapter_zai/adapter.py`.
- [X] T023 [US2] Publish the complete capability matrix and exclusions in `packages/organizations/zai/README.md` and `packages/organizations/zai/tests/fixtures/zai_capability_discovery.py`.
- [X] T024 [US2] Add Z.ai provider-specific structured-output-rejection and document-gate scenarios in `packages/organizations/zai/tests/e2e/conftest.py` and `packages/organizations/zai/tests/e2e/test_capability_boundaries.py`; shared Core E2E tests cover the supported baseline capabilities.
- [X] T025 [US2] Run the `e2e_zai` shared and package-local collection from `specs/003-zai-glm-provider/quickstart.md` with maintainer-authorized `ZAI_API_KEY`; keep documents rejected unless both direct forms pass.

**Checkpoint**: The matrix, adapter preflight, deterministic tests, and E2E profile agree, and no
unsupported request reaches the provider transport.

---

## Phase 5: User Story 3 — Upgrade and release independently (Priority: P3)

**Goal**: Maintainers can independently build, publish, install, and validate Core `0.9.7` and
provider `0.1.0` as a coordinated release train.

**Independent Test**: A clean TestPyPI environment resolves the Z.ai extra, discovers the plugin,
and runs `e2e_zai` against exact candidate artifacts with only `ZAI_API_KEY`.

- [X] T026 [P] [US3] Add Z.ai candidate and E2E-lane fields, outputs, and path selection in `.github/scripts/select_e2e_lanes.py`.
- [X] T027 [P] [US3] Add deterministic Z.ai lane-selection and credential-isolation tests in `tests/unit/test_ci_e2e_lane_selection.py`.
- [X] T028 [P] [US3] Add credential-free Python 3.10–3.14 package validation in `.github/workflows/ci-zai-dev.yml`.
- [X] T029 [P] [US3] Add credential-free main-branch package validation in `.github/workflows/ci-zai-main.yml`.
- [X] T030 [US3] Wire Z.ai paths, `zai-v*` tags, test gate, and independent PyPI publishing in `.github/workflows/ci-dev.yml` and `.github/workflows/ci-main.yml`.
- [X] T031 [US3] Add Z.ai version extraction, TestPyPI publishing, exact-candidate installation, plugin discovery, and `ZAI_API_KEY`-only E2E to `.github/workflows/ci-dev-release.yml`.
- [X] T032 [US3] Document the Core extra, Flash matrix, exclusions, pricing, and Z.ai release commands in `README.md`, `CONTRIBUTING.md`, and `packages/organizations/zai/README.md`.
- [X] T033 [US3] Build both distributions, install TestPyPI candidates cleanly, and run the full release command in `specs/003-zai-glm-provider/quickstart.md`.

**Checkpoint**: The package can publish as `zai-v0.1.0`; Core `v0.9.7` advertises its compatible
extra and the candidate lane proves the release artifact.

---

## Phase 6: Polish and cross-cutting verification

- [X] T034 [P] Review `packages/organizations/zai/`, `README.md`, and `CONTRIBUTING.md` for API keys, raw reasoning, and raw tool arguments in fixtures, docs, logs, and diagnostics.
- [X] T035 [P] Refresh Z.ai-relevant architecture artifacts in `docs/architecture.json`, `docs/organization_packages.json`, `docs/registry.json`, `docs/adapters.json`, `docs/messages.json`, `docs/errors.json`, and `docs/ci_cd.json` when their generator records the provider surface.
- [X] T036 Run all deterministic Core and Z.ai suites from `specs/003-zai-glm-provider/quickstart.md`, inspect the diff, and resolve unintended artifacts before review.
- [X] T037 After deterministic and authorized live E2E evidence fixes the released capability profile, update `specs/001-baseline-contract/spec.md` with Z.ai as an optional organization package and its exact declared capability boundary.
- [X] T038 Verify each requirement in `specs/003-zai-glm-provider/spec.md` has an implementation test, documentation entry, or explicit exclusion, then record requirement-quality findings in `specs/003-zai-glm-provider/checklists/requirements.md`.

---

## Dependencies and execution order

`T001–T007` block every story. US1 (`T008–T017`) produces the independently testable provider.
US2 (`T018–T025`) completes its truthful capability boundary. US3 (`T026–T033`) makes the stable
package independently buildable, publishable, and release-validated. Polish follows the desired
release scope.

## Parallel opportunities

- US1: `T008–T010` are independent test files after the foundation.
- US2: `T018–T020` investigate distinct behavior, but merge serially because they share one test module.
- US3: `T026–T029` touch distinct scripts, tests, or workflows.
- `T034` and `T035` are independent after the corresponding product surface settles.

## Implementation strategy

**MVP**: Complete setup, foundation, and US1; then prove plugin discovery and normalized Flash
text/sync/async/streaming with deterministic tests. Do not announce support yet.

**Incremental delivery**: Add US2 for a truthful, release-gated capability boundary. Then add US3
for independent artifacts, TestPyPI installation, and authorized provider E2E.

## Format validation

All 38 tasks use `- [ ]`, sequential IDs, exact paths, and story labels only in user-story phases.
`[P]` appears only for tasks with separable files or investigations.
