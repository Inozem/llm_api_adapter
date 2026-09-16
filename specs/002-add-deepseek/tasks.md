# Tasks: DeepSeek Provider Release

**Input**: Design documents from `/specs/002-add-deepseek/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`,
`contracts/deepseek-provider.md`, and `quickstart.md`

**Tests**: Required. FR-013 and every user story require deterministic evidence; the
post-publish DeepSeek E2E check is a separately authorized release check, not a
pull-request test.

**Organization**: Tasks are grouped by user story. `US1` is an independently
testable direct-package MVP; `US2` adds the Core optional-extra and missing-package
experience to that same package.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel with other marked tasks because it changes different files.
- **[US#]**: User story to which the task is traceable.
- Every task names its affected path(s).

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the independently versioned package and credential-free test layout.

- [X] T001 Create the `llm-api-adapter-deepseek` 0.1.0 distribution skeleton, Core `>=0.9.6,<1.0.0` dependency, forwarded `async`/`httpx` extras, and `deepseek` entry point in `packages/organizations/deepseek/pyproject.toml`, `packages/organizations/deepseek/MANIFEST.in`, `packages/organizations/deepseek/LICENSE`, `packages/organizations/deepseek/README.md`, `packages/organizations/deepseek/src/llm_api_adapter_deepseek/__init__.py`, `packages/organizations/deepseek/src/llm_api_adapter_deepseek/py.typed`, `packages/organizations/deepseek/src/llm_api_adapter_deepseek/clients/__init__.py`, and `packages/organizations/deepseek/src/llm_api_adapter_deepseek/registry/__init__.py`.
- [X] T002 [P] Create the E2E source-checkout test bootstrap and empty credential-free test layout in `packages/organizations/deepseek/tests/e2e/conftest.py`, `packages/organizations/deepseek/tests/test_package_scaffold.py`, `packages/organizations/deepseek/tests/test_capability_discovery.py`, and `packages/organizations/deepseek/tests/test_deepseek_adapter.py`.
- [X] T003 Register the `e2e_deepseek` marker alongside the existing organization markers in `pytest.ini` and `packages/organizations/deepseek/pyproject.toml`.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Make the coordinated Core 0.9.6 contract available before provider work begins.

**CRITICAL**: Complete this phase before starting a user-story phase.

- [X] T004 Add failing backward-compatibility and opaque-metadata cases for absent, populated, and provider-neutral `ChatResponse.provider_data`; a populated replay sentinel must not appear in `repr(response)`, `content`, or `reasoning_events` in `tests/unit/models/responses/test_chat_response.py`.
- [X] T005 Implement `provider_data` as an optional opaque mapping declared with `field(default=None, repr=False)`, absent for providers that do not use it and without changing existing response behavior, in `src/llm_api_adapter/models/responses/chat_response.py`.
- [X] T006 Set the Core distribution version to `0.9.6` in `pyproject.toml` so the locally developed 0.1.0 organization package can require its released Core contract.

**Checkpoint**: The package can depend on Core 0.9.6, and response metadata safely carries transport-only continuation state.

---

## Phase 3: User Story 1 - Use DeepSeek through the common facade (Priority: P1) MVP

**Goal**: A directly installed DeepSeek package registers `deepseek` and supplies normalized
sync, async, and streaming text chat through the unchanged public facade.

**Independent Test**: Install Core 0.9.6 plus the DeepSeek package directly in a clean environment,
use `UniversalLLMAPIAdapter(organization="deepseek", model="deepseek-flash", ...)` against
simulated Responses API results, and verify normalized sync, async, and stream outcomes.

### Tests for User Story 1

- [X] T007 [P] [US1] Write failing distribution and plugin-contract tests for the version range, transport extras, entry point, plugin API version, and `deepseek` registration in `packages/organizations/deepseek/tests/test_package_scaffold.py`.
- [X] T008 [P] [US1] Write failing mocked facade tests for normal text `chat()`, `achat()`, `stream_chat()`, and `astream_chat()` Responses behavior, callbacks, completion, cancellation, and malformed terminal events in `packages/organizations/deepseek/tests/test_deepseek_adapter.py`.

### Implementation for User Story 1

- [X] T009 [US1] Define the initial `deepseek-flash` model metadata and its first-party registration surface in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/registry/organizations/deepseek.json`, `packages/organizations/deepseek/src/llm_api_adapter_deepseek/registry/__init__.py`, and `packages/organizations/deepseek/src/llm_api_adapter_deepseek/plugin.py`.
- [X] T010 [US1] Implement the official Responses API synchronous client, authorization headers, core transport selection, request serialization, SSE event iteration, and normalized non-retry error boundary in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/clients/sync_client.py`.
- [X] T011 [US1] Implement DeepSeek message-to-Responses text request mapping and normalized synchronous `chat()` finalization in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/adapter.py`.
- [X] T012 [US1] Implement the official Responses API asynchronous client and normalized `achat()` path using the Core optional async transport in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/clients/async_client.py` and `packages/organizations/deepseek/src/llm_api_adapter_deepseek/adapter.py`.
- [X] T013 [US1] Implement the DeepSeek Responses SSE state machine and wire it to synchronous and asynchronous Core stream lifecycles without yielding incomplete or failed output as a final success in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/streaming.py` and `packages/organizations/deepseek/src/llm_api_adapter_deepseek/adapter.py`.
- [X] T014 [US1] Run and make green the direct-package MVP contract suite in `packages/organizations/deepseek/tests/test_package_scaffold.py` and `packages/organizations/deepseek/tests/test_deepseek_adapter.py` with `python -m pytest -v -m "unit or integration"` while excluding `packages/organizations/deepseek/tests/e2e`.

**Checkpoint**: Direct installation of the new package exposes DeepSeek through the existing facade for credential-free, mocked text, async, and streaming paths.

---

## Phase 4: User Story 2 - Install and diagnose DeepSeek support clearly (Priority: P1)

**Goal**: Core 0.9.6 advertises the optional add-on and distinguishes its absence from an unknown organization.

**Independent Test**: Test an environment with only Core and one with the DeepSeek package present;
the former supplies the exact install remedy and the latter discovers the plugin, while an arbitrary
organization remains unknown.

### Tests for User Story 2

- [X] T015 [US2] Add failing Core discovery cases for known-but-uninstalled `deepseek`, unknown organizations, and successful installed entry-point discovery in `tests/unit/test_organization_plugins.py`.

### Implementation for User Story 2

- [X] T016 [US2] Add the `deepseek` optional extra, DeepSeek keyword, and exact 0.1.x package range without adding it to base dependencies in `pyproject.toml`.
- [X] T017 [US2] Register `deepseek` as the known optional `llm-api-adapter-deepseek` package so existing lazy discovery emits the actionable missing-package error only for this known organization in `src/llm_api_adapter/organization_registry.py`.
- [X] T018 [US2] Document the Core optional-extra and direct-install routes, plus the known-versus-unknown selection result, in `README.md` and `packages/organizations/deepseek/README.md`.
- [X] T019 [US2] Verify the Core-only missing-package and isolated Core-plus-package discovery journeys described by `tests/unit/test_organization_plugins.py` and `specs/002-add-deepseek/quickstart.md`.

**Checkpoint**: The release has a truthful default installation surface and a precise remedy when its optional dependency is absent.

---

## Phase 5: User Story 3 - Select a capability-safe DeepSeek model (Priority: P2)

**Goal**: Only canonical `deepseek-flash` exposes officially verified feature combinations;
unsupported requests fail locally and never silently downgrade or change model.

**Independent Test**: Exercise each declared capability and a representative unsupported request
with mocked provider behavior; assert both synchronous transport choices and the live profile use
only the published matrix.

### Tests for User Story 3

- [X] T020 [P] [US3] Write the DeepSeek capability-contract fixture and exact-model discovery tests for `deepseek-flash`, aliases, limits, and thinking modes in `packages/organizations/deepseek/tests/test_capability_discovery.py`; discovery assertions use only the standard registry metadata, while supported and unsupported modes are exercised by adapter contracts.
- [X] T021 [P] [US3] Add failing mocked Responses contract cases for tools and tool choices, portable JSON Schema/Pydantic output, image URL/data input, reasoning capture and continuation replay that stays absent from visible response fields and `repr`, explicit parallel control rejection, and pre-HTTP capability failures in `packages/organizations/deepseek/tests/test_deepseek_adapter.py`.
- [X] T022 [P] [US3] Write the bounded live facade contract for only declared DeepSeek Flash capabilities in `packages/organizations/deepseek/tests/e2e/test_live_contract.py`.

### Implementation for User Story 3

- [X] T023 [US3] Keep DeepSeek Flash metadata aligned with the standard organization registry in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/registry/organizations/deepseek.json` and `packages/organizations/deepseek/src/llm_api_adapter_deepseek/registry/__init__.py`: only the canonical model's limits, pricing tiers, and reasoning values are registered; no provider-specific aliases, capability matrix, image restrictions, or unsupported-mode fields are added.
- [X] T024 [US3] Implement Responses input/output mapping for application function tools, normal tool results, tool choice, portable JSON Schema, response-model validation, user image URL/data parts, and adapter-local capability preflight that rejects provider-built-in tools or explicit parallel control before transport in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/adapter.py`.
- [ ] T025 [US3] Implement opt-in visible reasoning, opaque `deepseek.reasoning_replay` storage in `ChatResponse.provider_data`, and local validation/replay only from a matching DeepSeek `previous_response`, without server-side continuation IDs, rendering, debug representation, or logging in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/adapter.py`.
- [ ] T026 [US3] Extend the DeepSeek SSE parser for reasoning, response terminal states, fragmented function calls, usage, opaque continuation material, and failed/cancelled streams in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/streaming.py`.
- [ ] T027 [US3] Map documented DeepSeek 400/401/402/422/429/500/503 Responses failures to existing normalized errors, with no adapter retry loop, in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/clients/sync_client.py` and `packages/organizations/deepseek/src/llm_api_adapter_deepseek/clients/async_client.py`.
- [ ] T028 [US3] Run the model discovery, adapter capability-boundary, transport-parity, and mocked stream evidence in `packages/organizations/deepseek/tests/test_capability_discovery.py` and `packages/organizations/deepseek/tests/test_deepseek_adapter.py` for the documented contract only.

**Checkpoint**: Every published DeepSeek capability has deterministic evidence, and unsupported modes are rejected before a provider call.

---

## Phase 6: User Story 4 - Understand document and cost boundaries (Priority: P2)

**Goal**: Direct documents are rejected before transport, while valid provider usage receives only a
documented time-of-use standard-rate estimate.

**Independent Test**: Submit every unsupported document form and mocked usage responses with
complete, incomplete, and invalid token data; no unsupported request reaches a mocked transport
and unavailable pricing remains unavailable.

### Tests for User Story 4

- [ ] T029 [P] [US4] Add failing document preflight and usage/cost cases for `DocumentPart`, non-image files, missing or malformed usage, cached/reasoning token details, and peak/off-peak UTC dispatch times in `packages/organizations/deepseek/tests/test_deepseek_adapter.py`.
- [ ] T030 [P] [US4] Write the installed-distribution non-live file boundary test proving that document input cannot start a DeepSeek request in `packages/organizations/deepseek/tests/e2e/test_file_contract.py`.

### Implementation for User Story 4

- [ ] T031 [US4] Reject every `DocumentPart`, non-image file, OCR/upload/conversion route, unsupported image form, and unverified file capability before either client is invoked in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/adapter.py`.
- [ ] T032 [US4] Add package-local validated Flash peak/off-peak UTC rate selection and provider-reported usage normalization; missing, negative, non-integer, or inconsistent usage and every unverifiable rate must leave each relevant cost value unset in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/registry/cache_pricing.py` and `packages/organizations/deepseek/src/llm_api_adapter_deepseek/adapter.py`.
- [ ] T033 [US4] Publish the single-model compatibility matrix, image/file boundary, no-document behavior, continuation privacy, standard-rate-not-invoice wording, and official-source links in `packages/organizations/deepseek/README.md` and `README.md`.
- [ ] T034 [US4] Run and make green the document, usage, cost, and installed-file-boundary checks in `packages/organizations/deepseek/tests/test_deepseek_adapter.py` and `packages/organizations/deepseek/tests/e2e/test_file_contract.py` without setting a DeepSeek credential.

**Checkpoint**: The package makes neither a hidden document-processing promise nor a fabricated cost claim.

---

## Phase 7: Polish & Cross-Cutting Release Work

**Purpose**: Wire the provider into deterministic and post-publish release evidence, refresh
repository documentation, and verify the exact release artifacts.

- [ ] T035 [P] Add the DeepSeek profile with capability gates that skip unsupported generic scenarios rather than weakening them in `tests/e2e/conftest.py`.
- [ ] T036 [P] Add deterministic selection coverage asserting that only DeepSeek changes select its candidate and `e2e_deepseek` lane in `tests/unit/test_ci_e2e_lane_selection.py`.
- [ ] T037 Implement DeepSeek path, package, shared-Core, and E2E-lane detection in `.github/scripts/select_e2e_lanes.py`.
- [ ] T038 Create the credential-free pull-request DeepSeek package workflow in `.github/workflows/ci-deepseek-dev.yml`.
- [ ] T039 Create the DeepSeek main-branch package validation workflow in `.github/workflows/ci-deepseek-main.yml`.
- [ ] T040 Add DeepSeek change filters and workflow dispatch to `.github/workflows/ci-dev.yml` and `.github/workflows/ci-main.yml`, then add TestPyPI publication, exact candidate installation, `DEEPSEEK_API_KEY` validation, plugin discovery, and the bounded `e2e_deepseek` post-publish job to `.github/workflows/ci-dev-release.yml`.
- [ ] T041 Update the contributor and release guidance for independently versioned DeepSeek artifacts and the credential boundary in `CONTRIBUTING.md` and `specs/002-add-deepseek/quickstart.md`.
- [ ] T042 Update the living baseline contract with the DeepSeek optional package, its direct-document rejection boundary, and opaque continuation metadata that is never rendered or logged in `specs/001-baseline-contract/spec.md`.
- [ ] T043 Refresh and review the generated repository relationships after all source and documentation work with `graphify update .`, updating `docs/architecture.json`, `docs/organization_packages.json`, `docs/registry.json`, `docs/adapters.json`, `docs/messages.json`, `docs/errors.json`, and `docs/ci_cd.json` only when the generated output changes.
- [ ] T044 Run every credential-free command in `specs/002-add-deepseek/quickstart.md`, including Core discovery, package unit/integration evidence, and DeepSeek selector checks.
- [ ] T045 Build Core and DeepSeek wheels from `pyproject.toml` and `packages/organizations/deepseek/pyproject.toml`, then perform the clean-environment installation checks specified in `specs/002-add-deepseek/quickstart.md` without importing from the checkout.
- [ ] T046 After protected TestPyPI candidates exist and a maintainer explicitly authorizes paid verification, run the bounded `DEEPSEEK_API_KEY` release-candidate suite in `packages/organizations/deepseek/tests/e2e/test_live_contract.py` and record only sanitized results.
- [ ] T047 Review the final release diff for Core/package version alignment, baseline-contract alignment, generated-artifact drift, unintended public-facade changes, raw reasoning/tool data, and secrets across `pyproject.toml`, `packages/organizations/deepseek/`, `.github/`, `docs/`, `specs/001-baseline-contract/`, and `specs/002-add-deepseek/`.

---

## Dependencies & Execution Order

### Phase dependencies

- **Setup (Phase 1)** has no prerequisites.
- **Foundational (Phase 2)** depends on Phase 1 and blocks every user story.
- **US1 (Phase 3)** is the MVP. It depends only on the setup/foundation and validates the direct-package route.
- **US2 (Phase 4)** depends on the foundation; it adds the optional Core extra and known-missing diagnostic without changing the facade work from US1.
- **US3 (Phase 5)** depends on US1's Responses implementation and the opaque response metadata from Phase 2.
- **US4 (Phase 6)** depends on US3's adapter capability preflight and normalization paths.
- **Polish (Phase 7)** follows all selected user-story work. T046 requires separate maintainer authorization and runs only after TestPyPI candidates exist.

### User-story dependency graph

```text
Setup -> Foundation -> US1 (direct package MVP) -> US3 -> US4 -> Release polish
                    -> US2 (Core optional extra and diagnosis) --/
```

### Parallel opportunities

- T007 and T008 use separate test modules; T020, T021, and T022 use separate capability, mocked, and live-test modules.
- T035 and T036 can proceed in parallel after the provider behavior is stable; T038 and T039 can proceed in parallel after T037.
- Do not parallelize tasks that edit `adapter.py`, `streaming.py`, `pyproject.toml`, or `test_deepseek_adapter.py`.

### Parallel execution examples by user story

```text
US1: after the foundation, start T007 and T008 together; then perform T009 -> T010 -> T011 -> T012 -> T013 -> T014.
US2: after the foundation, perform T015 -> T016 -> T017 -> T018 -> T019; this story may proceed beside US1 after its test contract is stable.
US3: after US1, start T020, T021, and T022 together; then perform T023 -> T024 -> T025 -> T026 -> T027 -> T028.
US4: after US3, start T029 and T030 together; then perform T031 -> T032 -> T033 -> T034.
```

## Implementation Strategy

### MVP first

1. Finish T001–T006.
2. Complete T007–T014.
3. Validate a clean direct installation of Core 0.9.6 plus DeepSeek 0.1.0 with mocked text, async, and streaming behavior.
4. Stop for review before expanding capability scope.

### Incremental delivery

1. Add US2 to make the recommended `llm-api-adapter[deepseek]` installation and missing-package remedy available.
2. Add the complete verified Flash capability boundary in US3 through adapter behavior and contract evidence; never add undocumented aliases, registry extensions, or fallback modes.
3. Add the document and time-of-use cost boundary in US4.
4. Finish deterministic CI, TestPyPI candidate evidence, generated documentation, then the explicitly authorized live check.
