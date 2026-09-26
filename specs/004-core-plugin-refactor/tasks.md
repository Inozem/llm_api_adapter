# Tasks: Core / Plugin Architecture Refactor (0.9.8)

**Input**: `specs/004-core-plugin-refactor/{spec.md,plan.md,research.md,data-model.md,contracts/core-plugin-refactor.md,quickstart.md}` and the canonical `specs/001-baseline-contract/spec.md`.

**Tests**: Required by the feature specification: deterministic profile, scenario-selection, accounting, metadata, and compatibility evidence. Write each story's tests before the behavior they exercise; run them without credentials.

**Scope**: Refactor Core for 0.9.8. Preserve the public facade, legacy plugin registration, provider contracts, external package boundaries, independent release jobs, and credential isolation. Do not add cache-control requests, migrate xAI into Core, or start service-provider work. Paid E2E and publication belong to the later authorized release gate.

## Format: `[ID] [P?] [Story] Description`

- `[P]` means the task can proceed alongside the other indicated tasks because it uses different files and has no unfinished dependency.
- `[USn]` identifies the user story; setup, foundational, and polish tasks have no story label.
- All paths below are relative to the Git repository root (`llm_api_adapter/llm_api_adapter` in this workspace).

## Phase 1: Setup

**Purpose**: Freeze the change surface before editing shared interfaces.

- [x] T001 Record the complete baseline-to-code audit in `specs/004-core-plugin-refactor/research.md`: every affected `ModelSpec`/`PricingTier`/`Usage` caller and subclass, nine registry catalogues and their current model IDs, shared conformance/E2E scenarios, package cache-pricing hooks, optional extras, plugin entry points, CI lane outputs, and public compatibility imports; record official provider documentation sources for later exact-model decisions.

**Checkpoint**: The audit identifies where every common behavior and model-specific exception is currently represented.

---

## Phase 2: Foundational

**Purpose**: Establish one baseline capability vocabulary shared by profiles and test selection.

- [x] T002 Add failing catalogue tests in `tests/unit/llm_registry/test_model_capabilities.py` for unique stable IDs, distinct streaming/tool-choice/image/PDF/reasoning/outcome variants, duplicate or unknown IDs, and the `model-dependent` versus `always-on` scope. Keep the catalogue as the single executable list of IDs and baseline traceability in `data-model.md`.
- [x] T003 Implement the version-controlled capability IDs and `model-dependent` versus `always-on` scope in `src/llm_api_adapter/llm_registry/model_capabilities.py`; keep pytest scenario node IDs out of runtime metadata and make T002 pass.

**Checkpoint**: Profiles and selectors can refer to the same canonical capability IDs without making Core invariants optional.

---

## Phase 3: User Story 1 — Declare Model Exceptions to the Baseline (P1, MVP)

**Goal**: Every first-party exact model has an explicit exception list against the baseline; old third-party plugins remain usable without one.

**Independent test**: Validate all current first-party model entries; verify that an empty exception list selects baseline-positive checks, a declared exception selects its exception check, and an unknown/duplicate exception or missing behavior fails. Register a legacy third-party plugin without a profile and verify runtime loading still works.

### Tests

- [x] T004 [P] [US1] Write failing parser and validation tests in `tests/unit/llm_registry/test_model_profile.py`: an explicit empty exception list differs from a missing legacy profile; exception IDs are unique known `model-dependent` capabilities; `behavior` is required; each entry applies only to its exact capability ID and rejects nested variant overrides; and malformed, unknown, duplicate, or always-on exceptions fail with model and capability named. T016 checks that every first-party model explicitly declares a list.
- [x] T005 [P] [US1] Add legacy third-party plugin tests in `tests/unit/test_organization_plugins.py`: missing profile still registers and serves requests, while profile-based certification reports that the model is uncertified; retain known-but-uninstalled installation guidance.

### Implementation

- [x] T006 [US1] Parse an optional exact-model exception profile in `src/llm_api_adapter/llm_registry/llm_registry.py`, validate the T004 constraints using T003's catalogue, preserve existing `ModelSpec` construction/alias rules, and distinguish a legacy absent profile from an invalid present profile.
- [ ] T007 [P] [US1] Declare each exact OpenAI model's explicit `capability_exceptions` list and source-check every listed exception in `src/llm_api_adapter/llm_registry/organizations/openai.json`; an unlisted capability keeps its baseline-positive check.
- [ ] T008 [P] [US1] Declare each exact Anthropic model's explicit `capability_exceptions` list and source-check every listed exception in `src/llm_api_adapter/llm_registry/organizations/anthropic.json`; reconcile exceptions with reasoning and request-rule limits.
- [ ] T009 [P] [US1] Declare each exact Google model's explicit `capability_exceptions` list and source-check every listed exception in `src/llm_api_adapter/llm_registry/organizations/google.json`; preserve distinct input forms and outcome behavior.
- [ ] T010 [P] [US1] Declare each exact Mistral model's explicit `capability_exceptions` list and source-check every listed exception in `packages/organizations/mistral/src/llm_api_adapter_mistral/registry/organizations/mistral.json`, including document/OCR boundaries.
- [ ] T011 [P] [US1] Declare each exact xAI model's explicit `capability_exceptions` list and source-check every listed exception in `packages/organizations/xai/src/llm_api_adapter_xai/registry/organizations/xai.json`; keep xAI external.
- [ ] T012 [P] [US1] Declare each exact Qwen model's explicit `capability_exceptions` list and source-check every listed exception in `packages/organizations/qwen/src/llm_api_adapter_qwen/registry/organizations/qwen.json`, including image/document variants.
- [ ] T013 [P] [US1] Declare each exact Kimi model's explicit `capability_exceptions` list and source-check every listed exception in `packages/organizations/kimi/src/llm_api_adapter_kimi/registry/organizations/kimi.json`, including file and cache-related evidence boundaries.
- [ ] T014 [P] [US1] Declare each exact DeepSeek model's explicit `capability_exceptions` list and source-check every listed exception in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/registry/organizations/deepseek.json`; retain package-owned dynamic pricing.
- [ ] T015 [P] [US1] Declare each exact Z.ai model's explicit `capability_exceptions` list and source-check every listed exception in `packages/organizations/zai/src/llm_api_adapter_zai/registry/organizations/zai.json`, including documented structured-output limits.
- [ ] T016 [US1] Add a deterministic inventory test in `tests/unit/llm_registry/test_model_profile_inventory.py` that walks all nine first-party catalogues (currently 57 model entries), requires an explicit exception list on every model, rejects malformed/unknown/duplicate/always-on exceptions, and checks listed exceptions against existing reasoning/request-rule metadata without hard-coding 57 as a future limit.
- [ ] T017 [US1] Update `tests/unit/test_organization_profile_compatibility.py` to verify that profile parsing does not change built-in or external registry loading, exact-model resolution, aliases, plugin discovery, or current public facade behavior.

**Checkpoint**: US1 passes its deterministic tests without scenario-selection changes.

---

## Phase 4: User Story 2 — Get Complete Common Contract Evidence (P1)

**Goal**: Select baseline-positive evidence by default and route only declared exact-model exceptions to their matching evidence inside an already selected provider lane.

**Independent test**: Supply representative built-in and external profiles to the selector. An empty list selects baseline-positive checks; each declared exception selects its documented common or package-local check; missing profiles or missing evidence fail; package-local evidence replaces only the positive check for that exact exception.

### Tests

- [ ] T018 [P] [US2] Write failing selector tests in `tests/unit/conformance/test_capability_selection.py` for baseline-default, declared-exception, missing-profile, missing-scenario, duplicate-evidence, and package-supplement cases; assert always-on scenarios remain selected regardless of exceptions.
- [ ] T019 [P] [US2] Write failing credential-free collection tests in `tests/unit/test_e2e_profile_selection.py` for two models in one organization with different exception lists: collection must select baseline-positive versus exception scenarios as declared and error on invalid exceptions without package installation, provider keys, or a network call.

### Implementation

- [ ] T020 [US2] Map each `model-dependent` T003 capability to a baseline-positive scenario and documented exception scenarios, with package-local evidence slots where needed, in `tests/capability_scenarios.py`; map `always-on` capabilities only to unconditional shared checks, and fail deterministic validation if any model's baseline or declared exception lacks evidence.
- [ ] T021 [US2] Implement exact-model scenario selection and coverage-gap diagnostics in `tests/capability_selection.py`: choose the baseline-positive scenario unless the model declares that capability as an exception, then choose only its matching exception evidence; never let that exception remove unrelated shared checks.
- [ ] T022 [US2] Refactor `tests/e2e/conftest.py` so `E2EOrganizationProfile` retains only lane/install/key/operation settings and exact-model fixtures use T021 exception lists; remove organization-wide `supported_features` deselection and make missing or invalid profiles a collection error rather than a skip.
- [ ] T023 [P] [US2] Give every common chat, continuation, streaming, tool, and tool-choice case an explicit capability/scenario link and exact-model parameterization in `tests/e2e/test_llm_adapter_chat.py`, `tests/e2e/test_streaming.py`, and `tests/e2e/test_tools_auto_loop.py`.
- [ ] T024 [P] [US2] Give structured-output, image, and document scenarios distinct exact-model variant links in `tests/e2e/test_json_schema.py`, `tests/e2e/test_vision.py`, and `tests/e2e/test_file_uploads.py`; retain separate positive and declared-exception assertions.
- [ ] T025 [P] [US2] Keep always-on error and sync-HTTPX contract checks unconditional, and keep Mistral OCR as an additive package-specific check, in `tests/e2e/test_errors.py`, `tests/e2e/test_sync_httpx.py`, and `tests/e2e/test_mistral_ocr_costs.py`.
- [ ] T026 [US2] Replace `_TERMINAL_OUTCOME_CAPABILITIES` and outcome-driven skips with profile-derived positive/exception evidence in `tests/unit/conformance/test_portable_profile_matrix.py`; an unexpected refusal or incomplete result must fail a positive test.
- [ ] T027 [US2] Update async structured-output, media, and outcome scenarios in `tests/e2e/test_async.py` to use exact-model exception routing; remove `pytest.skip` after malformed/non-JSON responses when no matching exception is declared, and assert documented exceptions explicitly.
- [ ] T028 [US2] Connect declared external exceptions to matching package-local checks in `packages/organizations/zai/tests/e2e/test_capability_boundaries.py`, `packages/organizations/kimi/tests/e2e/test_live_contract.py`, `packages/organizations/deepseek/tests/e2e/test_live_contract.py`, and `packages/organizations/qwen/tests/e2e/test_live_contract.py`; for Mistral and xAI add explicit common rejection/special-behavior scenarios in the applicable `tests/e2e/test_llm_adapter_chat.py`, `tests/e2e/test_json_schema.py`, `tests/e2e/test_vision.py`, or `tests/e2e/test_file_uploads.py`, with async variants handled by T027; require T029 to fail any remaining evidence gap without removing unrelated baseline-positive coverage.
- [ ] T029 [US2] Complete T018 and T019 with coverage assertions over every first-party profile in `tests/unit/conformance/test_capability_selection.py` and `tests/unit/test_e2e_profile_selection.py`; report the model, exception, and missing scenario for each gap.
- [ ] T030 [US2] Add `tests/capability_selection.py` and `tests/capability_scenarios.py` to the E2E-affecting shared paths in `.github/scripts/select_e2e_lanes.py`, and extend `tests/unit/test_ci_e2e_lane_selection.py` to prove each path selects the affected provider E2E lanes; preserve changed-path-only lane selection, separate exact-candidate jobs, and job-local secrets in `.github/workflows/ci-dev-release.yml`.

**Checkpoint**: US2's selector and collection tests pass without live requests; each provider's authorized E2E lane remains a separate release step.

---

## Phase 5: User Story 3 — Read Honest Cached Input Accounting (P2)

**Goal**: Expose only provider-confirmed cached input and complete, correctly tiered costs when every required component is known.

**Independent test**: Exercise confirmed positive and zero cache splits, absent/malformed/contradictory splits, absent verified rate, tier boundaries, non-token charges, and DeepSeek peak/off-peak windows in mocked sync, async, and streaming responses.

### Tests and rate audit

- [ ] T031 [P] [US3] Write failing registry tests in `tests/unit/llm_registry/test_llm_registry.py` for optional `cached_input_per_1m`: finite nonnegative exact-model values only, no fallback to `input_per_1m`, existing strictly increasing tiers and final open tier, and tier selection by total reported input tokens.
- [ ] T032 [P] [US3] Write failing common `Usage` compatibility tests in `tests/unit/models/responses/test_chat_response.py` and `tests/unit/streaming/test_chunk_buffer.py`, plus mocked provider-extraction tests in `tests/unit/adapters/test_openai_adapter.py`, `tests/unit/adapters/test_anthropic_adapter.py`, and `tests/unit/adapters/test_google_adapter.py`: preserve positional `Usage(input_tokens, output_tokens, total_tokens)` and its zero defaults; test direct `ChatResponse.from_openai_response`, `from_openai_responses_response`, `from_anthropic_response`, and `from_google_response` calls with omitted input/output/total fields yielding public `None` versus reported `0`, including Google's composed output and Anthropic's exact-total rule; verify that `StreamChunk` and `StreamChunkBuffer` copies retain optional cached usage; update existing empty-usage assertions; accept only provider-confirmed `cached_tokens`; reject malformed/negative/noninteger counts; cover sync/async/streaming final responses and wholly absent usage.
- [ ] T033 [P] [US3] Write failing direct `ChatResponse.apply_pricing`/`apply_cost_breakdown` and adapter cost tests in `tests/unit/adapters/test_pricing_lifecycle.py` for `0 <= cached_tokens <= input_tokens` when input is known, nonnegative integer counts, ordinary-plus-cached input without double counting, public `None` versus reported zero, missing/corrupt cache split, unknown cached rate, output-only known cost, non-token currency completeness, preserved positional pricing calls, and sync/async/streaming parity.
- [ ] T034 [US3] Record official exact-model cached-rate evidence and the applicable pricing context in `specs/004-core-plugin-refactor/research.md`; mark unverified rates unknown and identify dynamic DeepSeek rates before changing registry pricing data.

### Implementation

- [ ] T035 [US3] Add optional finite nonnegative `cached_input_per_1m` parsing to `PricingTier` in `src/llm_api_adapter/llm_registry/llm_registry.py`, preserve existing constructor positions and overrides, and select the applicable rate tier only when provider-parsed `usage.input_tokens` is confirmed and not `None`.
- [ ] T036 [US3] Append optional `cached_tokens` to `Usage` in `src/llm_api_adapter/models/responses/chat_response.py`, preserving its first three positional fields, zero defaults, and direct-construction behavior while widening their annotations to permit `None`; update existing `ChatResponse.from_*` factories so omitted common input/output/total counts in provider-parsed partial usage become public `None` even on direct calls, except an existing exact-total derivation from both confirmed components. Preserve cached tokens when copying `Usage` in `src/llm_api_adapter/models/responses/stream_chunk.py` and `src/llm_api_adapter/llms/streaming.py`. Extract provider-specific cache wire fields in `src/llm_api_adapter/adapters/openai/payloads.py`, `src/llm_api_adapter/adapters/openai/streaming.py`, `src/llm_api_adapter/adapters/anthropic/payloads.py`, `src/llm_api_adapter/adapters/anthropic/streaming.py`, `src/llm_api_adapter/adapters/google/payloads.py`, and `src/llm_api_adapter/adapters/google/streaming.py`, without adding new provider wire parsing to `chat_response.py` or inferring hits from request settings.
- [ ] T037 [US3] Implement shared input/output/total accounting in `src/llm_api_adapter/models/responses/chat_response.py` (`apply_pricing` and `apply_cost_breakdown`) and `src/llm_api_adapter/adapters/base_adapter.py` (tier/rate selection), preserving existing direct/positional pricing calls while accepting an optional verified cached rate: `cost_input` remains aggregate ordinary plus cached cost, `cost_output` may survive `input_tokens=None`, and `cost_total` is `None` whenever a provider input/output count, required rate, or incurred non-token component is unknown or has incompatible currency.
- [ ] T038 [US3] Add T034-verified static cached rates only to exact model tiers in `src/llm_api_adapter/llm_registry/organizations/openai.json`, `src/llm_api_adapter/llm_registry/organizations/anthropic.json`, and `src/llm_api_adapter/llm_registry/organizations/google.json`; leave every unverified rate absent.
- [ ] T039 [P] [US3] Revise Z.ai mocked cache-usage and missing-split expectations in `packages/organizations/zai/tests/test_zai_adapter.py`, including reported zero versus omitted input/output and partial-cost cases; write assertions before package implementation.
- [ ] T040 [P] [US3] Revise Kimi mocked cache-usage and missing-split expectations in `packages/organizations/kimi/tests/test_kimi_adapter.py`, including reported zero versus omitted input/output and partial-cost cases; write assertions before package implementation.
- [ ] T041 [P] [US3] Add DeepSeek peak/off-peak, cache-split, public `None` versus reported zero, unknown-rate, and streaming finalization regression cases in `packages/organizations/deepseek/tests/test_deepseek_adapter.py`; explicitly exercise its separate streaming usage normalizer and write assertions before package implementation.
- [ ] T042 [P] [US3] Route Z.ai provider-confirmed cached usage, `None` for omitted input/output counts, and any verified exact-model static rates through shared accounting in `packages/organizations/zai/src/llm_api_adapter_zai/adapter.py`, `packages/organizations/zai/src/llm_api_adapter_zai/registry/cache_pricing.py`, and `packages/organizations/zai/src/llm_api_adapter_zai/registry/organizations/zai.json`; preserve package-specific wire parsing.
- [ ] T043 [P] [US3] Route Kimi provider-confirmed cached usage, `None` for omitted input/output counts, and any verified exact-model static rates through shared accounting in `packages/organizations/kimi/src/llm_api_adapter_kimi/adapter.py`, `packages/organizations/kimi/src/llm_api_adapter_kimi/registry/cache_pricing.py`, and `packages/organizations/kimi/src/llm_api_adapter_kimi/registry/organizations/kimi.json`; remove assumed cache misses.
- [ ] T044 [P] [US3] Keep DeepSeek dispatch-time peak/off-peak rate selection in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/registry/cache_pricing.py` while passing its selected rate set and provider-confirmed cache split through `packages/organizations/deepseek/src/llm_api_adapter_deepseek/adapter.py` to shared accounting; update the separate usage parser in `packages/organizations/deepseek/src/llm_api_adapter_deepseek/streaming.py` so streaming finalization retains confirmed counts, exposes `None` for omitted input/output counts, and never fabricates a complete cost. Do not flatten dynamic rates into `packages/organizations/deepseek/src/llm_api_adapter_deepseek/registry/organizations/deepseek.json`.
- [ ] T045 [US3] First add failing omitted-versus-zero and sync/async/streaming parity cases in `packages/organizations/mistral/tests/test_mistral_adapter.py`, `packages/organizations/xai/tests/test_xai_adapter.py`, `packages/organizations/qwen/tests/test_qwen_adapter.py`, `tests/unit/adapters/test_pricing_lifecycle.py`, and `tests/unit/adapters/test_base_adapter.py`; then expose public `None` for omitted provider input/output counts in `packages/organizations/mistral/src/llm_api_adapter_mistral/adapter.py`, `packages/organizations/xai/src/llm_api_adapter_xai/adapter.py`, `packages/organizations/xai/src/llm_api_adapter_xai/streaming.py`, `packages/organizations/qwen/src/llm_api_adapter_qwen/adapter.py`, and `packages/organizations/qwen/src/llm_api_adapter_qwen/streaming.py`.

**Checkpoint**: US3's mocked accounting tests pass for built-in and affected external providers with no invented savings or complete totals.

---

## Phase 6: User Story 4 — Keep External Organizations Consistent (P2)

**Goal**: Detect identity and distribution drift across all six external packages without changing runtime plugin discovery or independent releases.

**Independent test**: Compare Core mapping, extras, package manifests/entry points, E2E profiles, and CI lanes. Mutate one source to remove an organization or change its distribution and require a diagnostic naming both organization and source.

### Tests

- [ ] T046 [P] [US4] Write failing fixture-based metadata tests in `tests/unit/test_external_organization_metadata.py` for all six external organizations and missing package/extra/entry point/E2E profile/CI lane, duplicate key, and normalized distribution-name mismatch cases.
- [ ] T047 [P] [US4] Add facade and plugin-compatibility assertions in `tests/unit/test_organization_plugins.py` for lazy discovery, known-but-uninstalled guidance, and unchanged registration behavior after metadata cleanup.

### Implementation

- [ ] T048 [US4] Implement a test-only repository metadata reader and actionable consistency validator in `tests/external_organization_metadata.py`, comparing `src/llm_api_adapter/organization_registry.py`, Core `pyproject.toml`, all six manifests under `packages/organizations/`, `tests/e2e/conftest.py`, and `.github/scripts/select_e2e_lanes.py`; avoid runtime TOML parsing or importing uninstalled packages.
- [ ] T049 [US4] After T022's edits to the shared file are complete, derive only safe test-side distribution names from `KNOWN_ORGANIZATION_PACKAGES` in `tests/e2e/conftest.py`; retain explicit provider environment and operation settings and leave static package/build metadata independent, while keeping US4's fixture-based metadata validation independent of US2.
- [ ] T050 [US4] Reconcile the six existing Core extra requirements in `pyproject.toml` and the manifests `packages/organizations/mistral/pyproject.toml`, `packages/organizations/xai/pyproject.toml`, `packages/organizations/qwen/pyproject.toml`, `packages/organizations/kimi/pyproject.toml`, `packages/organizations/deepseek/pyproject.toml`, and `packages/organizations/zai/pyproject.toml` with `src/llm_api_adapter/organization_registry.py` only where T048 finds drift; preserve each package's version range, entry-point API, and independent release metadata.
- [ ] T051 [US4] Verify metadata validation and path-to-lane behavior in `tests/unit/test_external_organization_metadata.py` and `tests/unit/test_ci_e2e_lane_selection.py`, including explicit jobs in `.github/workflows/ci-dev-release.yml`, absent PR credentials in `.github/workflows/ci-dev.yml`, and no change to provider-specific publication jobs.

**Checkpoint**: US4's deterministic metadata test detects a single missing or conflicting source and leaves existing installs and CI boundaries intact.

---

## Phase 7: Polish and Cross-Cutting Validation

**Purpose**: Document the refactor, finish release metadata, and run the deterministic gates before review.

- [ ] T052 [P] Document every declared built-in exact-model exception, cached-usage behavior, and incomplete-cost result in `README.md`, including intentionally unknown `cost_input`/`cost_total` for a distinct cached rate with no valid provider cache split and 0.9.8 migration guidance to check parsed partial usage counts for `None` before arithmetic.
- [ ] T053 [P] Update contributor instructions in `CONTRIBUTING.md` for explicit per-model exception lists (including empty lists), source verification, baseline-positive plus exception evidence, credential-free collection, and the separate post-publish E2E gate.
- [ ] T054 [P] Document every declared exact-model exception and its verified behavior in the corresponding compatibility guide `packages/organizations/mistral/README.md`, `packages/organizations/xai/README.md`, `packages/organizations/qwen/README.md`, `packages/organizations/kimi/README.md`, `packages/organizations/deepseek/README.md`, or `packages/organizations/zai/README.md`; update cache-pricing notes in the affected guides and do not copy the canonical baseline.
- [ ] T055 Run the focused deterministic story checks in `specs/004-core-plugin-refactor/quickstart.md`; once all four checkpoints pass, bump only the Core version from 0.9.7 to 0.9.8 in `pyproject.toml`, keeping the six external package versions and release jobs independent, then perform the complete post-bump gate in T056.
- [ ] T056 After T055, run the credential-free commands in `specs/004-core-plugin-refactor/quickstart.md`: focused registry/conformance/accounting/metadata tests, E2E `--collect-only` for every provider marker in `pytest.ini`, and the full `unit or integration` suite; verify Python 3.10–3.14 CI remains configured in `.github/workflows/ci-dev.yml` with the canonical 3.10 coverage floor at 90%, and record any unavailable local matrix lanes as CI evidence pending.
- [ ] T057 Synchronize the observable usage/pricing clauses in `specs/001-baseline-contract/spec.md` with optional provider-confirmed `cached_tokens`, public `None` for omitted counts in parsed partial usage, preserved direct construction/pricing calls, and incomplete-cost behavior without revising provider admission criteria; update the living boundary description in `specs/001-baseline-contract/architecture.md` for exact-model profiles, test-only scenario selection, and repository-only metadata validation, then review `git diff` and record findings and remaining post-publish E2E gates in `specs/004-core-plugin-refactor/quickstart.md`.

---

## Dependencies and Execution Order

- **Setup**: T001 precedes all source changes, as required by the constitution's shared-interface audit.
- **Foundation**: T002 → T003; T003 blocks profile and selector work.
- **US1**: T004 and T005 can run together after T003. T006 follows T004. T007–T015 can run in parallel after T006, then T016–T017 validate the complete first-party inventory and compatibility.
- **US2**: T018 and T019 can run after T003 using synthetic profiles; full-inventory integration waits for US1. T020 → T021 → T022; T023–T025 can proceed across distinct files, T026–T027 follow T022, and T028 follows T023–T024 where common scenario files overlap; T029–T030 complete full coverage validation.
- **US3**: T031–T033 can run after T001, independently of US1/US2. T034 precedes static-rate edits. T035–T037 establish shared behavior; T039–T041 write package regression tests, then T042–T044 can run in parallel. T038 and T045 finish the story.
- **US4**: T046–T047 can run after T001, independently of the other stories. T048 precedes T049–T050; T049's edit of `tests/e2e/conftest.py` follows T022's edit of the same file, while US4's fixture-based validation remains independent. T051 validates all sources and CI boundaries.
- **Polish**: T052–T054 can proceed after the affected stories stabilize; T052's Core guide and T054's package guides together document every declared exception. After all four focused story checkpoints pass, execute T055 → T056 → T057. Live E2E and publication remain later release gates.

## Parallel Examples

- **US1**: T007, T008, and T009 may populate the three separate built-in catalogue files while T010–T015 populate separate external package catalogues.
- **US2**: T023–T025 cover separate common scenario files, and T018 and T019 can be written together. T028's package-local portions are independent, while its common scenario edits wait for T023 and T024 in the files they share.
- **US3**: T031, T032, and T033 target different test files; T039–T041 are separate provider regression files; T042–T044 are separate provider implementation files.
- **US4**: T046 and T047 are independent test files. After T048 and T022, E2E profile cleanup (T049) can proceed alongside any required manifest reconciliation (T050), which uses separate files.

## Implementation Strategy

1. Complete setup and foundation, then US1 as the MVP. Validate the full first-party profile inventory and legacy plugin loading before expanding test selection.
2. Add US2 to obtain complete shared and package-specific evidence inside existing provider lanes; collection checks stay credential-free.
3. Complete US3 and US4 as separate increments, each with its own deterministic acceptance tests.
4. Finish documentation, Core 0.9.8 metadata, local deterministic validation, and diff review. Use the existing authorized post-publish provider lanes for final live evidence; no task here authorizes a live call or publication.
