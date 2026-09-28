# Implementation Plan: Core / Plugin Architecture Refactor (0.9.8)

**Branch**: `main` (no feature branch created; active Spec Kit feature `004-core-plugin-refactor`) | **Date**: 2026-09-25 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/004-core-plugin-refactor/spec.md`; release scope from the [LLM API Adapter Implementation Plan](https://app.notion.com/p/34f33dd99fc8812ea5f2eae262910ab3) and its 0.9.8 section.

## Summary

Prepare Core `llm-api-adapter` 0.9.8 as a backward-compatible refactor of model capability evidence, automatic cache accounting, and external-organization metadata consistency. Use the provider-neutral baseline defined by the project constitution without creating a competing contract. Add an explicit exact-model exception list to the model registry; select baseline-positive scenarios by default for capabilities with shared E2E checks, keep that route for `pass` exceptions normalized by an adapter, and route other exceptions where a distinct scenario exists. Keep the existing common chat, streaming, and application-tool-loop E2E checks for every model without adding separate tool-choice or continuation requests. A missing profile is a certification error. Generalize provider-confirmed automatic cache-read and separately priced cache-write usage with optional independently verified static rates while preserving tiered and dispatch-time pricing. Record rates only when those components can occur during an ordinary adapter request; leave opt-in cache modes, TTL controls, cache resources, and storage outside 0.9.8. Detect distribution-name drift across Core, package metadata, E2E profiles, and CI selection. Keep plugin contracts, separate package release jobs, and provider credential boundaries intact. xAI and all other external organizations remain external packages.

## Technical Context

**Language/Version**: Python `>=3.10`; CI covers 3.10–3.14. Local `.venv/Scripts/python.exe` is Python 3.14.3.

**Primary Dependencies**: Core `requests>=2.32`; optional `httpx>=0.28` for async and explicit sync HTTPX. `pytest` and existing test dependencies for deterministic checks. No new runtime or provider SDK dependency is planned.

**Storage**: Version-controlled JSON model catalogues and package/project metadata. No database, persisted cache, or new user state.

**Testing**: `pytest` unit, mocked integration, shared conformance, and collection-time E2E-selection checks without network or credentials. Paid provider E2E remains in separate post-publish candidate lanes. Canonical Python 3.10 coverage floor remains 90%.

**Target Platform**: OS-independent Python library; release artifacts are separate Core and organization wheels.

**Project Type**: Core package at repository root plus six independently versioned organization distributions under `packages/organizations/`.

**Performance Goals**: No new provider request, cache lookup, or remote metadata query on the request path. Profile selection and accounting use local validated metadata. No numeric latency target is introduced by this refactor.

**Constraints**:

- Preserve the public facade and constructor, both supported synchronous transports, optional async installation, plugin entry-point API, and current package boundaries.
- Use the constitution's canonical provider-neutral baseline and capability catalogue as the capability source; retain model/provider-specific limitations in the exception profile even when a package adaptation fulfills the public baseline.
- Every first-party registered model needs an explicit exception list, including an empty list when it follows the baseline. Each declared exception needs a semantic `behavior_id` as well as its behavior description. An older third-party plugin stays usable at runtime, but cannot obtain profile-based certification without a profile.
- Provider reports are the only source of automatic cache-read and cache-write quantities. Registry rates apply only to components that can occur during an ordinary adapter request; provider support that requires an unsupported opt-in cache mode is not recorded. Provider-parsed partial usage exposes `None` for omitted input/output counts, including through direct `ChatResponse.from_*` calls, so omitted counts cannot masquerade as zero. An incomplete cache split or missing cost component cannot produce a complete discounted-input estimate or total.
- Dynamic DeepSeek rates remain selected at dispatch by its package; do not flatten them into a static registry rate.
- Pull requests remain credential-free. Model capability selection must not select a CI provider line or grant a secret.
- Do not move providers into Core, change provider contracts, add public cache control, or start service-provider/deployment work.

**Scale/Scope**: 57 current model entries across nine catalogues (three built-in organizations and six external packages), four public execution modes, existing common conformance/E2E scenarios, and six external distribution names. Current counts are an audit input, not a hard-coded registry limit.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Design evidence | Pre-design | Post-design |
| --- | --- | --- | --- |
| Stable provider-neutral public contract | Add optional provider-confirmed automatic cache usage metadata; preserve constructor, the existing `cached_tokens` cache-read meaning, aggregate cost fields, and installation errors. Document the intentional unknown-cost result when an incurred component split is absent. | Pass | Pass |
| Shared contract, isolated organization behavior | Core owns common profile validation and accounting; provider parsing, exact exceptions, and time-dependent DeepSeek rate selection stay organization-owned. No package moves. | Pass | Pass |
| Registry and abstraction first | Model registry becomes the source of exact-model exception lists and optional static automatic cache-read/cache-write rates; no model-name heuristics, cache-mode flags, opt-in-only prices, or provider-wide inferred defaults. | Pass | Pass |
| Deterministic contract evidence and baseline profiles | Every exact model gets applicable shared baseline-positive checks by default; declared exceptions with distinct shared E2E scenarios route to verified exception evidence. Missing profiles or scoped evidence fail validation. PR tests stay local; live E2E remains the existing bounded release gate. | Pass | Pass |
| Lightweight, safe extensibility | No new runtime dependency or automatic provider installation; legacy third-party plugins still register; secrets remain scoped to their existing jobs. | Pass | Pass |
| Change analysis and delivery discipline | The design is grounded in current registry, response, plugin, package, and CI evidence. Before editing a shared interface, implementation must audit every affected implementation and caller, then update relevant documentation and inspect the diff. | Pass | Pass |

No constitution violation or waiver is required. The design explicitly preserves runtime loading of older third-party plugins; only profile-dependent test certification rejects a missing model exception profile. If implementation reveals a real plugin API change is unavoidable, stop for a separate decision instead of widening 0.9.8 silently.

## Design Decisions

1. **Keep the baseline singular.** Maintain a strict catalogue of model-dependent capability identifiers governed by the constitution. Shared facade, normalization, transport, error, pricing, and cleanup invariants are always tested and are not discretionary model flags. See [research](./research.md) and [data model](./data-model.md).
2. **Make exact-model exceptions the test input.** Add an explicit `capability_exceptions` list to each first-party model entry. An empty list, or a capability omitted from that list, means its existing shared baseline-positive check remains required when one exists. Each exception carries a stable, value-independent `behavior_id` and verified prose description; exact values and limits remain in their dedicated structured fields. Reuse the shared IDs `pass`, `ignored`, `rejected_before_transport`, and `none_falls_back_to_minimum`, introducing a provider-specific ID only when the shared outcome and structured metadata cannot express the behavior. `behavior_id: "pass"` records a model/provider limitation compensated by the package, such as Mistral PDF via OCR, and keeps only the baseline-positive E2E check; package unit or integration tests verify the normalization itself. Other `(capability_id, behavior_id)` pairs route to expected-deviation checks where distinct shared E2E scenarios exist. Keep profile parsing optional for legacy third-party runtime compatibility; fail conformance/E2E selection if a profile itself is absent or invalid.
3. **Select scenarios within, not between, provider lanes.** Replace organization-level feature deselection with per-model exception routing for capabilities represented by existing shared E2E requests. The test-only scenario catalogue maps baseline capabilities to positive routes and non-`pass` behavior pairs with existing distinct E2E checks to replacement routes; `pass` exceptions reuse their positive route. It reports missing or duplicate routes within that scope and does not parse prose or store pytest node IDs in runtime model metadata. Reuse current common and package E2E functions without adding request forms; remove routes to nonexistent test nodes and keep untested variants as registry facts outside shared E2E selection. The common application-tool loop runs once per model with a viable tool-choice mode; fine-grained `tool_choice_*` and `provider_continuation` IDs remain registry facts without separate shared E2E routes. Extend `.github/scripts/select_e2e_lanes.py` to recognize the new shared test-selection modules as E2E-affecting paths; it remains responsible only for affected lanes. `.github/workflows/ci-dev-release.yml` remains responsible for separate publication, jobs, and secrets.
4. **Normalize automatic cache usage without guessing.** Audit exact models and ordinary adapter requests first. Add optional independently verified `cache_read_input_per_1m` and `cache_write_input_per_1m` rates only for automatically incurred components whose quantities the provider reports; add no cache-mode flag and no opt-in-only pricing. Retain the existing public `cached_tokens` meaning as cache-read/cache-hit usage and append optional cache-write usage while preserving the first three `Usage` constructor positions and their zero defaults. Provider-parsed partial responses expose `None` for omitted input/output counts. Update existing `ChatResponse.from_*` factories for that rule so direct calls behave the same as adapter calls. Extract provider-specific automatic-cache wire fields in organization adapters; do not add provider wire parsing to the shared response model or infer cache activity from request settings. Extend the existing `ChatResponse.apply_pricing` and `apply_cost_breakdown` path as well as `LLMAdapterBase` tier selection, preserving their direct calling patterns. Keep `cost_input` as the combined ordinary-plus-read-plus-write input charge, select the tier by confirmed total input, and retain package-owned non-static rate selection. Cover DeepSeek's separate streaming usage parser. Update current Z.ai/Kimi expectations that assume a cache miss when cache usage is absent. Document the 0.9.8 migration check for `None` before arithmetic on parsed token counts.
5. **Check organization metadata at repository validation time.** Compare Core known packages, extras, actual package manifests/entry points, E2E profiles, and CI selector. Remove safe duplicate test-only distribution strings, but keep static build metadata and explicit credential-bearing workflow jobs.
6. **Document only actual changed behavior.** Update Core and affected package pricing/compatibility guides, contributor validation instructions, and the constitution's observable usage/pricing clauses after implementation without weakening provider admission criteria; keep the 0.9.8 scope out of service-provider and xAI migration work.

## Project Structure

### Documentation (this feature)

```text
specs/004-core-plugin-refactor/
├── spec.md
├── checklists/requirements.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/core-plugin-refactor.md
└── tasks.md                       # Created later by $speckit-tasks
```

### Source Code and validation touchpoints (repository root)

```text
pyproject.toml
README.md
CONTRIBUTING.md
pytest.ini
src/llm_api_adapter/
├── organization_registry.py
├── llm_registry/{llm_registry.py,llm_registry.json,organizations/*.json}
├── models/responses/chat_response.py
└── adapters/base_adapter.py
packages/organizations/{mistral,xai,qwen,kimi,deepseek,zai}/
├── pyproject.toml
├── src/<organization_package>/registry/organizations/*.json
└── tests/
tests/
├── unit/{llm_registry,conformance,adapters,models/responses}/
├── unit/{test_organization_plugins.py,test_ci_e2e_lane_selection.py}
└── e2e/{conftest.py,harness.py,test_*.py}
.github/
├── scripts/select_e2e_lanes.py
└── workflows/{ci-dev.yml,ci-main.yml,ci-dev-release.yml}
```

**Structure Decision**: Extend existing registry, response, common-test, and package locations. The new scenario-selection logic belongs next to shared conformance/E2E support; package-specific parsers and dynamic pricing remain in their packages. No new distribution or service-provider layer is introduced.

## Validation Sequence

1. Validate registry schema, 57 first-party profiles, scenario catalogue coverage, legacy third-party registration, and known-but-uninstalled errors with deterministic tests.
2. Validate automatic cache-read/cache-write usage, optional independent rates, opt-in-only cache exclusion, tier boundaries, missing input/output versus explicit zero, direct `ChatResponse.apply_pricing`/`apply_cost_breakdown`, non-static DeepSeek windows, and sync/async/streaming finalization using mocked provider responses.
3. Validate organization/distribution consistency, missing package detection, selector outputs, and E2E collection without secrets.
4. Run focused tests and the existing full credential-free unit/integration suite across the Python matrix; retain the Python 3.10 coverage threshold and both sync transports.
5. During the later release process, run only the established post-publish provider E2E lanes against exact TestPyPI candidates with each lane's own credential. A local or pre-merge live run does not replace this gate.

## Complexity Tracking

No constitution violations or complexity waivers are required. The main implementation risk is auditing every model-specific capability decision without copying an organization-wide default. The second is moving only automatic, separately measurable cache pricing into shared metadata while excluding opt-in-only modes and retaining provider-specific dynamic pricing and honest incomplete-cost behavior.
