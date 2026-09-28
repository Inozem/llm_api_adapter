# Feature Specification: Core / Plugin Architecture Refactor (0.9.8)

**Feature Branch**: Not created (specification prepared on `main`)
**Created**: 2026-09-25
**Status**: Draft
**Input**: User description: "Prepare Core 0.9.8 as a refactor of the boundary between Core and independently versioned organization packages, following the LLM API Adapter Implementation Plan in Notion."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Declare Model Exceptions to the Baseline (Priority: P1)

A maintainer can inspect a registered model's explicit exceptions to the established Core baseline. Every applicable behavior without a declared exception remains required by the baseline. Capabilities represented by shared E2E scenarios retain their normal positive check.

**Why this priority**: An undocumented provider deviation must fail the baseline check instead of silently removing evidence.

**Independent Test**: Review every registered model's exception list. For capabilities represented by shared E2E scenarios, verify that an empty list or a `pass` exception selects baseline-positive scenarios, and a non-`pass` exception selects its documented deviation scenario. An invalid exception is rejected.

**Acceptance Scenarios**:

1. **Given** a first-party model with no exceptions, **When** its profile is inspected, **Then** it has an explicit empty exception list and baseline-positive checks remain selected for every applicable model-dependent capability represented by a shared E2E scenario.
2. **Given** a first-party model with a confirmed exception, **When** its profile is inspected, **Then** the canonical capability ID, stable `behavior_id`, and exact expected behavior are explicit.
3. **Given** a model profile with an unknown, duplicate, always-on, or malformed exception, including a missing or malformed `behavior_id`, **When** the profile is validated, **Then** validation fails with the model and exception named.
4. **Given** an organization package makes an exceptional model capability satisfy the public contract through its own conversion or fallback, **When** the exact model's checks are selected, **Then** its `behavior_id: "pass"` selects the normal positive scenario, while package tests verify the adaptation itself.

---

### User Story 2 - Get Complete Common Contract Evidence (Priority: P1)

A provider maintainer runs conformance and authorized live verification for an exact model. Shared baseline-positive scenarios run by default; model exceptions change routing only where a distinct shared E2E scenario exists. The standard chat, streaming, and tool-loop tests run for every model.

**Why this priority**: Every model needs its existing shared baseline checks, and exceptions with matching existing E2E evidence must route without manually maintained gaps.

**Independent Test**: For representative built-in and external models, compare selected scenarios with their exception lists. For capabilities with distinct shared E2E scenarios, confirm that no exception or `pass` selects the baseline-positive check, another behavior ID selects its documented deviation check, and a missing profile or route fails selection. Confirm the common tool loop runs once per model.

**Acceptance Scenarios**:

1. **Given** a model has no declared exception for a capability with a shared E2E scenario, **When** selection is prepared, **Then** the applicable shared positive scenario is selected.
2. **Given** a model with an explicit `pass` exception for a capability with shared E2E evidence, **When** selection is prepared, **Then** the baseline-positive scenario is selected for its exact model and capability.
3. **Given** a model with a non-`pass` exception for a capability with a distinct shared E2E scenario, **When** checks are selected, **Then** its `(capability_id, behavior_id)` pair selects a documented rejection or deviation scenario, replacing only the baseline-positive scenario for that exact capability; other applicable shared scenarios remain selected.
4. **Given** a provider-specific CI line, **When** its tests are selected, **Then** line selection and credential access remain independently controlled and credentials remain unavailable to pull-request checks.

---

### User Story 3 - Read Honest Automatic Cache Accounting (Priority: P2)

An application developer can read provider-confirmed automatic cache reads and separately reported cache writes from an ordinary adapter request, together with a corresponding standard-rate cost estimate when the selected model has verified rates for every incurred component. The developer never sees assumed cache activity or a fabricated total when usage is incomplete. Cache modes that require caller opt-in remain outside this release and do not populate the registry.

**Why this priority**: Cache reads and separately priced writes can materially change the cost estimate, but partial provider usage must not look like complete billing evidence.

**Independent Test**: Compare results with complete automatic cache-read/cache-write usage, missing component usage, malformed usage, a model without a verified component rate, and an opt-in-only cache mode that remains unrepresented. Include a model whose pricing cannot be represented by one static rate.

**Acceptance Scenarios**:

1. **Given** a model whose ordinary adapter request can automatically incur cache reads or separately priced cache writes and complete provider-confirmed usage and rates, **When** a response is accounted for, **Then** ordinary input, cache reads, and cache writes are priced once without double counting.
2. **Given** usage that does not confirm an incurred cache component or its token quantity, **When** a response is accounted for, **Then** no cache quantity or savings are invented and an incomplete total is not presented as complete.
3. **Given** tiered or otherwise non-static pricing, **When** automatic cache usage is priced, **Then** the applicable existing rate rules remain correct.
4. **Given** a provider cache feature that requires cache-control parameters, TTL selection, a cache resource, or another caller opt-in unsupported by the adapter, **When** registry pricing is prepared for 0.9.8, **Then** that feature's rates and modes are not recorded or inferred.

---

### User Story 4 - Keep External Organizations Consistent (Priority: P2)

A release maintainer can verify that every supported external organization has consistent identity and distribution naming across Core discovery, optional installation declarations, E2E profiles, and CI line selection.

**Why this priority**: Drift between these sources can make an installable provider undiscoverable or leave its verification line incomplete.

**Independent Test**: Check the current external organization inventory, then introduce a missing package declaration or mismatched name in a test fixture and confirm that deterministic validation fails.

**Acceptance Scenarios**:

1. **Given** current supported external organizations, **When** their metadata is checked, **Then** each organization and distribution name matches across all four sources.
2. **Given** a missing external package or a conflicting distribution name, **When** deterministic validation runs, **Then** it identifies the affected organization and conflicting source.
3. **Given** a correctly installed external organization, **When** a caller selects it through the existing facade, **Then** discovery and public behavior remain backward compatible.

### Edge Cases

- A first-party model has no exception-list field: the profile is invalid. A capability represented by a shared E2E scenario and absent from a valid exception list still selects its baseline-positive check, so an undeclared deviation fails that check.
- A provider API lacks a direct input form but its package supplies the public baseline behavior, such as Mistral PDF through OCR: the capability remains a documented exception with `behavior_id: "pass"`; its baseline-positive check runs and package tests verify the normalization.
- An exception names an unknown or always-on capability, is duplicated, or lacks a valid `behavior_id` or verified behavior: validation fails.
- A model declares an exception for a capability with a distinct shared E2E scenario whose `(capability_id, behavior_id)` pair has no matching route: validation identifies the evidence gap.
- A package-specific test duplicates a common scenario: the common scenario is still required.
- A provider reports automatic cache-read or cache-write tokens without enough total input usage to establish complete accounting, or reports inconsistent quantities: confirmed fields may be retained, but unconfirmed costs and a complete total are unavailable. A parsed, omitted input or output count remains `None`; an explicitly reported zero remains `0`. Direct `Usage` construction retains its existing zero defaults and first three positional arguments.
- An incurred automatic cache component's price is absent or unverified: usage may be reported if confirmed, while input and total cost remain unavailable.
- A pricing schedule has tiers or other conditional rates: new cache-read or cache-write rates do not flatten or bypass those rules.
- A provider documents cache write, TTL, storage, or cache read only behind an unsupported opt-in request: registry data remains absent until that request behavior is implemented and separately specified.
- An external organization is absent from one metadata source, uses a different distribution name, or lacks an E2E profile: deterministic validation fails with an actionable mismatch.
- A user selects a known but uninstalled external organization: existing installation guidance remains distinct from an unknown-organization error.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Core 0.9.8 MUST use the project constitution as the sole provider-neutral baseline and MUST NOT create a competing baseline in this feature. The constitution's observable usage and pricing rules MUST include the additive 0.9.8 response behavior.
- **FR-002**: The model registry MUST be the authoritative source for each first-party model's explicit exceptions to direct baseline capability handling, including provider limitations compensated by package-owned adaptation. Package placement MUST NOT be used to infer exceptions.
- **FR-003**: Every first-party exact-model profile MUST include an explicit exception list, which MAY be empty. Each listed exception MUST name one known `model-dependent` capability exactly once and include a nonempty stable `behavior_id` and its expected behavior. A capability represented by a shared E2E scenario and absent from the exception list MUST retain its baseline-positive check; no omission may skip that check. A missing profile or malformed exception list MUST invalidate certification.
- **FR-004**: Each explicit exception MUST identify the provider limitation and expected adapter behavior closely enough to verify it without weakening the shared baseline. Behavior IDs MUST be semantic and independent of exact registry values: `pass` records a package adaptation that fulfills the public capability, `ignored` records an accepted input with no provider effect, `rejected_before_transport` records local rejection, and `none_falls_back_to_minimum` derives its exact minimum from `reasoning_capability`. A provider-specific behavior ID is permitted only when those shared outcomes plus structured model metadata cannot express the behavior. Behavior IDs MUST NOT name pytest tests or duplicate exact values owned by structured registry fields.
- **FR-005**: For each capability represented by an existing shared E2E scenario, selection MUST choose its baseline-positive check when there is no exception or when `behavior_id: "pass"` applies. Where an existing distinct common or package-specific E2E check covers a non-`pass` `(capability_id, behavior_id)` pair, selection MUST use that check. A request exercising multiple capability IDs MUST NOT run when any marked capability has a non-`pass` exception that redirects to another scenario. Selection MUST NOT infer a route from prose in `behavior`. The canonical capability catalogue MAY include more detailed IDs than the existing E2E scenario inventory; IDs without an existing matching request MUST NOT be claimed as shared E2E evidence.
- **FR-006**: A `pass` exception within the existing shared E2E scope MUST retain the baseline-positive check and MUST NOT require a model-specific E2E route. Package unit or integration tests MUST verify the adaptation. An existing package-specific non-`pass` exception check MAY replace the positive check for the same model and capability only. Neither kind may remove unrelated common checks; missing replacement routes within the shared E2E scope MUST be reported as a coverage gap. The 0.9.8 test refactor MUST reuse existing common and package E2E functions and request forms rather than create new scenarios. `tool_choice_*` and `provider_continuation` do not create separate shared E2E requests; the standard application-tool loop runs once per model using a viable tool-choice mode.
- **FR-007**: Selection of a provider's CI line and access to that provider's credentials MUST remain separate from model-capability-based scenario selection. Pull-request checks MUST remain deterministic and credential-free.
- **FR-008**: The model registry MUST allow optional, independently verified `cache_read_input_per_1m` and `cache_write_input_per_1m` prices for an exact model only when the component can occur automatically during an ordinary adapter request and the provider reports its quantity. A missing or unverified price MUST NOT be inferred from ordinary input pricing or from the other cache component. The registry MUST NOT record a cache mode flag or rates for unsupported opt-in cache control, TTL, cache resources, or storage.
- **FR-009**: Usage and cost reporting MUST distinguish provider-confirmed automatic cache reads, separately reported automatic cache writes, and ordinary input; normalize provider-specific fields into a common inclusive input total only when official response semantics make the sum exact; avoid double counting; and MUST NOT infer cache activity or quantity from request settings or incomplete provider usage. The existing public `cached_tokens` meaning remains cache-read/cache-hit usage for compatibility; optional cache-write usage is separate.
- **FR-010**: A complete cost total MUST be reported only when all incurred components needed for that total are confirmed and priceable. In provider-parsed partial usage, an omitted input or output token count MUST remain `None`, distinct from an explicitly reported `0`; the existing `Usage` constructor order and defaults MUST remain valid. Existing tiered and other non-static pricing semantics MUST remain correct.
- **FR-011**: A deterministic consistency check MUST cover external organization identities and distribution names across Core organization discovery, optional installation declarations, E2E profiles, and CI line selection; missing or conflicting entries MUST fail with an actionable result.
- **FR-012**: Safe metadata duplication MAY be removed, but the existing public facade, provider discovery behavior, optional installation experience, provider contracts, independent package release jobs, and credential boundaries MUST remain backward compatible. The documented `None` correction for omitted counts in provider-parsed partial usage requires migration guidance for callers that perform arithmetic on token counts.
- **FR-013**: The refactor MUST pass applicable shared conformance checks and backward-compatibility checks for built-in and external organizations without requiring changes to existing callers.
- **FR-014**: Core 0.9.8 MUST NOT move xAI or another external organization into Core, add a public cache-control capability, change provider contracts, or begin the service-provider/deployment-profile layer.

### Key Entities

- **Canonical baseline capability**: An externally observable behavior defined by the project constitution and represented by the version-controlled capability catalogue.
- **Model exception profile**: A first-party model's explicit list of verified model/provider limitations and their adapter behavior; `behavior_id: "pass"` records a compensated limitation, and IDs with distinct shared E2E scenarios route to deviation evidence. Absence of an exception retains any applicable shared baseline-positive check.
- **Scenario evidence**: A positive common check or, where a distinct shared E2E route exists, a check of an explicit rejection or special behavior; package-specific evidence is additional where required.
- **Automatic cache-read usage**: The portion of input tokens explicitly reported by a provider as read from provider-managed cache during an ordinary adapter request, exposed through the backward-compatible `cached_tokens` meaning with no locally assumed hit.
- **Automatic cache-write usage**: A separately reported portion of input tokens automatically written to provider-managed cache during an ordinary adapter request; it is represented only when independently metered.
- **Automatic cache component price**: An optional exact-model verified rate or pricing rule applicable to confirmed cache-read or separately priced cache-write input. Unsupported opt-in cache modes and storage are absent.
- **External organization identity**: The organization key and its separately installable distribution name, consistently represented across discovery, installation, E2E, and CI metadata.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every first-party model included in the Core 0.9.8 and six external organization catalogues has an explicit, valid exception list; removing that list or declaring an unknown/duplicate exception or a missing/malformed `behavior_id` causes validation to fail. A legacy third-party plugin without a profile remains usable at runtime but cannot pass profile-based certification.
- **SC-002**: For every tested model, 100% of applicable capabilities with shared E2E scenarios select their baseline-positive scenario when no exception or a `pass` exception applies. Every other declared `(capability_id, behavior_id)` pair requiring distinct E2E evidence selects a check of its documented outcome. The common application-tool loop runs once per model.
- **SC-003**: A distinct non-`pass` exception check replaces only the positive check for the same model and capability; all other applicable common scenarios remain selected. Missing or duplicate evidence within the shared E2E scope fails deterministic validation.
- **SC-004**: Deterministic accounting examples with complete automatic cache-read/cache-write usage produce the expected quantities and cost without double counting. In partial provider-parsed usage, omitted input or output counts are `None` in the public `Usage` value, while explicit reported zero remains `0`; neither partial nor malformed usage produces fabricated cache activity or a complete total. Opt-in-only cache pricing is absent. Direct `Usage` construction and the existing `cached_tokens` cache-read meaning remain compatible.
- **SC-005**: Every supported external organization has matching identity and distribution name across the four existing metadata sources; a single missing entry or name mismatch is detected by a deterministic check.
- **SC-006**: Existing caller-facing conformance and compatibility scenarios pass for all affected built-in and external organizations, including existing installation errors and both supported synchronous transport choices.
- **SC-007**: Provider credentials remain absent from pull-request checks, and each provider's separate release and E2E line remains available after the refactor.

## Assumptions

- The 0.9.8 scope comes from the [LLM API Adapter Implementation Plan](https://app.notion.com/p/34f33dd99fc8812ea5f2eae262910ab3), specifically “Следующий этап — 0.9.8: Core / Plugin Architecture Refactor.”
- The constitution defines the provider-neutral baseline and capability scopes; this feature records model decisions against it without creating another contract.
- A cache-read or cache-write rate is added only when verified for the exact model and pricing context, the component can occur automatically during an ordinary adapter request, and the provider reports enough usage to account for it. No provider-wide default, cache activity estimate, opt-in cache mode, storage charge, or user-facing cache-control behavior is assumed.
- The current organization inventory and existing public behavior are the starting compatibility target; metadata cleanup is permitted only where the same behavior and release isolation are preserved.
- Provider-specific live checks remain subject to the project's established authorization and post-publish release gates. The specification does not authorize a live run or publication.
