# Feature Specification: Z.ai / GLM Provider

**Feature Branch**: `003-zai-glm-provider`

**Created**: 2026-09-18

**Status**: Draft

**Input**: User description: "Add Z.ai / GLM as an independently versioned provider package, released as `llm-api-adapter-zai` 0.1.0 alongside core `llm-api-adapter` 0.9.7."

## Clarifications

### Session 2026-09-18

- Q: Which initial GLM model set should the 0.1.0 release publicly support? → A: Include every GLM model that is maximally compatible with the current core contract; if several qualify, include several.
- Q: What level of core compatibility is required for an initial-release GLM model? → A: The Z.ai package remains an external extra. Select the models with the greatest number of confirmed core-baseline capabilities; record every unsupported capability explicitly in the compatibility matrix.
- Q: Must a model have officially confirmed pricing and currency to enter the 0.1.0 release? → A: Yes. Include only models with officially confirmed pricing and currency.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Use a verified GLM model (Priority: P1)

A Python application developer installs the Z.ai extension and selects a documented GLM model through the existing public adapter interface, so that the application can use Z.ai without changing its provider-neutral calling pattern.

**Why this priority**: This is the primary value of the release: one supported Z.ai / GLM integration that behaves consistently with the existing adapter contract.

**Independent Test**: With only the Z.ai extension installed, a developer can configure valid Z.ai credentials, select a model included in the published compatibility matrix, and receive a normalized successful response through the public adapter interface.

**Acceptance Scenarios**:

1. **Given** the Z.ai extension and valid credentials are available, **When** a developer selects a documented supported GLM model, **Then** the request completes through the existing public adapter interface and returns the standard response shape.
2. **Given** the Z.ai organization is selected but its extension is not installed, **When** a developer creates the adapter, **Then** they receive actionable installation guidance rather than an unknown-provider error.

---

### User Story 2 - Reliably use declared capabilities (Priority: P2)

A developer can rely on the published Z.ai compatibility information to determine which request features are supported by each initial GLM model and receives a clear outcome for unsupported combinations.

**Why this priority**: A provider is useful only when its supported models and capabilities are explicit and do not over-promise portability.

**Independent Test**: Each declared capability can be exercised against deterministic provider fixtures, while an unsupported capability is rejected before a provider request or reported through the documented normalized behavior.

**Acceptance Scenarios**:

1. **Given** a documented supported model and capability combination, **When** the developer makes the corresponding request, **Then** its result follows the common adapter contract.
2. **Given** an unsupported model or capability combination, **When** the developer makes the request, **Then** the outcome clearly identifies that the combination is unsupported and does not claim success.

---

### User Story 3 - Upgrade and release independently (Priority: P3)

A maintainer can publish and validate the Z.ai integration independently while releasing core version 0.9.7 as the coordinated compatibility milestone.

**Why this priority**: Independent provider releases are a core product promise and keep provider-specific changes isolated from the shared library.

**Independent Test**: The provider distribution can be installed in isolation with the stated compatible core release and can pass its published release-validation gate.

**Acceptance Scenarios**:

1. **Given** an environment containing the compatible core release, **When** the Z.ai provider distribution is installed, **Then** the public Z.ai integration is discoverable without requiring other provider distributions.
2. **Given** a candidate Z.ai provider release, **When** the release-validation suite is run, **Then** deterministic contract evidence and the selected authorized live verification determine whether it is eligible for publication.

### Edge Cases

- Z.ai credentials are missing, invalid, or lack access to the selected model.
- A caller uses an unknown, retired, or unsupported GLM model identifier.
- A caller requests a feature that the selected model or endpoint does not support.
- The provider reports malformed, incomplete, or unavailable usage data; the library must not fabricate usage or cost.
- A provider request fails, times out, or streaming ends before a complete response; callers receive the project's normalized error or incomplete-result behavior.
- The optional Z.ai distribution is absent while the organization is selected.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The product MUST provide Z.ai / GLM as a separately installable, independently versioned organization integration named `llm-api-adapter-zai` with an initial `0.1.0` release.
- **FR-002**: Core `llm-api-adapter` version `0.9.7` MUST identify the Z.ai integration as known and provide actionable installation guidance when it is selected without its optional distribution.
- **FR-003**: The Z.ai integration MUST use only Z.ai's official API and MUST preserve the existing public organization-selection interface.
- **FR-004**: The release MUST publish an explicit initial compatibility matrix that identifies each GLM model maximally compatible with the current core contract, including every model that meets that criterion, and records its supported request capabilities, limits, and applicable pricing information.
- **FR-005**: The Z.ai integration MUST remain an external extra. It MUST select the GLM models with the greatest number of confirmed current core-baseline capabilities and support only their verified model-and-capability combinations within each model's documented capability boundary; every unsupported capability MUST be explicit rather than inferred from model naming or provider similarity.
- **FR-005a**: A GLM model MUST NOT be included in the initial release unless its price and currency are confirmed by official Z.ai documentation.
- **FR-006**: Successful Z.ai requests MUST produce the shared normalized response, streaming, asynchronous-operation, usage, pricing, and error behavior for every capability declared as supported.
- **FR-007**: The integration MUST preserve the documented portable structured-output boundary and MUST not silently weaken a caller's schema or request semantics.
- **FR-008**: The release MUST provide deterministic, credential-free evidence for its declared behavior across the supported synchronous, asynchronous, and streaming paths.
- **FR-009**: The release MUST pass the common conformance suite, deterministic transport-parity checks, and a focused live verification using maintainer-controlled credentials before it is announced as supported.
- **FR-010**: The initial release MUST exclude deployment backends, arbitrary user-supplied endpoints, and models or capabilities not present in the published compatibility matrix.

### Key Entities

- **Z.ai provider package**: The independently installable organization integration and its own release lifecycle.
- **GLM model profile**: A verified model entry that defines the identifier, supported capabilities, limits, and pricing applicable to one initial-release model.
- **Compatibility matrix**: The user-facing record of supported Z.ai model-and-capability combinations and their explicit exclusions.
- **Provider release gate**: The required deterministic and authorized live evidence that permits the package to be announced as supported.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of selected GLM models appear in the compatibility matrix with an explicit status for every core-baseline capability, verified limits, and officially confirmed pricing and currency.
- **SC-002**: 100% of declared supported Z.ai scenarios pass the common conformance suite and deterministic synchronous, asynchronous, streaming, and transport-parity checks.
- **SC-003**: A clean environment can install the Z.ai package with core `0.9.7`, discover it, and complete every declared deterministic scenario without requiring another provider package.
- **SC-004**: 100% of unsupported model or capability combinations covered by release tests fail clearly without a provider request being presented as successful.
- **SC-005**: The focused authorized live verification passes for every initial-release capability marked live-verifiable before the provider is publicly announced as supported.

## Assumptions

- Z.ai / GLM remains an external organization package under the existing plugin criteria; no new core-provider exception has been approved.
- The initial release includes the officially documented and verified GLM models with the greatest number of confirmed current core-baseline capabilities. The Z.ai package remains an external extra, and any capability a selected model does not support remains explicit in the matrix.
- A model without officially confirmed Z.ai pricing and currency is excluded from the initial release.
- The baseline contract in `specs/001-baseline-contract` governs Z.ai. A selected model need not support every core capability, but its declared profile must cover every applicable shared conformance and Core E2E scenario; only explicitly unsupported scenarios may be excluded.
- The exact initial model list, endpoint details, authentication method, capability statuses, limits, and pricing will be established from official Z.ai documentation during planning.
- The existing public facade, transport policy, provider registry, compatibility-matrix convention, and common conformance suite remain the governing product constraints.
- Paid live verification is performed only with maintainer-controlled credentials in an authorized release or post-publish lane.
