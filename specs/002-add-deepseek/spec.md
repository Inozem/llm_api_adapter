# Feature Specification: DeepSeek Provider Release

**Feature Branch**: `feat/deepseek`

**Created**: 2026-09-16

**Status**: Draft

**Input**: User description: "Plan the Core 0.9.6 and llm-api-adapter-deepseek 0.1.0 release from the Implementation Plan."

## Clarifications

### Session 2026-09-16

- Q: Which verified DeepSeek model set must 0.1.0 support? → A: Models with the greatest common-Core compatibility, selected through official verification.
- Q: What minimum functionality is required for a selected DeepSeek model before the 0.1.0 release? → A: Select the model with the maximum officially verified functionality; unsupported functions remain explicit.

### Session 2026-09-17

- Q: How is the final paid DeepSeek check authorized and run? → A: Before promoting the reviewed candidate to protected `dev`, verify only that `DEEPSEEK_API_KEY` is present in the maintainer environment, present the exact test command, and let the maintainer run the bounded local E2E check. Promote only after its sanitized result passes. The subsequent `dev` merge independently triggers the post-publish TestPyPI workflow and its bounded DeepSeek E2E lane automatically. Both checks read the key from their environment; no command, log, or result may reveal the key.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Use DeepSeek through the common facade (Priority: P1)

An application developer installs the DeepSeek organization add-on, selects `deepseek` through the
existing SDK entry point, and uses a verified DeepSeek model for a supported conversation without
writing DeepSeek-specific application code.

**Why this priority**: A usable DeepSeek integration is the release's central user outcome. The
common facade is what lets callers add the organization without altering their application design.

**Independent Test**: Install the released Core and DeepSeek distributions into an otherwise clean
environment, make a representative supported request through the common facade with provider
responses simulated, and verify a normalized result.

**Acceptance Scenarios**:

1. **Given** the DeepSeek add-on is installed and a verified supported model is selected, **When**
   a developer submits a valid conversation through the common facade, **Then** the developer
   receives the same normalized response concepts available for other supported organizations.
2. **Given** a declared supported DeepSeek model, **When** the developer requests a completed
   response, an asynchronous response, or a supported streaming response, **Then** the SDK
   applies the documented capability boundary for that model and mode.
3. **Given** an existing application that uses another organization through the common facade,
   **When** it adds a DeepSeek selection, **Then** it does not need to change its public SDK
   constructor or request shape solely to select DeepSeek.

---

### User Story 2 - Install and diagnose DeepSeek support clearly (Priority: P1)

An application developer can discover the DeepSeek add-on, install it with the documented Core
option, and receive a specific remedy if DeepSeek is selected before its add-on is installed.

**Why this priority**: Separately versioned support is valuable only if missing optional support is
distinguishable from an unknown organization and straightforward to enable.

**Independent Test**: In clean environments with and without the DeepSeek add-on, select
`deepseek` through the common facade and verify the documented installation path or result.

**Acceptance Scenarios**:

1. **Given** a developer installs Core with its documented DeepSeek option, **When** the developer
   selects `deepseek`, **Then** the organization is available through normal discovery.
2. **Given** DeepSeek is known to the installed Core but its add-on is absent, **When** the
   developer selects it, **Then** the SDK reports that the optional add-on is missing and includes
   actionable installation guidance.
3. **Given** a developer selects an organization not supported by the SDK, **When** construction
   is attempted, **Then** the SDK reports an unknown organization rather than advising the
   developer to install DeepSeek.

---

### User Story 3 - Select a capability-safe DeepSeek model (Priority: P2)

An application developer can identify which DeepSeek models and request capabilities have been
verified, use the supported combinations, and receive a clear error before an unsupported request
is sent.

**Why this priority**: DeepSeek capability differences must not be hidden behind an apparent
provider-neutral promise. Explicit boundaries let the developer choose a compatible model before
incurring provider cost or handling ambiguous output.

**Independent Test**: Exercise every declared model/capability combination and representative
unsupported combinations with simulated provider behavior; verify that support is honored and
unsupported combinations are clearly rejected.

**Acceptance Scenarios**:

1. **Given** a verified model and a capability declared for that model, **When** a developer makes
   the corresponding valid request, **Then** the request completes with the common normalized
   response, error, usage, and terminal-state semantics.
2. **Given** a model, endpoint, or request capability outside the published DeepSeek compatibility
   matrix, **When** a developer requests it, **Then** the SDK rejects it clearly before an
   unsupported provider request is made.
3. **Given** a feature such as tool use, structured output, reasoning, image input, continuation,
   or streaming is not declared for a selected model, **When** it is requested, **Then** the SDK
   does not silently substitute a different behavior or model.

---

### User Story 4 - Understand document and cost boundaries (Priority: P2)

An application developer can see that direct PDF/document input is not part of the initial
DeepSeek support and can rely on provider-reported usage and standard cost information only where
it is available for the selected model.

**Why this priority**: Clear limits prevent users from assuming that all common SDK file and cost
features apply to every organization.

**Independent Test**: Submit supported and unsupported file inputs and simulated responses with
complete or missing usage; verify documented errors and availability of cost data.

**Acceptance Scenarios**:

1. **Given** a developer provides a PDF or other document form not declared as supported for
   DeepSeek, **When** the request is submitted, **Then** it fails with an explicit compatibility
   error before the provider is contacted.
2. **Given** provider-reported usage can be priced by the published model metadata, **When** a
   response completes, **Then** the normalized result presents the resulting standard estimate.
3. **Given** usage or a price cannot be verified, **When** a response completes, **Then** the
   corresponding cost value remains unavailable rather than being estimated.

### Edge Cases

- The DeepSeek add-on is not installed, while an unrelated organization is selected or the caller
  uses an unrecognized organization name.
- A direct model identifier, alias, or endpoint is not in the verified compatibility matrix.
- A selected model supports chat but not one of streaming, tools, structured output, reasoning,
  image input, or continuation.
- The provider returns an incomplete result, refusal, malformed response, missing usage, an
  authentication failure, a rate limit, or an unavailable service.
- A stream is cancelled, closed by the caller, or fails before completion; incomplete text must
  not be presented as a successful final result.
- A caller supplies a PDF, another document form, or an image form outside the explicit initial
  DeepSeek support boundary.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The release MUST ship Core version `0.9.6` and the independently versioned
  `llm-api-adapter-deepseek` version `0.1.0` as one compatible release train.
- **FR-002**: Core `0.9.6` MUST make the optional DeepSeek integration discoverable through its
  documented installation surface without adding DeepSeek to the base installation footprint.
- **FR-003**: The DeepSeek distribution MUST be installable independently and state the compatible
  Core version range.
- **FR-004**: A developer MUST select DeepSeek through the existing public organization selector;
  the release MUST preserve the public facade and the construction contract for existing callers.
- **FR-005**: Selecting known-but-uninstalled DeepSeek support MUST produce actionable
  installation guidance, while selecting an unknown organization MUST produce a distinct error.
- **FR-006**: Official verification MUST select the DeepSeek models with the greatest compatible
  common-Core capability coverage for the initial release. The release MUST publish an explicit
  compatibility matrix that identifies every selected model and, for each model, its supported
  request modes and capabilities. The initial set MUST include the officially verified DeepSeek
  model with the greatest supported functional coverage; capabilities it lacks MUST remain
  explicitly unsupported.
- **FR-007**: Every matrix-declared DeepSeek combination MUST provide the common normalized
  semantics for messages, completed responses, errors, usage, terminal states, and any capability
  it declares.
- **FR-008**: The release MUST provide supported synchronous, asynchronous, and streaming chat
  behavior for every DeepSeek model/mode combination declared in the compatibility matrix.
- **FR-009**: Tools, structured output, reasoning, image input, continuation, and pricing MUST be
  exposed only for the DeepSeek model combinations that explicitly declare support; an
  undeclared capability MUST not be silently emulated, weakened, or routed to another model.
- **FR-010**: The initial DeepSeek release MUST reject direct PDF and other unsupported document
  inputs clearly before provider submission. It MUST NOT add client-side document processing,
  automatic fallback, or an undocumented compatibility route.
- **FR-011**: The release MUST use verified provider-reported usage and published model metadata
  for standard cost estimates. Where either required value is unavailable, the relevant usage or
  cost value MUST remain unavailable.
- **FR-012**: DeepSeek support MUST use only the official DeepSeek API and MUST not introduce a
  deployment backend, self-hosted runtime, retry policy, or unrestricted endpoint compatibility
  feature.
- **FR-013**: Each DeepSeek model/capability combination declared as supported MUST pass the
  shared conformance evidence, deterministic behavior checks for both supported synchronous
  transport choices, and the focused authorized live verification required for release. A merge
  into protected `dev` MUST publish the changed candidates to TestPyPI and run the affected
  bounded E2E lane using `DEEPSEEK_API_KEY` from GitHub Actions Secrets. Before that promotion,
  tooling MUST check only whether the variable is present in the maintainer environment, then
  provide the precise E2E command for maintainer invocation; the test MUST consume the key from
  the environment and MUST NOT display it.
- **FR-014**: The release documentation MUST state installation, verified model support,
  capability limitations, direct-document rejection, and the boundary between standard estimates
  and provider invoices.

### Key Entities

- **DeepSeek organization add-on**: The separately installable, independently versioned SDK
  distribution that supplies official DeepSeek support to a compatible Core release.
- **DeepSeek model capability record**: The published, verified declaration of a DeepSeek model's
  supported request modes, features, limits, and standard cost information.
- **Organization selection**: A caller's choice of `deepseek` through the existing common facade,
  which either resolves to the installed add-on or produces a clear selection error.
- **Normalized DeepSeek response**: The common SDK result for a supported DeepSeek request,
  including visible content and applicable structured data, tools, usage, cost, reasoning, or
  terminal state.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a clean installation, 100% of documented DeepSeek installation paths either
  enable the organization successfully or provide the prescribed actionable installation remedy.
- **SC-002**: 100% of model/mode combinations published as supported pass the shared conformance
  evidence and deterministic transport-parity checks before release.
- **SC-003**: For every tested unsupported DeepSeek capability or document form, the SDK returns a
  clear compatibility result before making a provider request in 100% of contract-test cases.
- **SC-004**: Across all declared DeepSeek chat modes, 100% of completed simulated outcomes expose
  the common normalized response or documented error/terminal state rather than raw
  provider-shaped data.
- **SC-005**: For 100% of test responses without verifiable usage or pricing, the relevant cost
  field is unavailable; no estimated value is reported.
- **SC-006**: Existing callers of the public facade complete their pre-existing supported
  organization flows without changes attributable to the Core `0.9.6` release.

## Assumptions

- The current Implementation Plan's order—DeepSeek after the completed Kimi release—supersedes
  the older deferred status in the separate DeepSeek implementation-plan page. That page remains
  useful for its scope boundaries until it is updated.
- The initial DeepSeek scope uses only the official direct service. Managed deployment profiles,
  self-hosted inference, automatic retries, fallback routing, and unrestricted compatible
  endpoints are outside this release.
- Direct PDF/document support is out of scope for `llm-api-adapter-deepseek` `0.1.0`; unsupported
  document inputs are rejected rather than processed locally. A later release may change this
  only through an explicit product decision.
- The initial verified model list and any model-specific capability declarations are determined by
  official-API research before release. Selection prioritizes the models with the greatest
  compatible common-Core capability coverage; no model is assumed supported merely because its
  name is accepted by the provider.
- Core `0.9.6` and DeepSeek `0.1.0` retain independent versions, tags, publication, and patch
  cadence after their coordinated first release.
