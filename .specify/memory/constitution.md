# llm-api-adapter Constitution

## Core Principles

### I. Stable Provider-Neutral Public Contract
`UniversalLLMAPIAdapter`, typed messages, tools, responses, errors, structured output,
streaming, and async APIs form the public contract. Changes MUST preserve backward compatibility
unless the user explicitly approves a breaking change. Before changing a shared interface,
contributors MUST inspect every implementation, caller, compatibility import, and relevant unit,
integration, and conformance test. A breaking change MUST state its migration path and be
reflected in the package version and public documentation.

Rationale: callers select this SDK for a stable interface across organizations and transports.

### II. Shared Contract, Isolated Organization Behavior
Shared core code MUST contain only behavior that belongs to the documented organization-neutral
contract. Wire payloads, endpoints, authentication, provider error details, special document
handling, and model-specific capabilities MUST remain in organization adapters, clients, or
independently versioned organization packages. New core organizations MUST meet the established
sync/async, streaming, message, tool, structured-output, file-input, response, usage, pricing,
and error conformance baseline; incomplete support remains a plugin package.

Rationale: the facade and `LLMAdapterBase` remain predictable while providers can evolve
independently.

### III. Registry and Abstraction First
Verified model limits, pricing tiers, reasoning capabilities, and request exceptions MUST be
declared in the organization registry and executed by existing closed generic rule handlers.
Contributors MUST reuse established adapters, transports, message/response normalizers, stream
helpers, registries, and plugin discovery before creating an abstraction. They MUST NOT add
model-name prefixes, ad-hoc model lists, or one-off provider conditionals when registry metadata
or a generic handler can express the behavior.

Values with shared meaning MUST use existing configuration, constants, mappings, registry data,
or a narrowly scoped new abstraction rather than hardcoded duplicates. Registry data changes MUST
be checked against official organization documentation; unsupported or missing provider data MUST
remain explicit rather than inferred.

Rationale: centralized metadata and reuse prevent drift across adapters and transports.

### IV. Deterministic Contract Evidence and Baseline Profiles
Every behavior or contract change MUST update the smallest relevant deterministic unit or mocked
integration test. Tests must be network-free, credential-free, and reproducible. Changes to
shared request, response, streaming, tools, structured-output, transport, or registry behavior
MUST cover all affected organization implementations and sync/async paths. Real E2E tests are
paid verification: they MUST run only with explicit manual authorization or in the designated
post-publish provider-specific CI lane.

An explicitly authorized manual E2E run before merge is a preflight only; it MUST NOT satisfy a
provider's final release gate. The final gate MUST run after the staging pull request has merged
to `dev`, against exact TestPyPI candidate artifacts in a clean environment, through the
provider-specific post-publish lane, with only that provider's credential. It MUST cover every
applicable shared Core and package-local E2E scenario before the candidate is promoted to `main`.

`specs/001-baseline-contract/spec.md` is the canonical specification of the SDK's stable,
externally observable provider-neutral contract. Every new optional organization package MUST
declare a capability profile against that baseline and run every applicable shared conformance and
Core E2E scenario. A scenario may be excluded only when the package explicitly declares the
corresponding capability unsupported; missing implementation, flaky behavior, or cost is not a
valid exclusion. A model or organization is not required to support every baseline capability,
but every declared difference MUST be explicit in its compatibility documentation and tests.

Rationale: the repository already separates reliable local evidence from bounded live-provider
validation, and capability profiles make permitted provider differences reviewable.

### V. Lightweight, Safe Extensibility
Core MUST preserve the Python 3.10+ `src` layout and minimal runtime dependency footprint.
`requests` remains the default synchronous path; HTTPX and async behavior remain optional extras.
New dependencies require a demonstrated need that cannot be met by existing standard-library or
project facilities, plus updates to packaging, tests, and documentation. External organizations
MUST integrate through the established entry-point plugin and service-provider registry rather
than changing the public facade.

API keys MUST appear only in environment variables, ignored local files, or CI secrets. They MUST
NOT be committed or written to fixtures, examples, documentation, logs, or artifacts. Diagnostics
must treat raw provider events, reasoning content, and tool arguments as potentially sensitive.

Rationale: a small core and safe plugin boundary are explicit project design choices.

## Compatibility and Data Constraints

The shared message and response models normalize organization input and output in both directions.
Organization adapters may apply only documented wire-format transformations; they MUST NOT weaken
the Core portable JSON Schema profile or silently alter a caller's semantic request. Provider
capability limits and compatibility exceptions are exact-model registry data. Unknown models
receive no inferred special behavior.

Usage and cost fields depend on provider-reported values. Missing usage MUST remain unset rather
than be locally estimated. Non-token metered operations MUST be represented separately from token
cost. `previous_response` remains an optional provider optimization: unsupported adapters accept
it without serializing an unsupported provider request and use the caller-supplied history.

## Canonical Baseline Contract

`specs/001-baseline-contract/spec.md` defines the current stable contract that feature
specifications, provider-package plans, compatibility matrices, and release gates MUST use as
their baseline. The implementation, deterministic tests, and this constitution remain the
authoritative evidence for resolving a demonstrated conflict; any resulting baseline correction
MUST be reviewed and recorded before relying on it for a release decision.

New provider packages MUST retain the common facade and declare their supported subset through a
named capability profile. That profile determines the shared conformance and Core E2E scenarios
that apply; it is the sole basis for an exclusion. Provider-specific tests add evidence for native
behavior but MUST NOT replace the applicable shared scenarios.

## Change Analysis and Delivery Discipline

Planning MUST begin with the canonical baseline contract and the actual repository: inspect the
affected source, registry/configuration, public API, adapter implementations, callers, tests, and
documentation. Contributors MUST use existing repository conventions where the evidence is clear
and MUST NOT ask the user questions that can be answered reliably from that evidence.

If analysis exposes a decision that materially changes architecture, public API, backward
compatibility, dependency footprint, provider behavior, or feature scope and the repository does
not determine the answer, contributors MUST pause and request the user's decision before choosing
an approach. They MUST NOT modify production code, migrate or delete documentation, or create a
git commit unless the user explicitly requests that action.

Before review, run the affected deterministic test commands and inspect the diff for unintended
artifacts or sensitive data. User-visible behavior, provider mappings, configuration, test flow,
or packaging changes MUST update the matching README, contributor guide, baseline specification,
and living architecture artifact when those artifacts are in scope. Releases use protected pull
requests, independently versioned distributions, TestPyPI candidates, and bounded
provider-specific E2E lanes. A staging pull request merges into `dev` only after deterministic
review evidence. Its post-publish workflow then publishes the changed candidate artifacts,
installs them cleanly from TestPyPI, verifies plugin discovery, and runs the provider-specific
full E2E lane before promotion to `main`. Candidate artifacts that do not yet exist in TestPyPI
leave this final gate pending; a local build or pre-merge manual E2E does not replace it.

## Governance

This constitution supersedes conflicting repository practices. Amendments MUST identify the
rationale, affected principles, compatibility impact, and necessary test or documentation work.
Reviewers MUST assess compliance before approval. Exceptions require an explicit, time-bounded
rationale in the reviewed change.

Constitution versions follow semantic versioning: MAJOR for incompatible removal or redefinition
of governance, MINOR for a new principle or material guidance expansion, and PATCH for
clarifications that preserve meaning. The constitution version is governance metadata and does
not need to equal the independently released core or organization-package versions. Each
amendment MUST update the temporary Sync Impact Report before review; remove that report before
committing the amended constitution. Compliance is checked during planning, implementation,
review, and release preparation.

**Version**: 0.3.0 | **Ratified**: 2026-09-16 | **Last Amended**: 2026-09-23
