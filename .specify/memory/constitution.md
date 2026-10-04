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
and error conformance baseline; incomplete support remains a plugin package. Full baseline
conformance is necessary but does not require moving an organization into Core: a conformant
organization MAY remain an independently versioned package for modular installation and release.
Package placement MUST NOT be treated as evidence of missing capabilities; the exact-model
capability profile and its tests define supported behavior and exceptions.

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

Dedicated structured registry fields MUST be the sole source of exact model values, limits, and
supported-value sets. Capability exceptions MUST record only the observable class of a baseline
deviation or behavior that existing structured metadata cannot express; they MUST NOT duplicate
exact values or become an alternative source of truth. Adapters and tests MUST derive the concrete
result from the owning structured field. The same observable behavior MUST reuse the same semantic,
value-independent `behavior_id` across models and organizations. A custom `behavior_id` is
permitted only when neither an existing shared behavior nor structured metadata can express the
result. Any contradiction between structured metadata and an exception MUST fail deterministic
validation.

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

This constitution is the sole normative source for the SDK's stable, externally observable
provider-neutral baseline. Every new optional organization package MUST declare a capability
profile against that baseline and run every applicable shared conformance and Core E2E scenario.
A scenario may be excluded only when the exact-model profile explicitly declares the corresponding
capability exception; missing implementation, flaky behavior, or cost is not a valid exclusion. A
model or organization is not required to support every baseline capability, but every declared
difference MUST be explicit in registry metadata, compatibility documentation, and tests.

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
than be locally estimated, and non-token metered operations MUST be represented separately from
token cost. `previous_response` remains an optional provider optimization: unsupported adapters
accept it without serializing an unsupported provider request and use the caller-supplied history.

## Canonical Provider-Neutral Baseline

This section defines the current stable SDK contract. Feature specifications, provider-package
plans, compatibility matrices, implementation, tests, and release gates MUST conform to it and
MUST NOT establish a competing baseline. A demonstrated conflict requires a reviewed constitution
amendment together with the necessary implementation, registry, test, migration, and documentation
work before the corrected behavior may be used for a release decision.

### Public facade and distribution boundary

- `UniversalLLMAPIAdapter` MUST select an adapter from the caller's organization, exact model,
  API key, optional service provider, and documented transport without changing the common API.
- OpenAI, Anthropic, and Google remain built into Core. Mistral, xAI, Qwen, Kimi, DeepSeek, and
  Z.ai remain independently installable organization packages discovered through the established
  entry point. A known but absent optional package, an unknown organization, and an unsupported
  service provider MUST produce distinct actionable errors.
- The base installation MUST retain its minimal dependency boundary. HTTPX, async support, and
  organization packages MUST remain optional installations unless an approved breaking change
  explicitly revises that boundary.

### Requests, responses, and streaming

- Typed messages and supported OpenAI-style dictionaries, including mixed input, MUST normalize
  system, user, assistant, and tool-result turns without losing their roles or supported content.
- The common API MUST expose synchronous chat, asynchronous chat, synchronous visible-text
  streaming, and asynchronous visible-text streaming with the same applicable request and response
  concepts. Completed responses MAY include visible content, normalized tool calls, provider
  response identity, refusal or incomplete state, provider-reported usage, cost, parsed structured
  output, and opt-in reasoning events.
- Streams MUST yield normalized visible text only. Buffer limits MUST be positive and MUST NOT be
  exceeded. Reasoning and raw provider events MUST remain separate from visible text. Successful
  streams MUST flush pending visible text before completion and deliver completed tool calls before
  the final callback; failed, cancelled, or caller-closed streams MUST NOT emit pending text as a
  successful final chunk or invoke final completion.
- Sync and async streaming callbacks MAY be synchronous or awaitable as documented. Callback
  failures remain caller failures and MUST NOT be reclassified as provider failures.

### Tools, structured output, files, and reasoning

- Tool definitions and supported automatic, disabled, any-tool, and named-tool choices MUST be
  validated and normalized before transport. A named tool MUST exist in the supplied tool list.
  Completed calls MUST expose normalized names, parsed arguments, and provider call identifiers
  where available; the SDK MUST NOT execute application tools.
- Portable JSON Schema and compatible response-model requests MUST be validated before transport.
  Valid completed JSON MUST populate parsed output, response models MUST additionally validate the
  typed result, and incompatible schemas, combinations, or completed output MUST raise the public
  schema error. Refusal and incomplete outcomes MUST leave parsed fields unavailable.
- The common message model MUST accept supported image and PDF forms. Every unsupported URL, byte,
  document, or non-image file form MUST be declared by the exact-model profile and rejected before
  an invalid provider request. A verified package adaptation such as OCR MAY satisfy the public
  contract and remains visible as a `pass` exception.
- Reasoning capture MUST be opt-in and separate from visible text. When requested reasoning data is
  unavailable, the response MUST expose no synthesized reasoning. Reasoning controls and their
  fallback or ignored behavior MUST come from exact-model registry metadata.

### Registry, continuation, usage, pricing, and errors

- The organization registry MUST own verified model limits, standard token rates, reasoning
  capability, request rules, aliases, snapshot inheritance, and capability exceptions. Only
  documented direct Anthropic and OpenAI snapshot forms MAY inherit registered base metadata while
  retaining their requested wire model identifiers. Other aliases, fine-tuned identifiers, and
  unknown models MUST receive no inferred special behavior.
- Request rules MUST either reject an unsupported request before transport or apply their declared
  transformation and warning behavior. Removing a declared default value MUST remain silent.
- `previous_response` MUST be accepted as a common compatibility input. A provider continuation
  identifier or opaque replay material MAY be sent only when the exact model and adapter declare
  that behavior; otherwise the adapter MUST use caller-provided history and serialize no unsupported
  continuation field.
- Usage and costs MUST use provider-reported data and verified rates. Missing or contradictory
  usage MUST NOT be estimated or treated as zero. Reported cache reads and writes MUST be confirmed
  input components, remain distinct, and never be double-counted. Parsed omitted input/output counts
  MUST remain `None`; an explicitly reported zero remains `0`. Cache rates apply only to components
  that may occur automatically during ordinary requests and whose quantities providers report.
  A calculated total MUST remain unavailable whenever an incurred component cannot be priced
  completely. A complete provider-reported total MAY remain available without a component
  breakdown; the SDK MUST NOT invent missing component costs. Non-token metered operations MUST
  remain separate from token cost.
- Known authorization, rate-limit, token-limit, client, server, timeout, usage-limit, tool-input,
  tool-argument, tool-choice, structured-output, and configuration failures MUST map to the public
  error hierarchy. Common validation failures and declared unsupported inputs MUST fail before an
  outbound provider request.

### Capability and E2E evidence catalogue

The version-controlled capability catalogue in
`src/llm_api_adapter/llm_registry/model_capabilities.py` classifies the following exact-model
capabilities as `model-dependent`: `sync_chat`, `async_chat`, `sync_streaming`,
`async_streaming`, `application_tools`, `tool_choice_auto`, `tool_choice_none`,
`tool_choice_any`, `tool_choice_named`, `structured_output_schema`,
`structured_output_model`, `image_url`, `image_bytes`, `image_data_url`, `pdf_url`,
`pdf_bytes`, `reasoning_control`, `reasoning_events`, `provider_continuation`,
`usage_reporting`, `refusal_outcome`, and `incomplete_outcome`.

The following capabilities are `always-on` Core invariants and MUST NOT be disabled by a model
profile: `facade_discovery`, `message_normalization`, `response_normalization`,
`transport_parity`, `stream_cleanup`, `tool_validation`, `schema_validation`,
`error_normalization`, `registry_exactness`, `request_rule_fidelity`, `pricing_correctness`, and
`missing_usage_honesty`.

`tests/capability_scenarios.py` is the canonical test-only evidence map for the capabilities backed
by current shared scenarios. `BASELINE_SCENARIOS` maps supported model-dependent capabilities to
positive evidence, `EXCEPTION_SCENARIOS` maps non-`pass` behavior to common or package-local
replacement evidence, and `ALWAYS_ON_SCENARIOS` maps unconditional Core invariants. It MUST NOT be
treated as a second product contract or as an inventory of every repository test. Registry metadata
MUST contain semantic capability and behavior IDs, never pytest node IDs.

An absent exception and a `pass` exception MUST retain the baseline-positive route. A non-`pass`
exception MUST replace only the affected route and MUST resolve to exactly one collected evidence
scenario within the shared E2E scope. A request exercising several marked capabilities MUST run
only when none of them redirects to another non-`pass` scenario. Missing profiles, unknown or
duplicate exceptions, unknown behavior routes, missing evidence, duplicate evidence, and attempts
to except an always-on capability MUST fail deterministic validation.

Every optional organization package MUST declare a named Core E2E profile and provider marker.
The provider release lane MUST collect the applicable shared Core scenarios and its package-local
scenarios against the exact candidate distributions. Package-local tests add evidence for native
behavior and explicit rejection boundaries; they MUST NOT remove unrelated shared scenarios or
replace an applicable shared scenario without a declared exact-model exception.

## Change Analysis and Delivery Discipline

Planning MUST begin with this canonical provider-neutral baseline and the actual repository: inspect the
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
or packaging changes MUST update this constitution and the matching README, contributor guide,
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

**Version**: 0.5.1 | **Ratified**: 2026-09-16 | **Last Amended**: 2026-10-04
