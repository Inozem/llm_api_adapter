# Data Model: Core / Plugin Architecture Refactor (0.9.8)

This document describes the planning-level data contracts. It does not replace the canonical public baseline in `specs/001-baseline-contract/spec.md`.

## Baseline capability catalogue

**Purpose**: Give common scenarios a stable capability identifier without allowing a model profile to disable cross-cutting Core invariants.

| Field | Meaning | Validation |
| --- | --- | --- |
| `id` | Stable identifier derived from one observable baseline behavior or variant | Unique; no undeclared identifier is accepted |
| `scope` | `model-dependent` or `always-on` | Only model-dependent entries may be listed as exceptions; always-on scenarios are unconditional |
| `positive_scenarios` | Applicable common scenarios for confirmed support | At least one scenario when the capability has a shared E2E check; detailed registry-only IDs need no separate shared E2E route |
| `exception_scenarios` | Common or package-local checks keyed by `(capability_id, behavior_id)` where distinct E2E evidence is required | A non-`pass` pair needs one replacement route in this E2E scope; a `pass` pair uses the positive scenario without another route |

The canonical capability catalogue records text/chat modes, synchronous and asynchronous streaming, application tools and `tool_choice` variants, portable structured output, supported image/document forms, reasoning controls and events, continuation, usage availability, refusal, and incomplete outcomes. The shared E2E scenario inventory is narrower: it reuses existing common test functions and their request forms. The tool loop selects a viable `tool_choice` mode for each model; it does not separately exercise `tool_choice_auto`, `tool_choice_none`, `tool_choice_any`, `tool_choice_named`, or provider-side continuation. Image, document, structured-output, and async variants enter shared E2E selection only when an existing test request actually exercises them; no new scenario is created to match a registry ID. Facade/discovery, message and error normalization, transport parity/cleanup, request-rule fidelity, pricing correctness, and missing-usage honesty are always-on checks. The catalogue is derived from the existing baseline and shared scenario inventory, then version-controlled and tested.

The ID trace below is planning documentation for the canonical baseline. Runtime metadata and executable tests use stable capability IDs and scopes; they do not read specification files or carry requirement numbers.

| Scope | Capability IDs | Baseline requirements |
| --- | --- | --- |
| Model-dependent | `sync_chat`, `async_chat` | FR-004 |
| Model-dependent | `sync_streaming`, `async_streaming` | FR-004, FR-006, FR-017 |
| Model-dependent | `application_tools`, `tool_choice_auto`, `tool_choice_none`, `tool_choice_any`, `tool_choice_named` | FR-007 |
| Model-dependent | `structured_output_schema`, `structured_output_model` | FR-008 |
| Model-dependent | `image_url`, `image_bytes`, `image_data_url`, `pdf_url`, `pdf_bytes` | FR-009 |
| Model-dependent | `reasoning_control` | FR-010, FR-018 |
| Model-dependent | `reasoning_events` | FR-018 |
| Model-dependent | `provider_continuation` | FR-014 |
| Model-dependent | `usage_reporting` | FR-005, FR-015 |
| Model-dependent | `refusal_outcome`, `incomplete_outcome` | FR-005 |
| Always-on | `facade_discovery` | FR-001, FR-002 |
| Always-on | `message_normalization` | FR-003 |
| Always-on | `response_normalization` | FR-005 |
| Always-on | `transport_parity` | FR-004 |
| Always-on | `stream_cleanup` | FR-006, FR-017 |
| Always-on | `tool_validation` | FR-007 |
| Always-on | `schema_validation` | FR-008 |
| Always-on | `error_normalization` | FR-012, FR-016 |
| Always-on | `registry_exactness` | FR-010, FR-011 |
| Always-on | `request_rule_fidelity` | FR-019 |
| Always-on | `pricing_correctness` | FR-010, FR-015 |
| Always-on | `missing_usage_honesty` | FR-015 |

FR-013 (base dependency boundary), FR-020 (optional-package capability documentation), and FR-021 (optional-package E2E profile and evidence) are baseline obligations outside exact-model scenario selection. They remain in installation, package, and release validation; a model profile cannot turn them off.

## Model exception profile

**Owner**: Exact model entry in a built-in or external organization's registry metadata.

| Field | Meaning | Validation |
| --- | --- | --- |
| `model_id` | Existing exact registered model identifier | Existing registry uniqueness and alias rules apply |
| `capability_exceptions` | Explicit model/provider limitations, including those compensated by the package | Required for each first-party model; may be empty; IDs must be known, unique, and model-dependent |
| `capability_id` | Canonical capability ID whose baseline behavior differs | Must match one catalogue entry; always-on IDs cannot be excepted |
| `behavior_id` | Stable, value-independent code for the adapter's outcome | Required on every declared exception; lowercase ASCII identifier matching `^[a-z][a-z0-9_]*$` |
| `behavior` | Verified expected rejection or provider-specific normalized behavior | Required for every exception; must not duplicate exact values owned by structured model metadata |

Capability IDs already identify meaningful variants separately (for example, `image_url` and `image_bytes`). An exception therefore applies to exactly its listed ID and has no nested variant overrides; this prevents a profile entry from silently broadening to neighboring capabilities.

The shared behavior vocabulary is `pass` for a compensated limitation, `ignored` for an accepted input with no provider effect, `rejected_before_transport` for a locally rejected input, and `none_falls_back_to_minimum` when reasoning cannot be disabled. The exact minimum for the latter comes from `reasoning_capability.allowed_values` or `reasoning_capability.min_budget_tokens`. A provider-specific behavior ID is allowed only when the shared vocabulary plus the capability's structured metadata cannot express the observable behavior, such as an effort alias or opaque reasoning replay.

The same `(capability_id, behavior_id)` pair is reused by models with the same observable outcome. A `pass` exception records a provider limitation compensated by the package: Mistral PDF through OCR remains visible in the profile and selects the ordinary positive PDF scenario. Package unit or integration tests verify OCR routing and page cost outside E2E selection. `behavior_id: null` is invalid, so compensated cases cannot silently lose their positive evidence. Structured registry fields are authoritative for exact supported values, limits, and request-rule sets; behavior IDs and prose classify the exception without becoming a second source for those values.

For first-party models, the explicit `capability_exceptions` field is required even when empty. Every applicable capability with a shared E2E scenario that is absent from that list, or listed with `behavior_id: "pass"`, is tested against its baseline-positive scenario. An undocumented deviation in that shared E2E scope therefore fails its positive check; omission never skips a scenario. An older third-party package that uses the current plugin API may have no profile and can still register and serve requests, but it is **uncertified**: profile-based conformance or E2E selection must raise a clear error. This preserves `OrganizationPlugin` registration compatibility.

**Evidence relationship**: Each capability exercised by a shared E2E test maps to a common baseline-positive scenario. A declared `pass` exception keeps that positive scenario without an additional E2E route because the adapter exposes the baseline behavior. A non-`pass` exception requiring distinct E2E evidence routes its `(capability_id, behavior_id)` pair to one common rejection/deviation scenario or a package-local scenario. The selector rejects missing or duplicate routes within this E2E scope and never parses the free-form `behavior` text. Behavior IDs are model facts rather than pytest node IDs; test locations remain in `tests/capability_scenarios.py`. Other applicable common scenarios remain selected. Always-on scenarios run regardless of the exception list.

## Pricing tier and selected rate set

**Owner**: Existing `PricingTier` in `src/llm_api_adapter/llm_registry/llm_registry.py`; package-owned dynamic rate selection remains external.

| Field | Meaning | Validation |
| --- | --- | --- |
| `up_to_prompt_tokens` | Existing tier boundary | Existing strictly increasing final-open-tier rule |
| `input_per_1m` | Existing standard ordinary-input rate | Existing validated rate rule |
| `output_per_1m` | Existing standard output rate | Existing validated rate rule |
| `cached_input_per_1m` | New optional standard cached-input rate | Finite, nonnegative, exact-model verified; no default from ordinary input |
| `currency` | Existing organization pricing currency | Same currency applies to the selected rate set |

The existing tier selection uses the provider-reported **total input token count**, not only the uncached portion. Static cached rates follow that same tier. A package with dispatch-time or other conditional pricing, notably DeepSeek, selects its verified rate set under its existing provider-specific rules; it must not claim that one registry tier is the complete applicable rate. Shared accounting may consume the already selected rate set without adding a plugin API requirement.

## Usage and response accounting

**Owner**: Common `Usage` and `ChatResponse` in `src/llm_api_adapter/models/responses/chat_response.py`.

| Field | Meaning | Compatibility |
| --- | --- | --- |
| `input_tokens`, `output_tokens`, `total_tokens` | Provider-reported token counts; omitted components in parsed partial usage are `None` | Keep existing names, constructor order, and zero defaults for direct construction; document the `None` correction for partial provider responses |
| `cached_tokens` | Optional provider-confirmed subset of input tokens | Add at the end of `Usage`; never infer from request settings |
| `cost_input` | Aggregate ordinary plus cached input charge | Existing public meaning remains aggregate input cost |
| `cost_output` | Output-token charge | Existing public field |
| `cost_total` | Complete token and separately metered total where priceable | `None` when any incurred required component is unknown |
| `cost_breakdown` | Existing non-token operations such as OCR | Cached input remains a token component, not a non-token line item |

For a valid static cache split, `0 <= cached_tokens <= input_tokens` and both counts are nonnegative integers. If the selected ordinary rate is `r_in`, cached rate is `r_cached`, and the selected output rate is `r_out`:

The public constructor keeps its existing first three positional fields and zero defaults. Provider-parsed partial usage sets each omitted input or output count to `None` before constructing `Usage`; an explicitly reported zero remains `0`. Existing `ChatResponse.from_*` factories must apply the same rule when called directly, including without an adapter. Missing provider input makes `cost_input` and `cost_total` unavailable; missing provider output makes `cost_output` and `cost_total` unavailable. A wholly absent usage object remains absent. An omitted provider total remains `None` unless an existing documented parser computes an exact sum from both confirmed component counts. This public `None` correction requires migration guidance for consumers that perform arithmetic on parsed token counts; direct legacy `Usage(input_tokens, output_tokens, total_tokens)` construction keeps its existing pricing behavior.

```text
cost_input  = (input_tokens - cached_tokens) × r_in
            + cached_tokens × r_cached
cost_output = output_tokens × r_out
cost_total  = cost_input + cost_output + known non-token charges
```

If a distinct cached rate applies but the provider omits, corrupts, or contradicts the cache split, keep only independently confirmed values; `cost_input` and `cost_total` are unavailable. `cost_output` may remain if output usage and its rate are valid. If cached usage is confirmed positive but its rate is unverified, input and total cost are unavailable. An explicit provider-reported `cached_tokens=0` is a confirmed cache miss and may use the ordinary rate. Existing simple pricing remains available for models without a distinct cached rate and without reported cache usage; it must not be presented as a complete discounted estimate when a separately billed cached portion is known. Existing currency-completeness rules for combining token and non-token charges still apply.

Package-specific `Usage` subclasses can inherit the common cached field. All first-party provider parsers, including streaming finalizers, must preserve omitted input/output counts as `None` when constructing common or subclass usage. Provider-specific extra usage such as DeepSeek reasoning tokens remains package-owned. The refactor does not add request parameters for cache control or require a new public cost-breakdown field.

## External organization identity

**Owner**: Existing `KnownOrganizationPackage` map in `src/llm_api_adapter/organization_registry.py`.

| Relationship | Required consistency |
| --- | --- |
| Core known organization key → distribution | Unique, exact known package name and installation guidance |
| Core extra → distribution requirement | Same organization key and normalized distribution name |
| Package directory/manifest/entry point | Package exists; manifest name and entry-point key match the known identity |
| E2E profile | Same organization identity and distribution; profile retains environment/operation settings |
| CI lane selector and release workflow | Explicit lane/job remains present for each external organization; secret access remains job-local |

A repository validation test compares these sources and reports the missing/mismatched source. Runtime Core does not parse repository TOML or import an external package merely to validate packaging. Workflow jobs, package versions, and credentials remain explicit and independently controlled.

## State transitions

1. **Registry load**: Parse existing model metadata and optional new fields. A first-party profile must contain a valid `capability_exceptions` list before release; legacy third-party metadata remains loadable.
2. **Scenario selection**: Resolve one exact model profile. Select baseline-positive evidence for each applicable capability with a shared E2E scenario, including capabilities declared with `behavior_id: "pass"`, or replace it through the documented `(capability_id, behavior_id)` route for a non-`pass` exception requiring distinct E2E evidence. The common tool loop runs once per model with a viable tool-choice mode. A missing profile, invalid behavior ID, or missing route in the shared E2E scope produces a profile error, never a deselection.
3. **Response accounting**: Parse provider usage → validate cached subset and selected rate context → calculate known components → expose a complete total only if every incurred component is known.
4. **Release validation**: Compare package identity sources → run deterministic conformance → use the existing independent post-publish provider lane for authorized live evidence.
