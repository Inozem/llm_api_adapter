# Data Model: Core / Plugin Architecture Refactor (0.9.8)

This document describes the planning-level data contracts. It does not replace the canonical public baseline in `specs/001-baseline-contract/spec.md`.

## Baseline capability catalogue

**Purpose**: Give common scenarios a stable capability identifier without allowing a model profile to disable cross-cutting Core invariants.

| Field | Meaning | Validation |
| --- | --- | --- |
| `id` | Stable identifier derived from one observable baseline behavior or variant | Unique; no undeclared identifier is accepted |
| `scope` | `model-dependent` or `always-on` | Only model-dependent entries receive model decisions |
| `positive_scenarios` | Applicable common scenarios for confirmed support | At least one scenario or an explicit explanation for a non-E2E-only check |
| `exception_scenarios` | Common or package-local checks of documented refusal/special behavior | Required when a model declares an exception |

The initial catalogue covers exact-model decisions for text/chat modes, synchronous and asynchronous streaming, application tools and `tool_choice` variants, portable structured output, supported image/document forms, reasoning controls and events, continuation, usage availability, refusal, and incomplete outcomes. Scenario granularity must preserve meaningful variants such as image URL versus bytes, PDF forms, and restricted tool-choice modes. Facade/discovery, message and error normalization, transport parity/cleanup, pricing correctness, and missing-usage honesty are always-on checks. The complete catalogue is derived from the existing baseline and shared scenario inventory during implementation, then version-controlled and tested.

## Model capability profile

**Owner**: Exact model entry in a built-in or external organization's registry metadata.

| Field | Meaning | Validation |
| --- | --- | --- |
| `model_id` | Existing exact registered model identifier | Existing registry uniqueness and alias rules apply |
| `decisions` | One decision per model-dependent catalogue ID | Complete for first-party models; no unknown key or missing status |
| `status` | `supported` or `exception` | No implicit false or fallback status |
| `exception_behavior` | Exact expected rejection or provider-specific normalized behavior | Required for `exception`; absent for `supported` |
| `variant_limits` | Explicit supported or excluded forms within a capability | Cannot silently broaden a parent decision |

An older third-party package that uses the current plugin API may have no profile and can still register and serve requests. Such a model is **uncertified**, not implicitly unsupported: a profile-based conformance or E2E selector must raise a clear error. This preserves `OrganizationPlugin` registration compatibility while satisfying the requirement that unknown capability status never skips a test. A first-party profile missing a decision fails deterministic validation before release.

**Evidence relationship**: Each `supported` decision maps to common positive scenarios. Each `exception` maps to a common rejection/special-behavior scenario or a package-local scenario. The selector checks that mapping; the registry records model facts rather than test node IDs. Package-local evidence never removes an applicable common scenario.

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

1. **Registry load**: Parse existing model metadata and optional new fields. First-party profile validation must pass before release; legacy third-party metadata remains loadable.
2. **Scenario selection**: Resolve one exact model profile. A complete profile selects positive or exception evidence; a missing/unknown status produces a profile error, never a deselection.
3. **Response accounting**: Parse provider usage → validate cached subset and selected rate context → calculate known components → expose a complete total only if every incurred component is known.
4. **Release validation**: Compare package identity sources → run deterministic conformance → use the existing independent post-publish provider lane for authorized live evidence.
